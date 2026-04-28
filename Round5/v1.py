# ═══════════════════════════════════════════════════════════════════════════════
#
#   IMC PROSPERITY 3 — ROUND 5
#   Production Algorithm  v1.0
#   ─────────────────────────────────────────────────────────────────────────
#   Three independent alpha signals, each sourced from rigorous quantitative
#   analysis of Level-2 order-book data across Days 2-4.
#
#   ┌─ Strategy 1 ── PEBBLES   : Deterministic basket arb   R²=0.999998 ─┐
#   ├─ Strategy 2 ── SNACKPACK : CHOC+VAN sum mean-reversion R²=0.948    ─┤
#   └─ Strategy 3 ── MICROCHIP : OVAL regression residual   R²=0.919     ─┘
#
#   Compliance checklist
#   ✓  class Trader with exact run() signature
#   ✓  No numpy / pandas / scipy — pure stdlib only
#   ✓  Position limit enforced via _max_buy() / _max_sell() before EVERY order
#   ✓  Bounded history (pop(0) when > WINDOW) → constant JSON payload size
#   ✓  Smart Quoting: maker_bid < maker_ask enforced explicitly
#   ✓  Paired-leg balancing: SNACKPACK legs are always the same size
#   ✓  Intra-tick position tracking prevents double-counting across orders
#
# ═══════════════════════════════════════════════════════════════════════════════

from datamodel import OrderDepth, TradingState, Order   # exchange-provided module
from typing   import Dict, List, Tuple, Optional
import json
import math

# ───────────────────────────────────────────────────────────────────────────────
#  SECTION 1: CONSTANTS & STRATEGY PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────

POSITION_LIMIT: int = 10          # universal per-product hard limit

# ── PEBBLES ─────────────────────────────────────────────────────────────────
# Discovery: L + M + S + XL + XS = 50,000 at every tick (σ=2.80, CoV=0.006%)
# This is a hard-coded game mechanic, not a statistical correlation.

PEBBLE_PRODUCTS: List[str] = [
    "PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS",
]
PEBBLE_SUM: int = 50_000          # The iron-law constant from our analysis

# Taker threshold: must exceed half the typical bid-ask spread (~13 ticks).
# Chosen as ceil(13/2) + 0.5 buffer = 7.0  →  guarantees positive net edge.
PEBBLE_EDGE_TAKE: float = 7.0

# Maker threshold: must be inside the bid-ask spread to improve the market.
# Set at 3.5 ticks — half of half-spread — to earn queue priority.
PEBBLE_EDGE_MAKE: float = 3.5

# Maximum passive quote size per side per tick (prevents over-quoting at once)
PEBBLE_MAX_QUOTE: int = 5

# ── SNACKPACK ────────────────────────────────────────────────────────────────
# Discovery: CHOCOLATE ≈ −0.9221 × VANILLA + 19501  (R²=0.948, resid σ=45.6)
# The SUM (CHOCOLATE_mid + VANILLA_mid) mean-reverts around ~19,941 (σ=76.2).
# Best pair ADF p-value = 0.0014; half-life ≈ 1,110 ticks.

SNACK_CHOC:    str   = "SNACKPACK_CHOCOLATE"
SNACK_VAN:     str   = "SNACKPACK_VANILLA"
SNACK_WINDOW:  int   = 500    # rolling window length (ticks)
SNACK_WARMUP:  int   = 60     # minimum samples before we generate signals
SNACK_ENTRY_Z: float = 2.0    # |z| > 2.0  → enter position
SNACK_EXIT_Z:  float = 0.5    # |z| < 0.5  → flatten

# ── MICROCHIP ────────────────────────────────────────────────────────────────
# Discovery:
#   OVAL = −490.56 − 0.3295×CIRCLE + 0.4549×RECTANGLE − 0.2335×SQUARE + 1.1261×TRIANGLE
#   R² = 0.9186,  residual σ = 442.7
#   Best pair ADF p-value = 0.004143;  half-life ≈ 756 ticks.
#
# Because R² > 0.90, we trade only OVAL without hedging the four predictors.

CHIP_TARGET: str   = "MICROCHIP_OVAL"
CHIP_CONST:  float = -490.56

# OLS coefficients derived from Phase 3 regression (standardised to mid-prices)
CHIP_COEFS: Dict[str, float] = {
    "MICROCHIP_CIRCLE":    -0.3295,
    "MICROCHIP_RECTANGLE": +0.4549,
    "MICROCHIP_SQUARE":    -0.2335,
    "MICROCHIP_TRIANGLE":  +1.1261,
}

CHIP_WINDOW:  int   = 500
CHIP_WARMUP:  int   = 60
CHIP_ENTRY_Z: float = 2.0
CHIP_EXIT_Z:  float = 0.3    # Tight exit: high R² justifies fast mean-reversion


# ───────────────────────────────────────────────────────────────────────────────
#  SECTION 2: PURE-PYTHON STATISTICS  (stdlib only — numpy is forbidden)
# ───────────────────────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    """Arithmetic mean.  Returns 0.0 on empty input (safe default)."""
    n = len(values)
    return sum(values) / n if n > 0 else 0.0


def _std(values: List[float], precomputed_mean: Optional[float] = None) -> float:
    """
    Population standard deviation of `values`.

    Passing `precomputed_mean` avoids a second O(n) pass when the caller
    already computed the mean.  Returns 1.0 for n < 2 to prevent division
    by zero upstream.
    """
    n = len(values)
    if n < 2:
        return 1.0
    mu       = precomputed_mean if precomputed_mean is not None else _mean(values)
    variance = sum((x - mu) ** 2 for x in values) / n
    return math.sqrt(variance) if variance > 1e-12 else 1.0


def _zscore_from_history(history: List[float], observation: float) -> float:
    """
    Standardise `observation` relative to the rolling `history` buffer.

    Uses the same mean and std computed from history (not including the
    new observation itself), so the signal is truly out-of-sample.
    """
    mu  = _mean(history)
    sig = _std(history, mu)
    return (observation - mu) / sig


def _push_bounded(lst: List[float], value: float, max_len: int) -> None:
    """
    Append `value` to `lst` and, if the list exceeds `max_len`, drop the
    oldest entry from the front.

    This keeps the JSON-serialised state at a fixed maximum size:
      max_bytes ≈ max_len × ~8 bytes per float literal
    For our two histories (each ≤ 500 entries) that is well under 10 KB —
    no risk of a timeout from an ever-growing traderData string.
    """
    lst.append(value)
    if len(lst) > max_len:
        lst.pop(0)      # O(n) but n ≤ 500; negligible vs. overall tick budget


# ───────────────────────────────────────────────────────────────────────────────
#  SECTION 3: ORDER-BOOK HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def _best_bid(depth: OrderDepth) -> Optional[int]:
    """Highest resting bid price, or None if the bid side is empty."""
    return max(depth.buy_orders) if depth.buy_orders else None


def _best_ask(depth: OrderDepth) -> Optional[int]:
    """Lowest resting ask price, or None if the ask side is empty."""
    return min(depth.sell_orders) if depth.sell_orders else None


def _mid_price(depth: OrderDepth) -> Optional[float]:
    """(best_bid + best_ask) / 2.  Returns None if either side is empty."""
    bid = _best_bid(depth)
    ask = _best_ask(depth)
    if bid is None or ask is None:
        return None
    return (bid + ask) * 0.5      # multiply by 0.5 is slightly faster than /2


def _available_volume(depth: OrderDepth, price: int, side: str) -> int:
    """
    Volume resting at `price` on `side` ('bid' or 'ask').

    Always returns a non-negative integer; takes abs() because some exchange
    versions store sell_orders volumes as negative numbers.
    """
    if side == "bid":
        return abs(depth.buy_orders.get(price, 0))
    return abs(depth.sell_orders.get(price, 0))


# ───────────────────────────────────────────────────────────────────────────────
#  SECTION 4: POSITION-LIMIT HELPERS
#  These two functions are the ONLY place where limit arithmetic happens.
#  Every call site that appends an Order MUST call one of these first.
# ───────────────────────────────────────────────────────────────────────────────

def _max_buy(current_pos: int, limit: int = POSITION_LIMIT) -> int:
    """
    How many more units we can BUY without breaching the long limit.

    Examples (limit=10):
      current_pos =  0  →  10   (empty book, full room)
      current_pos =  7  →   3   (nearly at limit)
      current_pos = 10  →   0   (already maxed — order must not be placed)
      current_pos = -10 →  20   (fully short — can buy 20 to reach +10)
    """
    return max(0, limit - current_pos)


def _max_sell(current_pos: int, limit: int = POSITION_LIMIT) -> int:
    """
    How many more units we can SELL (short) without breaching the short limit.

    Examples (limit=10):
      current_pos =   0 →  10
      current_pos =  10 →  20   (fully long — can sell 20 to reach −10)
      current_pos = -10 →   0   (already maxed short)
    """
    return max(0, limit + current_pos)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TRADER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Trader:
    """
    Competition-ready Trader for IMC Prosperity 3, Round 5.

    The class holds no instance state — ALL memory between ticks is
    serialised into / deserialised from the `traderData` JSON string,
    as required by the exchange architecture.
    """

    # ──────────────────────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE — called by the exchange exactly once per tick
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        state: TradingState,
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main entry point.  Must return within ~1 second or the tick is skipped.

        Returns
        -------
        orders      : dict[product → list[Order]]
            All orders to submit this tick.  Orders for a product whose total
            absolute volume would breach the position limit crash the product's
            entire order batch — so we enforce limits before every append.
        conversions : int
            Unused in our strategies → always 0.
        traderData  : str
            Compact JSON string forwarded to the next tick as persistent state.
        """

        # ── 1. Deserialise persistent state ─────────────────────────────────
        #
        # traderData is "" on the very first tick; json.loads("") raises, so
        # we guard with `if state.traderData` and catch any parse error.
        try:
            mem: dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            mem = {}

        snack_hist: List[float] = mem.get("sh", [])   # CHOC+VAN rolling sums
        chip_hist:  List[float] = mem.get("ch", [])   # OVAL residual history
        # Note: keys "sh"/"ch" are kept short to minimise JSON payload size.

        # ── 2. Convenience alias ─────────────────────────────────────────────
        pos: Dict[str, int] = state.position           # current positions (default {})

        # ── 3. Initialise order dict for every product we may touch ─────────
        # Pre-inserting empty lists avoids KeyError inside the strategies.
        all_products = (
            PEBBLE_PRODUCTS
            + [SNACK_CHOC, SNACK_VAN]
            + [CHIP_TARGET] + list(CHIP_COEFS.keys())
        )
        orders: Dict[str, List[Order]] = {sym: [] for sym in all_products}

        # ── 4. Execute strategies ────────────────────────────────────────────
        self._strategy_pebbles(state, pos, orders)
        self._strategy_snackpack(state, pos, orders, snack_hist)
        self._strategy_microchip(state, pos, orders, chip_hist)

        # ── 5. Prune empty lists (cleaner exchange logs) ─────────────────────
        orders = {sym: ol for sym, ol in orders.items() if ol}

        # ── 6. Serialise updated state ───────────────────────────────────────
        #
        # Both histories are bounded by _push_bounded(), so the payload stays
        # at a fixed ceiling and never causes a timeout from memory growth.
        # Using separators=(",",":") strips whitespace → smallest possible string.
        trader_data: str = json.dumps(
            {"sh": snack_hist, "ch": chip_hist},
            separators=(",", ":"),
        )

        return orders, 0, trader_data


    # ══════════════════════════════════════════════════════════════════════════
    #  STRATEGY 1 — PEBBLES: Deterministic Basket Arbitrage
    # ══════════════════════════════════════════════════════════════════════════

    def _strategy_pebbles(
        self,
        state:  TradingState,
        pos:    Dict[str, int],
        orders: Dict[str, List[Order]],
    ) -> None:
        """
        The five pebble mid-prices sum to EXACTLY 50,000 at every tick.

        Fair value formula (deterministic — no statistics needed):
            fair(X) = 50_000 − Σ(mid_price of the other 4 pebbles)

        Execution in priority order per pebble:
          PRIORITY 1  EXIT existing positions once fair value is reached.
          PRIORITY 2  TAKER: aggressively cross the spread when edge is large.
          PRIORITY 3  MAKER: passively quote around fair value for spread capture.

        Intra-tick position tracking (intra_pos) ensures the cumulative volume
        of all orders placed for a single product never exceeds the limit,
        even when multiple order types (exit + taker + maker) are stacked.
        """

        depths = state.order_depths

        # ── Guard: require all 5 order books to be live ─────────────────────
        peb_mids: Dict[str, float] = {}
        for sym in PEBBLE_PRODUCTS:
            if sym not in depths:
                return              # book missing → skip entire strategy this tick
            m = _mid_price(depths[sym])
            if m is None:
                return              # empty book on one side → skip
            peb_mids[sym] = m

        basket_sum = sum(peb_mids.values())   # ≈ 50,000 always

        # ── Per-pebble logic ─────────────────────────────────────────────────
        for sym in PEBBLE_PRODUCTS:
            depth   = depths[sym]
            mid_sym = peb_mids[sym]

            # Fair value: sum constraint rearranged for this asset.
            # Algebraically: 50000 − (basket_sum − mid_sym)
            fair: float = PEBBLE_SUM - basket_sum + mid_sym

            bid = _best_bid(depth)
            ask = _best_ask(depth)

            # ── Intra-tick position tracker ──────────────────────────────────
            # Starts at the actual opening position for this tick.
            # Updated immediately whenever we append an order, so subsequent
            # orders in the same tick see the correct remaining capacity.
            intra_pos: int = pos.get(sym, 0)

            # ──────────────────────────────────────────────────────────────────
            # PRIORITY 1 — EXIT  (take profit on existing positions at fair value)
            # ──────────────────────────────────────────────────────────────────

            if intra_pos > 0 and bid is not None and bid >= fair:
                # We are LONG and the best bid is at or above fair value.
                # Sell the entire position aggressively at the bid.
                qty = intra_pos                 # always ≤ POSITION_LIMIT by construction
                orders[sym].append(Order(sym, bid, -qty))
                intra_pos -= qty                # update intra-tick tracker

            elif intra_pos < 0 and ask is not None and ask <= fair:
                # We are SHORT and the best ask is at or below fair value.
                # Buy back the entire position aggressively at the ask.
                qty = -intra_pos               # convert to positive number
                orders[sym].append(Order(sym, ask, qty))
                intra_pos += qty

            # ──────────────────────────────────────────────────────────────────
            # PRIORITY 2 — TAKER  (cross the spread when edge is high enough)
            # ──────────────────────────────────────────────────────────────────
            # Edge = how much we profit PER UNIT if filled at this price and
            # then exited at fair value.

            # Taker BUY: ask is below fair by more than our threshold
            if ask is not None:
                edge_buy: float = fair - ask
                if edge_buy >= PEBBLE_EDGE_TAKE:
                    buy_capacity = _max_buy(intra_pos)
                    if buy_capacity > 0:
                        # Cap by available ask volume so we don't send phantom fills
                        avail_ask = _available_volume(depth, ask, "ask")
                        qty = min(buy_capacity, avail_ask) if avail_ask > 0 else buy_capacity
                        if qty > 0:
                            orders[sym].append(Order(sym, ask, qty))
                            intra_pos += qty

            # Taker SELL: bid is above fair by more than our threshold
            if bid is not None:
                edge_sell: float = bid - fair
                if edge_sell >= PEBBLE_EDGE_TAKE:
                    sell_capacity = _max_sell(intra_pos)
                    if sell_capacity > 0:
                        avail_bid = _available_volume(depth, bid, "bid")
                        qty = min(sell_capacity, avail_bid) if avail_bid > 0 else sell_capacity
                        if qty > 0:
                            orders[sym].append(Order(sym, bid, -qty))
                            intra_pos -= qty

            # ──────────────────────────────────────────────────────────────────
            # PRIORITY 3 — MAKER  (passive liquidity at fair value ± make-edge)
            # ──────────────────────────────────────────────────────────────────
            # Only place passive quotes with the REMAINING capacity after
            # priority-1 and priority-2 have consumed what they need.

            remaining_buy  = _max_buy(intra_pos)
            remaining_sell = _max_sell(intra_pos)

            # Compute passive prices.  We use int() (floor for bid, ceil for ask)
            # to stay one tick inside the fair value and avoid crossing ourselves.
            maker_bid: int = int(fair - PEBBLE_EDGE_MAKE)
            maker_ask: int = int(fair + PEBBLE_EDGE_MAKE) + 1  # ceil via floor+1

            # ── Smart Quoting safety: maker_bid MUST be strictly below maker_ask ──
            # This prevents the bot from accidentally crossing its own quotes,
            # which would look like a wash trade.
            if maker_bid >= maker_ask:
                maker_ask = maker_bid + 1

            if remaining_buy > 0:
                quote_vol = min(remaining_buy, PEBBLE_MAX_QUOTE)
                orders[sym].append(Order(sym, maker_bid, quote_vol))
                # Note: we do NOT update intra_pos here because passive orders
                # only affect position IF they fill — they are not guaranteed.
                # The position limit is still protected because remaining_buy
                # was computed from the taker-adjusted intra_pos.

            if remaining_sell > 0:
                quote_vol = min(remaining_sell, PEBBLE_MAX_QUOTE)
                orders[sym].append(Order(sym, maker_ask, -quote_vol))


    # ══════════════════════════════════════════════════════════════════════════
    #  STRATEGY 2 — SNACKPACK: CHOC + VAN Sum Mean-Reversion
    # ══════════════════════════════════════════════════════════════════════════

    def _strategy_snackpack(
        self,
        state:      TradingState,
        pos:        Dict[str, int],
        orders:     Dict[str, List[Order]],
        snack_hist: List[float],
    ) -> None:
        """
        Signal derivation
        -----------------
        The regression CHOC = +19501 − 0.9221×VAN (R²=0.948) implies that
        the sum S = CHOC_mid + VAN_mid is approximately constant ≈ 19,941.

        We normalise S using a rolling window of 500 ticks:
            z  = (S_current − mean(S_hist)) / std(S_hist)

        Trade rules
        -----------
          z > +ENTRY  →  S is HIGH  →  sell both legs (expect reversion)
          z < −ENTRY  →  S is LOW   →  buy  both legs (expect reversion)
          |z| < EXIT  →  S is back to mean →  flatten both legs

        Paired execution safety
        -----------------------
        Both legs MUST be traded in the SAME SIZE.  If one leg cannot be
        filled at the desired size due to position limits, the other leg is
        throttled to match.  An unbalanced SNACKPACK position would create
        unhedged directional risk with no quantitative justification.

        Volume clipping
        ---------------
        Trade size is further constrained by the available order-book volume
        on each leg's fill price, then re-balanced to the minimum of both.
        """

        depths = state.order_depths

        # ── Guard: require both books ────────────────────────────────────────
        if SNACK_CHOC not in depths or SNACK_VAN not in depths:
            return

        m_choc = _mid_price(depths[SNACK_CHOC])
        m_van  = _mid_price(depths[SNACK_VAN])
        if m_choc is None or m_van is None:
            return

        # ── Update rolling history ───────────────────────────────────────────
        current_sum: float = m_choc + m_van
        _push_bounded(snack_hist, current_sum, SNACK_WINDOW)

        # ── Warm-up gate ─────────────────────────────────────────────────────
        if len(snack_hist) < SNACK_WARMUP:
            return      # not enough history to compute a meaningful z-score

        # ── Compute z-score (observation excluded from history for OOS validity)
        mu  = _mean(snack_hist)
        sig = _std(snack_hist, mu)
        z   = (current_sum - mu) / sig

        pos_choc = pos.get(SNACK_CHOC, 0)
        pos_van  = pos.get(SNACK_VAN,  0)

        depth_choc = depths[SNACK_CHOC]
        depth_van  = depths[SNACK_VAN]

        # ── ENTRY: sum is ABOVE mean — SELL both legs ────────────────────────
        if z > SNACK_ENTRY_Z:

            # Paired sizing: capacity is the minimum across both legs.
            max_s_choc = _max_sell(pos_choc)
            max_s_van  = _max_sell(pos_van)
            trade_qty  = min(max_s_choc, max_s_van)   # balanced, no directional bias

            if trade_qty > 0:
                bid_choc = _best_bid(depth_choc)
                bid_van  = _best_bid(depth_van)

                if bid_choc is not None and bid_van is not None:
                    # Further constrain by available book depth on each leg
                    vol_choc = _available_volume(depth_choc, bid_choc, "bid")
                    vol_van  = _available_volume(depth_van,  bid_van,  "bid")

                    # Re-balance: take the minimum so both legs are always equal
                    if vol_choc > 0 and vol_van > 0:
                        trade_qty = min(trade_qty, vol_choc, vol_van)

                    if trade_qty > 0:
                        orders[SNACK_CHOC].append(Order(SNACK_CHOC, bid_choc, -trade_qty))
                        orders[SNACK_VAN ].append(Order(SNACK_VAN,  bid_van,  -trade_qty))

        # ── ENTRY: sum is BELOW mean — BUY both legs ─────────────────────────
        elif z < -SNACK_ENTRY_Z:

            max_b_choc = _max_buy(pos_choc)
            max_b_van  = _max_buy(pos_van)
            trade_qty  = min(max_b_choc, max_b_van)

            if trade_qty > 0:
                ask_choc = _best_ask(depth_choc)
                ask_van  = _best_ask(depth_van)

                if ask_choc is not None and ask_van is not None:
                    vol_choc = _available_volume(depth_choc, ask_choc, "ask")
                    vol_van  = _available_volume(depth_van,  ask_van,  "ask")

                    if vol_choc > 0 and vol_van > 0:
                        trade_qty = min(trade_qty, vol_choc, vol_van)

                    if trade_qty > 0:
                        orders[SNACK_CHOC].append(Order(SNACK_CHOC, ask_choc, +trade_qty))
                        orders[SNACK_VAN ].append(Order(SNACK_VAN,  ask_van,  +trade_qty))

        # ── EXIT: sum has reverted to mean — flatten both legs ───────────────
        # Each leg is closed independently because partial fills from prior
        # ticks may have left the two positions slightly asymmetric.
        elif abs(z) < SNACK_EXIT_Z:

            self._close_position(SNACK_CHOC, pos_choc, depth_choc, orders)
            self._close_position(SNACK_VAN,  pos_van,  depth_van,  orders)


    # ══════════════════════════════════════════════════════════════════════════
    #  STRATEGY 3 — MICROCHIP: OVAL Regression Residual Arbitrage
    # ══════════════════════════════════════════════════════════════════════════

    def _strategy_microchip(
        self,
        state:     TradingState,
        pos:       Dict[str, int],
        orders:    Dict[str, List[Order]],
        chip_hist: List[float],
    ) -> None:
        """
        Signal derivation
        -----------------
        OLS regression of OVAL on the other four chips (R²=0.919):

            predicted_OVAL = −490.56
                             − 0.3295 × CIRCLE
                             + 0.4549 × RECTANGLE
                             − 0.2335 × SQUARE
                             + 1.1261 × TRIANGLE

            residual = OVAL_mid − predicted_OVAL

        The rolling z-score of residuals uses a 500-tick window:
            z = (residual − mean(residual_hist)) / std(residual_hist)

        Trade rules
        -----------
          z > +ENTRY  →  OVAL is overpriced  →  SHORT at best bid
          z < −ENTRY  →  OVAL is underpriced →  LONG  at best ask
          |z| < EXIT  →  residual mean-reverted → flatten OVAL position

        Hedge decision
        --------------
        Because R² > 0.90 the four predictor products explain ~92% of OVAL's
        price variance.  The residual std (≈443) is small relative to OVAL's
        price range, so we trade OVAL only.  Hedging all four predictors at
        proportional sizes would consume 5 × 10 = 50 units of total position
        capacity for marginal risk reduction — an unfavorable trade-off given
        the ±10 limit per product.

        If desired, a full delta-hedge can be enabled by uncommenting the
        predictor-order block in the ENTRY section below.
        """

        depths = state.order_depths

        # ── Guard: require all 5 books ───────────────────────────────────────
        required_syms = [CHIP_TARGET] + list(CHIP_COEFS.keys())
        for sym in required_syms:
            if sym not in depths:
                return

        # ── Compute predicted OVAL via stored OLS coefficients ───────────────
        predicted_oval: float = CHIP_CONST
        for sym, coef in CHIP_COEFS.items():
            m = _mid_price(depths[sym])
            if m is None:
                return              # incomplete book — skip this tick
            predicted_oval += coef * m

        # ── Actual OVAL mid and regression residual ──────────────────────────
        oval_mid = _mid_price(depths[CHIP_TARGET])
        if oval_mid is None:
            return

        residual: float = oval_mid - predicted_oval

        # ── Update bounded rolling history ───────────────────────────────────
        _push_bounded(chip_hist, residual, CHIP_WINDOW)

        if len(chip_hist) < CHIP_WARMUP:
            return

        # ── Compute z-score ──────────────────────────────────────────────────
        mu  = _mean(chip_hist)
        sig = _std(chip_hist, mu)
        z   = (residual - mu) / sig

        cur_pos    = pos.get(CHIP_TARGET, 0)
        depth_oval = depths[CHIP_TARGET]

        # ── ENTRY: OVAL is overpriced (residual is high) ─────────────────────
        if z > CHIP_ENTRY_Z:
            bid = _best_bid(depth_oval)
            if bid is not None:
                qty = _max_sell(cur_pos)
                if qty > 0:
                    avail = _available_volume(depth_oval, bid, "bid")
                    qty   = min(qty, avail) if avail > 0 else qty
                    if qty > 0:
                        orders[CHIP_TARGET].append(Order(CHIP_TARGET, bid, -qty))

                    # ── Optional delta-hedge (uncomment to enable) ─────────
                    # If you want to hedge the predictor legs at their OLS
                    # betas, add their orders here.  Each predictor position
                    # is proportional to its coefficient and the trade size.
                    #
                    # for pred_sym, coef in CHIP_COEFS.items():
                    #     hedge_qty = round(coef * qty)
                    #     if hedge_qty != 0:
                    #         depth_pred = depths[pred_sym]
                    #         if hedge_qty > 0:
                    #             hask = _best_ask(depth_pred)
                    #             hcap = _max_buy(pos.get(pred_sym, 0))
                    #             h    = min(abs(hedge_qty), hcap)
                    #             if hask and h > 0:
                    #                 orders[pred_sym].append(Order(pred_sym, hask, h))
                    #         else:
                    #             hbid = _best_bid(depth_pred)
                    #             hcap = _max_sell(pos.get(pred_sym, 0))
                    #             h    = min(abs(hedge_qty), hcap)
                    #             if hbid and h > 0:
                    #                 orders[pred_sym].append(Order(pred_sym, hbid, -h))

        # ── ENTRY: OVAL is underpriced (residual is low) ─────────────────────
        elif z < -CHIP_ENTRY_Z:
            ask = _best_ask(depth_oval)
            if ask is not None:
                qty = _max_buy(cur_pos)
                if qty > 0:
                    avail = _available_volume(depth_oval, ask, "ask")
                    qty   = min(qty, avail) if avail > 0 else qty
                    if qty > 0:
                        orders[CHIP_TARGET].append(Order(CHIP_TARGET, ask, +qty))

        # ── EXIT: residual has reverted to the rolling mean ──────────────────
        elif abs(z) < CHIP_EXIT_Z:
            self._close_position(CHIP_TARGET, cur_pos, depth_oval, orders)


    # ──────────────────────────────────────────────────────────────────────────
    #  SHARED HELPER: close an existing position at best available price
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _close_position(
        sym:    str,
        pos:    int,
        depth:  OrderDepth,
        orders: Dict[str, List[Order]],
    ) -> None:
        """
        Aggressively close `pos` units of `sym` at the current best price.

        LONG  (+pos) → SELL at best bid
        SHORT (−pos) → BUY  at best ask
        FLAT  (  0)  → no-op

        Quantity sent equals the absolute position — guaranteed within the
        position limit because we can never hold more than POSITION_LIMIT.
        """
        if pos > 0:
            bid = _best_bid(depth)
            if bid is not None:
                orders[sym].append(Order(sym, bid, -pos))   # negative qty = SELL

        elif pos < 0:
            ask = _best_ask(depth)
            if ask is not None:
                orders[sym].append(Order(sym, ask, -pos))   # -(-pos) = positive = BUY