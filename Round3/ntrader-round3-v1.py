"""
Round 3 Trading Strategy - GOAT (Great Orbital Ascension Trials)
Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_* vouchers (options)

STRATEGY OVERVIEW:
1. HYDROGEL_PACK    -> Pure market making around fair value 10000
2. VELVETFRUIT_EXTRACT -> Market making + delta hedge for options
3. VEV_* VOUCHERS   -> Options market making with BS pricing + delta hedging
   - Focus on VEV_5300, VEV_5400, VEV_5500 (most liquid, real edge)
   - Avoid VEV_6000, VEV_6500 (no edge, bid=0)
   - Deep ITM (4000, 4500) -> trade near parity, limited edge
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle
import math

# ─────────────────────────────────────────────
# BLACK-SCHOLES HELPERS (no scipy needed)
# ─────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation of the standard normal CDF."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    approx = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return approx if x >= 0 else 1.0 - approx


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes European call price. r=0 (no risk-free rate)."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """BS delta of a call option."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return norm_cdf(d1)


# ─────────────────────────────────────────────
# PARAMETERS (calibrated from historical data)
# ─────────────────────────────────────────────

# Implied volatility (stable 23-25% across all strikes & days)
IV = 0.24

# Fair value for HYDROGEL (mean-reverts to 10000)
HYDROGEL_FAIR = 10000

# Option strikes and their labels
VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
}

# Products to actually trade (skip far OTM worthless ones)
ACTIVE_VOUCHERS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]

# Position limits
LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{v: 300 for v in VOUCHER_STRIKES},
}

# Market making parameters
HYDROGEL_SPREAD = 4       # half-spread to post around fair value
VEV_SPREAD = 3            # half-spread for VEV market making
VOUCHER_SPREAD = 1        # half-spread for voucher market making (in option price units)

# How aggressively to size orders (fraction of remaining position capacity)
ORDER_FRACTION = 0.4

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def mid_price(order_depth: OrderDepth) -> float | None:
    """Compute mid price from best bid and ask."""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0


def best_bid(od: OrderDepth) -> float | None:
    return max(od.buy_orders.keys()) if od.buy_orders else None


def best_ask(od: OrderDepth) -> float | None:
    return min(od.sell_orders.keys()) if od.sell_orders else None


def available_volume(od: OrderDepth, side: str, price: float) -> int:
    """Volume available on the book at or better than price."""
    if side == "buy":  # we want to buy, look at asks <= price
        total = sum(v for p, v in od.sell_orders.items() if p <= price)
        return abs(total)
    else:  # we want to sell, look at bids >= price
        total = sum(v for p, v in od.buy_orders.items() if p >= price)
        return total


# ─────────────────────────────────────────────
# MAIN TRADER CLASS
# ─────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        # ── Load persistent state ──────────────────────────────────────
        trader_data = {}
        if state.traderData and state.traderData != "":
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except Exception:
                trader_data = {}

        # Track EMA of VEV spot price for a smooth fair-value estimate
        vev_ema = trader_data.get("vev_ema", None)
        step = trader_data.get("step", 0)

        # ── Determine TTE (time to expiry in years) ───────────────────
        # Round 3 starts at TTE = 5 days. Within a round (~10000 timestamps),
        # time decreases from 5 to 4 days.
        round_progress = min(state.timestamp / 1_000_000, 1.0)  # 0..1 within round
        TTE_days = 5.0 - round_progress  # interpolate from 5 to 4
        TTE = max(TTE_days / 365.0, 1e-6)

        orders: Dict[str, List[Order]] = {}
        positions = state.position

        # ── 1. Estimate VEV fair value via EMA ────────────────────────
        vev_spot = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vev_od = state.order_depths["VELVETFRUIT_EXTRACT"]
            vev_mid = mid_price(vev_od)
            if vev_mid is not None:
                alpha = 0.1
                vev_ema = vev_mid if vev_ema is None else (alpha * vev_mid + (1 - alpha) * vev_ema)
                vev_spot = vev_ema

        # ── 2. HYDROGEL_PACK market making ────────────────────────────
        hyd_orders = self._market_make(
            product="HYDROGEL_PACK",
            fair_value=HYDROGEL_FAIR,
            half_spread=HYDROGEL_SPREAD,
            state=state,
        )
        if hyd_orders:
            orders["HYDROGEL_PACK"] = hyd_orders

        # ── 3. VELVETFRUIT_EXTRACT market making ─────────────────────
        if vev_spot is not None:
            vev_orders = self._market_make(
                product="VELVETFRUIT_EXTRACT",
                fair_value=vev_spot,
                half_spread=VEV_SPREAD,
                state=state,
            )
            if vev_orders:
                orders["VELVETFRUIT_EXTRACT"] = vev_orders

        # ── 4. Options market making + delta hedging ──────────────────
        if vev_spot is not None:
            net_delta = 0.0  # total option delta we accumulate

            for voucher in ACTIVE_VOUCHERS:
                K = VOUCHER_STRIKES[voucher]
                if voucher not in state.order_depths:
                    continue

                fair = bs_call_price(vev_spot, K, TTE, IV)
                delta = bs_delta(vev_spot, K, TTE, IV)

                v_orders = self._options_market_make(
                    product=voucher,
                    fair_value=fair,
                    half_spread=max(VOUCHER_SPREAD, 1),
                    state=state,
                )
                if v_orders:
                    orders[voucher] = v_orders

                # Estimate net option delta from current voucher position
                pos = positions.get(voucher, 0)
                net_delta += pos * delta

            # ── 5. Delta hedge: offset option delta with VEV spot ─────
            # net_delta > 0 means we are long delta -> sell VEV to hedge
            # net_delta < 0 means we are short delta -> buy VEV to hedge
            hedge_target = -round(net_delta)  # target VEV position to neutralize
            vev_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
            vev_limit = LIMITS["VELVETFRUIT_EXTRACT"]

            hedge_delta = hedge_target - vev_pos
            hedge_delta = max(-vev_limit - vev_pos, min(vev_limit - vev_pos, hedge_delta))

            if abs(hedge_delta) >= 1 and vev_spot is not None and "VELVETFRUIT_EXTRACT" in state.order_depths:
                vev_od = state.order_depths["VELVETFRUIT_EXTRACT"]
                hedge_orders = orders.get("VELVETFRUIT_EXTRACT", [])

                if hedge_delta > 0:
                    # Buy VEV to hedge
                    ask = best_ask(vev_od)
                    if ask is not None:
                        hedge_orders.append(Order("VELVETFRUIT_EXTRACT", ask, int(hedge_delta)))
                else:
                    # Sell VEV to hedge
                    bid = best_bid(vev_od)
                    if bid is not None:
                        hedge_orders.append(Order("VELVETFRUIT_EXTRACT", bid, int(hedge_delta)))

                orders["VELVETFRUIT_EXTRACT"] = hedge_orders

        # ── Save state ────────────────────────────────────────────────
        trader_data["vev_ema"] = vev_ema
        trader_data["step"] = step + 1
        new_trader_data = jsonpickle.encode(trader_data)

        conversions = 0
        return orders, conversions, new_trader_data

    # ──────────────────────────────────────────
    # MARKET MAKING HELPER
    # ──────────────────────────────────────────

    def _market_make(
        self,
        product: str,
        fair_value: float,
        half_spread: int,
        state: TradingState,
    ) -> List[Order]:
        """Post competitive bid/ask around fair_value."""
        if product not in state.order_depths:
            return []

        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = LIMITS[product]

        orders: List[Order] = []
        bb = best_bid(od)
        ba = best_ask(od)

        # ── Take aggressive opportunities first ──────────────────────
        # If someone is selling below our fair value -> buy
        if ba is not None and ba < fair_value - 1:
            buy_qty = min(limit - pos, available_volume(od, "buy", fair_value - 1))
            if buy_qty > 0:
                orders.append(Order(product, ba, buy_qty))
                pos += buy_qty

        # If someone is buying above our fair value -> sell
        if bb is not None and bb > fair_value + 1:
            sell_qty = min(pos + limit, available_volume(od, "sell", fair_value + 1))
            if sell_qty > 0:
                orders.append(Order(product, bb, -sell_qty))
                pos -= sell_qty

        # ── Post passive quotes ───────────────────────────────────────
        # Skew quotes based on current position to manage inventory
        skew = int(pos / limit * half_spread)  # nudge fair value against position

        our_bid = int(fair_value) - half_spread - skew
        our_ask = int(fair_value) + half_spread - skew

        # Don't cross the market
        if bb is not None:
            our_bid = min(our_bid, bb)  # don't overpay
        if ba is not None:
            our_ask = max(our_ask, ba)  # don't undersell

        buy_cap = limit - pos
        sell_cap = limit + pos

        bid_size = max(1, int(buy_cap * ORDER_FRACTION))
        ask_size = max(1, int(sell_cap * ORDER_FRACTION))

        if buy_cap > 0 and our_bid > 0:
            orders.append(Order(product, our_bid, bid_size))
        if sell_cap > 0:
            orders.append(Order(product, our_ask, -ask_size))

        return orders

    # ──────────────────────────────────────────
    # OPTIONS MARKET MAKING HELPER
    # ──────────────────────────────────────────

    def _options_market_make(
        self,
        product: str,
        fair_value: float,
        half_spread: int,
        state: TradingState,
    ) -> List[Order]:
        """Market make on an option with BS-derived fair value."""
        if product not in state.order_depths:
            return []

        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = LIMITS[product]

        orders: List[Order] = []
        bb = best_bid(od)
        ba = best_ask(od)

        # ── Aggressive fills: take mispricings ───────────────────────
        # Buy if ask is well below fair value
        if ba is not None and ba < fair_value - half_spread:
            qty = min(limit - pos, abs(sum(v for p, v in od.sell_orders.items() if p <= fair_value - half_spread)))
            if qty > 0:
                orders.append(Order(product, ba, qty))
                pos += qty

        # Sell if bid is well above fair value
        if bb is not None and bb > fair_value + half_spread:
            qty = min(pos + limit, sum(v for p, v in od.buy_orders.items() if p >= fair_value + half_spread))
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                pos -= qty

        # ── Passive quotes ────────────────────────────────────────────
        # Skew toward flat position
        skew = int(pos / limit * half_spread * 0.5)

        our_bid = math.floor(fair_value) - half_spread - skew
        our_ask = math.ceil(fair_value) + half_spread - skew

        # Must be positive prices for options
        if our_bid <= 0:
            our_bid = 1
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        buy_cap = limit - pos
        sell_cap = limit + pos

        bid_size = max(1, int(buy_cap * ORDER_FRACTION))
        ask_size = max(1, int(sell_cap * ORDER_FRACTION))

        if buy_cap > 0:
            orders.append(Order(product, our_bid, bid_size))
        if sell_cap > 0:
            orders.append(Order(product, our_ask, -ask_size))

        return orders
