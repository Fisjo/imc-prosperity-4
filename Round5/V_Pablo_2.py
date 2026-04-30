from __future__ import annotations

from datamodel import Order, OrderDepth, TradingState

# ---------------------------------------------------------------------------
# MICROCHIP UNIVERSE
# ---------------------------------------------------------------------------
MICROCHIP_PRODUCTS = {
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
}

# ---------------------------------------------------------------------------
# MICROCHIP CONFIGURATIONS
# ---------------------------------------------------------------------------

# Ordinary Least Squares (OLS) fair value adjustments
# Format: {target: (intercept, {contributor: coef}, price_clip)}
MICRO_OLS = {
    "MICROCHIP_CIRCLE": (
        14328.038892,
        {
            "MICROCHIP_OVAL": -0.2144216226,
            "MICROCHIP_RECTANGLE": 0.1239698234,
            "MICROCHIP_SQUARE": -0.1626194856,
            "MICROCHIP_TRIANGLE": -0.2303292061,
        },
        6.0,
    ),
    "MICROCHIP_RECTANGLE": (
        12595.888647,
        {
            "MICROCHIP_CIRCLE": 0.1056072980,
            "MICROCHIP_OVAL": 0.2521946893,
            "MICROCHIP_SQUARE": -0.2712089572,
            "MICROCHIP_TRIANGLE": -0.3316449080,
        },
        6.0,
    ),
    "MICROCHIP_TRIANGLE": (
        8897.550320,
        {
            "MICROCHIP_CIRCLE": -0.1771853113,
            "MICROCHIP_OVAL": 0.5638225952,
            "MICROCHIP_RECTANGLE": -0.2994843207,
            "MICROCHIP_SQUARE": 0.0312598043,
        },
        6.0,
    ),
}

# Structural priors — constant fair value lean
MICRO_PRIORS = {
    "MICROCHIP_OVAL": -0.35,
    "MICROCHIP_TRIANGLE": -0.15,
}

# Lead-lag signals
# Format: (leader, follower, sign)
MICRO_LEAD_LAG = (
    ("MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE", -1.0),
)

# Trend overlay
# Format: {product: {"threshold": float, "min_ticks": int}}
TREND_OVERLAY = {
    "MICROCHIP_SQUARE": {"threshold": 300.0, "min_ticks": 400},
}

# Open momentum overlay
# Format: {product: {"k": float, "th": float}}
OPEN_MOM_OVERLAY = {
    "MICROCHIP_OVAL": {"k": 1000.0, "th": 0.0},
}

# ---------------------------------------------------------------------------
# GLOBAL MARKET-MAKER PARAMETERS
# ---------------------------------------------------------------------------
DEFAULT_LIMIT = 10
EDGE = 1


# ---------------------------------------------------------------------------
# MICROCHIP HELPERS
# ---------------------------------------------------------------------------

def _get_best_market(
    depth: OrderDepth,
) -> tuple[int | None, int | None, int, int]:
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    if best_bid is None or best_ask is None:
        return (None, None, 0, 0)
    bid_vol = sum(v for v in depth.buy_orders.values() if v > 0)
    ask_vol = sum(abs(v) for v in depth.sell_orders.values() if v < 0)
    return (best_bid, best_ask, bid_vol, ask_vol)


def compute_microchip_fair(
    depth: OrderDepth,
    mid: float,
    score: float,
    prior: float,
    pos: int,
    kalman_x: float,
) -> float:
    best_bid, best_ask, bid_vol, ask_vol = _get_best_market(depth)
    if not best_bid or not best_ask or bid_vol + ask_vol <= 0:
        return mid

    # Volume-weighted micro-price
    micro = (best_ask * bid_vol + best_bid * ask_vol) / (bid_vol + ask_vol)
    spread = best_ask - best_bid

    inventory_skew = 0.28 * pos
    signal_shift = max(-4.0, min(4.0, 1.15 * score + prior))
    imbalance_shift = max(
        -2.0,
        min(
            2.0,
            (bid_vol - ask_vol) / max(1, bid_vol + ask_vol) * spread * 0.35,
        ),
    )

    return (
        (0.52 * mid)
        + (0.43 * micro)
        + (0.05 * kalman_x)
        + signal_shift
        + imbalance_shift
        - inventory_skew
    )


def apply_micro_ols_adjustments(
    mids: dict[str, float],
    adjs: dict[str, float],
) -> dict[str, float]:
    for product, (intercept, coefs, clip) in MICRO_OLS.items():
        if product not in mids or not all(o in mids for o in coefs):
            continue
        predicted = intercept + sum(c * mids[o] for o, c in coefs.items())
        diff = predicted - mids[product]
        adjs[product] = clip if diff > 0 else (-clip if diff < 0 else 0.0)
    return adjs


# ---------------------------------------------------------------------------
# TRADER
# ---------------------------------------------------------------------------

class Trader:
    def __init__(
        self,
        *,
        limit: int = DEFAULT_LIMIT,
        edge: int = EDGE,
    ) -> None:
        self.limit = int(limit)
        self.edge = int(edge)
        if self.limit <= 0:
            raise ValueError("limit must be positive.")
        if self.edge < 0:
            raise ValueError("edge must be non-negative.")

        # Persistent state for Microchip signal computation
        self.history: dict[str, list[float]] = {}
        self.open_prices: dict[str, float] = {}

    def run(
        self, state: TradingState
    ) -> tuple[dict[str, list[Order]], int, str]:
        result_orders: dict[str, list[Order]] = {}

        # ------------------------------------------------------------------
        # 1. Update Microchip history and open prices
        # ------------------------------------------------------------------
        mids: dict[str, float] = {}
        for p in MICROCHIP_PRODUCTS:
            if p not in state.order_depths:
                continue
            mid = _midprice(state.order_depths[p])
            if mid is None:
                continue
            mids[p] = mid
            hist = self.history.setdefault(p, [])
            hist.append(mid)
            if len(hist) > 405:
                hist.pop(0)
            if p not in self.open_prices:
                self.open_prices[p] = mid

        # ------------------------------------------------------------------
        # 2. Compute Microchip scores
        # ------------------------------------------------------------------
        scores: dict[str, float] = {p: 0.0 for p in MICROCHIP_PRODUCTS}
        apply_micro_ols_adjustments(mids, scores)

        # Lead-lag: adj = sign * (leader[-1] - leader[-2])
        for leader, follower, sign in MICRO_LEAD_LAG:
            leader_hist = self.history.get(leader, [])
            if len(leader_hist) >= 2:
                adj = sign * (leader_hist[-1] - leader_hist[-2])
                scores[follower] = scores.get(follower, 0.0) + adj

        # Trend overlay: binary ±1 if price moved >= threshold over min_ticks
        for product, cfg in TREND_OVERLAY.items():
            hist = self.history.get(product, [])
            min_ticks = cfg["min_ticks"]
            threshold = cfg["threshold"]
            if len(hist) >= min_ticks:
                delta = hist[-1] - hist[-min_ticks]
                if abs(delta) >= threshold:
                    scores[product] = scores.get(product, 0.0) + (
                        1.0 if delta > 0 else -1.0
                    )

        # Open momentum: score += (mid - open_price) / k  when move >= th
        for product, cfg in OPEN_MOM_OVERLAY.items():
            k = cfg["k"]
            th = cfg["th"]
            mid = mids.get(product)
            open_p = self.open_prices.get(product)
            if mid is not None and open_p is not None and abs(mid - open_p) >= th:
                scores[product] = scores.get(product, 0.0) + (mid - open_p) / k

        # ------------------------------------------------------------------
        # 3. Universal order generation loop
        # ------------------------------------------------------------------
        for product, depth in state.order_depths.items():
            position = int(state.position.get(product, 0))

            if product in MICROCHIP_PRODUCTS and product in mids:
                score = scores.get(product, 0.0)
                prior = MICRO_PRIORS.get(product, 0.0)
                fair_price: float | None = compute_microchip_fair(
                    depth, mids[product], score, prior, position, kalman_x=0.0
                )
            else:
                fair_price = _midprice(depth)

            product_orders = _orders_for_product(
                product=product,
                depth=depth,
                position=position,
                limit=self.limit,
                edge=self.edge,
                fair_price=fair_price,
            )
            if product_orders:
                result_orders[product] = product_orders

        return result_orders, 0, ""


# ---------------------------------------------------------------------------
# MARKET-MAKING CORE
# ---------------------------------------------------------------------------

def _orders_for_product(
    *,
    product: str,
    depth: OrderDepth,
    position: int,
    limit: int,
    edge: int,
    fair_price: float | None,
) -> list[Order]:
    if fair_price is None:
        return []

    orders: list[Order] = []
    position_after_orders = position
    buy_limit_price = fair_price - edge
    sell_limit_price = fair_price + edge

    # Aggressive buys: take cheap asks
    for ask_price, ask_volume in sorted(depth.sell_orders.items()):
        if ask_price > buy_limit_price or position_after_orders >= limit:
            break
        quantity = min(-ask_volume, limit - position_after_orders)
        if quantity > 0:
            orders.append(Order(product, ask_price, quantity))
            position_after_orders += quantity

    # Aggressive sells: hit expensive bids
    for bid_price, bid_volume in sorted(depth.buy_orders.items(), reverse=True):
        if bid_price < sell_limit_price or position_after_orders <= -limit:
            break
        quantity = min(bid_volume, limit + position_after_orders)
        if quantity > 0:
            orders.append(Order(product, bid_price, -quantity))
            position_after_orders -= quantity

    # Passive quotes
    passive_bid = _passive_bid(depth, buy_limit_price)
    passive_ask = _passive_ask(depth, sell_limit_price)
    if passive_bid is None or passive_ask is None or passive_bid >= passive_ask:
        return orders

    passive_reference_position = position_after_orders
    buy_quantity = limit - passive_reference_position
    if buy_quantity > 0:
        orders.append(Order(product, passive_bid, buy_quantity))

    sell_quantity = limit + passive_reference_position
    if sell_quantity > 0:
        orders.append(Order(product, passive_ask, -sell_quantity))

    return orders


def _midprice(depth: OrderDepth) -> float | None:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def _passive_bid(depth: OrderDepth, buy_limit_price: float) -> int | None:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return None
    return min(best_bid + 1, best_ask - 1, int(buy_limit_price))


def _passive_ask(depth: OrderDepth, sell_limit_price: float) -> int | None:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return None
    return max(best_ask - 1, best_bid + 1, int(sell_limit_price))


def _best_bid(depth: OrderDepth) -> int | None:
    prices = [price for price, volume in depth.buy_orders.items() if volume > 0]
    return max(prices) if prices else None


def _best_ask(depth: OrderDepth) -> int | None:
    prices = [price for price, volume in depth.sell_orders.items() if volume < 0]
    return min(prices) if prices else None
