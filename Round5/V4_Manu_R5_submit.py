from __future__ import annotations

# Submission-friendly imports. Tries the IMC server convention first
# (`from datamodel import ...`) and falls back to the local backtester
# package so the file still runs unchanged in prosperity4btx.
try:
    from datamodel import Order, OrderDepth, TradingState
except ModuleNotFoundError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

import json
import math
from typing import Any, Dict, List, Optional


ROUND_NUM = 5
ASSET_LIMIT = 10
INVENTORY_LAMBDA = 0.02
PASSIVE_EDGE = 0.0
TAKE_EDGE = 0.0
TAKE_SIZE = ASSET_LIMIT

# ---------------------------------------------------------------------------
# Online macro-skew configuration (replaces the historical OLS plan).
# ---------------------------------------------------------------------------
# The original V4 precomputed `fair_skew` and `target_position` per (timestamp,
# product) using rolling OLS, half-life weighted pair pressures, basket-level
# pair pressures, and a cross-sectional return rank. None of that is feasible
# tick-by-tick on the IMC server (no CSVs, no precompute step).
#
# The online replacement below preserves the *behavioural intent* -- "if a
# product is dislocated relative to its basket peers, push the reservation
# the other way" -- with a tick-local computation:
#
#   1. Compute a microprice per product.
#   2. For each basket, compute a one-vs-rest residual:
#        residual_p = mid_p - mean(mid_q for q != p in same basket)
#   3. Maintain an EWMA mean and EWMA variance of that residual per product.
#      After warmup, derive a clipped z-score and use it as a fair_skew
#      pressure on `p`.
#   4. Inventory targets default to 0 -- the original `target_position` was
#      itself bounded to ±asset_limit and dominated by inventory-pull.
#
# The reservation formula in `MmSkewBaseTrader.run` stays identical:
#   reservation = fair + fair_skew - INVENTORY_LAMBDA * (position - target)

OVR_ALPHA = 2.0 / 101.0          # EWMA decay of residual mean and variance
OVR_WARMUP = 100                 # ticks before z-scores are emitted
OVR_SCALE = 0.20                 # base sensitivity of fair_skew to z-score
OVR_Z_CLIP = 3.0                 # absolute z cap before applying scale
OVR_BASKET_SCALES: Dict[str, float] = {
    # Basket-level multipliers tuned to mirror the relative weights the
    # original `_pair_skews` placed on tightly-cointegrated baskets vs the
    # noisier ones. PEBBLES is intentionally kept off this list because the
    # `BaseTrader` PEBBLES override below already handles that basket
    # directly via its own residual machinery.
    "GALAXY_SOUNDS": 1.0,
    "MICROCHIP":     1.0,
    "OXYGEN_SHAKE":  1.0,
    "PANEL":         1.0,
    "ROBOT":         1.0,
    "SLEEP_POD":     1.0,
    "SNACKPACK":     2.0,
    "TRANSLATOR":    1.0,
    "UV_VISOR":      1.0,
}
EPS = 1e-12

BASKET_PREFIXES: Dict[str, str] = {
    "GALAXY_SOUNDS": "GALAXY_SOUNDS_",
    "MICROCHIP": "MICROCHIP_",
    "OXYGEN_SHAKE": "OXYGEN_SHAKE_",
    "PANEL": "PANEL_",
    "PEBBLES": "PEBBLES_",
    "ROBOT": "ROBOT_",
    "SLEEP_POD": "SLEEP_POD_",
    "SNACKPACK": "SNACKPACK_",
    "TRANSLATOR": "TRANSLATOR_",
    "UV_VISOR": "UV_VISOR_",
}

# Universe of all products this bot expects to see, ordered alphabetically
# (so iteration is deterministic across runs). Built from BASKET_PREFIXES at
# import time -- no I/O. The list is used to keep the EWMA state in a fixed
# 50-element array regardless of whether a product appears in a given tick.
_PRODUCT_LIST: tuple[str, ...] = tuple(sorted(
    f"{prefix}{suffix}"
    for prefix, suffix_set in (
        ("GALAXY_SOUNDS_", ("BLACK_HOLES", "DARK_MATTER", "PLANETARY_RINGS",
                            "SOLAR_FLAMES", "SOLAR_WINDS")),
        ("MICROCHIP_",     ("CIRCLE", "OVAL", "RECTANGLE", "SQUARE", "TRIANGLE")),
        ("OXYGEN_SHAKE_",  ("CHOCOLATE", "EVENING_BREATH", "GARLIC",
                            "MINT", "MORNING_BREATH")),
        ("PANEL_",         ("1X2", "1X4", "2X2", "2X4", "4X4")),
        ("PEBBLES_",       ("L", "M", "S", "XL", "XS")),
        ("ROBOT_",         ("DISHES", "IRONING", "LAUNDRY", "MOPPING", "VACUUMING")),
        ("SLEEP_POD_",     ("COTTON", "LAMB_WOOL", "NYLON", "POLYESTER", "SUEDE")),
        ("SNACKPACK_",     ("CHOCOLATE", "PISTACHIO", "RASPBERRY",
                            "STRAWBERRY", "VANILLA")),
        ("TRANSLATOR_",    ("ASTRO_BLACK", "ECLIPSE_CHARCOAL", "GRAPHITE_MIST",
                            "SPACE_GRAY", "VOID_BLUE")),
        ("UV_VISOR_",      ("AMBER", "MAGENTA", "ORANGE", "RED", "YELLOW")),
    )
    for suffix in suffix_set
))


# ---------------------------------------------------------------------------
# Online macro-skew engine
# ---------------------------------------------------------------------------
# `MmSkewBaseTrader` is the submission-friendly online replacement of the
# original V4 `MmSkewBaseTrader` + `_build_plan` machinery. The original
# computed `fair_skew` and `target_position` arrays of shape (T, P) up front
# from rolling OLS, half-lives and basket pressures over a full day's CSV.
# This online version keeps the *same reservation formula* but derives those
# two quantities tick-by-tick from EWMA statistics of basket residuals.
#
# Public surface preserved (so `BaseTrader` can subclass it and call the
# same hooks the V4 `BaseTrader.run` expects):
#
#   self.asset_limit, self.inventory_lambda,
#   self.passive_edge,  self.take_edge, self.take_size
#   self.run(state)               -> (orders, conversions, trader_data)
#   self._current_skew(product)   -> fair_skew used this tick (for PEBBLES blend)
#   self._current_target(product) -> target_position used this tick (= 0)
#
# Note: `target_position` returns 0 because the cleanest online surrogate of
# the original "skew toward the fair-pressure-aligned side" is to fold that
# pressure entirely into `fair_skew` and let inventory mean-revert via the
# `INVENTORY_LAMBDA * position` term in the reservation. This matches what
# the original logic effectively produced on a tick-by-tick basis once the
# rolling z-scores were below the signal threshold (the common case).


class MmSkewBaseTrader:
    """Online version of the macro-skew base trader.

    Mirrors the V4 trader's reservation formula:
        reservation = microprice + fair_skew - INVENTORY_LAMBDA * (position - target)
    The `fair_skew` per product is now an online z-score on a one-vs-rest
    basket residual, and `target` is fixed at 0 (see module docstring above).
    """

    def __init__(
        self,
        *args: Any,
        inventory_lambda: float = INVENTORY_LAMBDA,
        passive_edge: float = PASSIVE_EDGE,
        take_edge: float = TAKE_EDGE,
        take_size: int = TAKE_SIZE,
        asset_limit: int = ASSET_LIMIT,
        **kwargs: Any,
    ) -> None:
        # `*args, **kwargs` are accepted but ignored to remain compatible with
        # callers that pass legacy plan-related kwargs (window=, zscore_window=
        # etc.). Those parameters no longer have any effect.
        self.inventory_lambda = float(inventory_lambda)
        self.passive_edge = float(passive_edge)
        self.take_edge = float(take_edge)
        self.take_size = int(take_size)
        self.asset_limit = int(asset_limit)

        # Per-product residual EWMA state (mean, variance) keyed by product
        # name. Kept as plain Python dicts (no numpy required) so the file
        # can run on a vanilla Python environment if numpy were ever dropped.
        self._residual_mean: Dict[str, float] = {}
        self._residual_var: Dict[str, float] = {}
        self._residual_count: Dict[str, int] = {}

        # Cached per-product values for the *current* tick; re-derived at the
        # start of every `run`. Subclasses can read them via _current_skew /
        # _current_target -- this is how `BaseTrader`'s PEBBLES override
        # picks up the same fair_skew the parent used on this tick.
        self._fair_skew_now: Dict[str, float] = {}
        self._target_now: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Hooks for subclasses (BaseTrader uses these for the PEBBLES override)
    # ------------------------------------------------------------------
    def _current_skew(self, product: str) -> float:
        return self._fair_skew_now.get(product, 0.0)

    def _current_target(self, product: str) -> float:
        return self._target_now.get(product, 0.0)

    # ------------------------------------------------------------------
    # Online residual / fair_skew computation
    # ------------------------------------------------------------------
    def _update_skews(self, mids: Dict[str, float]) -> None:
        """Update EWMA state and refresh self._fair_skew_now / _target_now."""
        self._fair_skew_now = {}
        self._target_now = {}

        for basket, prefix in BASKET_PREFIXES.items():
            members = tuple(p for p in mids.keys() if p.startswith(prefix))
            if len(members) < 2:
                continue
            basket_scale = OVR_BASKET_SCALES.get(basket, 1.0)
            if basket_scale <= 0.0:
                continue

            # One-vs-rest residual for each member.
            for product in members:
                rest = [p for p in members if p != product]
                rest_mean = sum(mids[p] for p in rest) / len(rest)
                residual = mids[product] - rest_mean

                count = self._residual_count.get(product, 0)
                mean = self._residual_mean.get(product, 0.0)
                variance = self._residual_var.get(product, 0.0)

                if count >= OVR_WARMUP and variance > EPS:
                    z = (residual - mean) / math.sqrt(variance)
                    z = max(-OVR_Z_CLIP, min(OVR_Z_CLIP, z))
                    # Negative sign: a positive z (product rich vs basket)
                    # pushes our reservation DOWN, i.e. we are happier to
                    # sell and stingier to buy. This mirrors the sign the
                    # original `_pair_skews` produced via -np.sign(signal).
                    self._fair_skew_now[product] = -OVR_SCALE * basket_scale * z
                    self._target_now[product] = 0.0

                # EWMA update (Welford-style for variance) regardless of
                # whether we emitted a skew this tick.
                if count <= 0:
                    new_mean, new_var = residual, 0.0
                else:
                    delta = residual - mean
                    new_mean = mean + OVR_ALPHA * delta
                    new_var = (1.0 - OVR_ALPHA) * (variance + OVR_ALPHA * delta * delta)
                    if new_var < 0.0:
                        new_var = 0.0
                self._residual_mean[product] = new_mean
                self._residual_var[product] = new_var
                self._residual_count[product] = count + 1

    # ------------------------------------------------------------------
    # Per-tick driver
    # ------------------------------------------------------------------
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        # 1. Snapshot midprices for all products visible this tick.
        mids: Dict[str, float] = {}
        microprices: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            mid = _midprice(depth)
            micro = _microprice(depth)
            if mid is None or micro is None:
                continue
            if mid <= 0.0:
                continue
            mids[product] = mid
            microprices[product] = micro

        # 2. Refresh online basket EWMAs and derive this tick's fair_skews.
        self._update_skews(mids)

        # 3. Build orders product by product using the same reservation
        #    formula V4 used on top of the precomputed plan.
        orders: Dict[str, List[Order]] = {}
        for product, depth in state.order_depths.items():
            fair = microprices.get(product)
            if fair is None:
                continue
            position = int(state.position.get(product, 0))
            reservation = (
                fair
                + self._fair_skew_now.get(product, 0.0)
                - self.inventory_lambda * (position - self._target_now.get(product, 0.0))
            )
            product_orders = _orders_for_product(
                product=product,
                depth=depth,
                position=position,
                reservation=reservation,
                limit=self.asset_limit,
                passive_edge=self.passive_edge,
                take_edge=self.take_edge,
                take_size=self.take_size,
            )
            if product_orders:
                orders[product] = product_orders
        return orders, 0, ""


# ---------------------------------------------------------------------------
# Order construction helpers (unchanged from V4 except for type-hint syntax
# for the `int | None` returns, kept compatible with Python 3.9+).
# ---------------------------------------------------------------------------
def _orders_for_product(
    *,
    product: str,
    depth: OrderDepth,
    position: int,
    reservation: float,
    limit: int,
    passive_edge: float,
    take_edge: float,
    take_size: int,
) -> List[Order]:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return []

    orders: List[Order] = []
    working_position = position

    buy_capacity = min(take_size, max(0, limit - working_position))
    for ask_price in sorted(depth.sell_orders):
        if buy_capacity <= 0 or reservation - ask_price <= take_edge:
            break
        quantity = min(abs(int(depth.sell_orders[ask_price])), buy_capacity)
        if quantity > 0:
            orders.append(Order(product, int(ask_price), quantity))
            working_position += quantity
            buy_capacity -= quantity

    sell_capacity = min(take_size, max(0, limit + working_position))
    for bid_price in sorted(depth.buy_orders, reverse=True):
        if sell_capacity <= 0 or bid_price - reservation <= take_edge:
            break
        quantity = min(int(depth.buy_orders[bid_price]), sell_capacity)
        if quantity > 0:
            orders.append(Order(product, int(bid_price), -quantity))
            working_position -= quantity
            sell_capacity -= quantity

    passive_buy_capacity = max(0, limit - working_position)
    passive_bid = best_bid + 1 if best_bid + 1 < best_ask else best_bid
    if passive_buy_capacity > 0 and reservation - passive_bid > passive_edge:
        orders.append(Order(product, int(passive_bid), passive_buy_capacity))

    passive_sell_capacity = max(0, limit + working_position)
    passive_ask = best_ask - 1 if best_ask - 1 > best_bid else best_ask
    if passive_sell_capacity > 0 and passive_ask - reservation > passive_edge:
        orders.append(Order(product, int(passive_ask), -passive_sell_capacity))

    return orders


def _midprice(depth: OrderDepth) -> Optional[float]:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def _microprice(depth: OrderDepth) -> Optional[float]:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return None

    bid_notional = 0.0
    bid_volume = 0
    for price, volume in depth.buy_orders.items():
        if volume > 0:
            bid_notional += float(price * volume)
            bid_volume += int(volume)

    ask_notional = 0.0
    ask_volume = 0
    for price, volume in depth.sell_orders.items():
        if volume < 0:
            size = abs(int(volume))
            ask_notional += float(price * size)
            ask_volume += size

    total_volume = bid_volume + ask_volume
    if total_volume <= 0:
        return (best_bid + best_ask) / 2.0
    bid_vwap = bid_notional / bid_volume if bid_volume > 0 else float(best_bid)
    ask_vwap = ask_notional / ask_volume if ask_volume > 0 else float(best_ask)
    return (bid_vwap * ask_volume + ask_vwap * bid_volume) / total_volume


def _best_bid(depth: OrderDepth) -> Optional[int]:
    prices = [price for price, volume in depth.buy_orders.items() if volume > 0]
    return max(prices) if prices else None


def _best_ask(depth: OrderDepth) -> Optional[int]:
    prices = [price for price, volume in depth.sell_orders.items() if volume < 0]
    return min(prices) if prices else None




PEBBLES_PRODUCTS = ("PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL")
PEBBLES_ACTIVE = ("PEBBLES_S", "PEBBLES_XL")
PEBBLES_TARGET_SUM = 50000.0
PEBBLES_TRIGGER = 8.5
PEBBLES_TAKE_SIZE = 2
PEBBLES_PLAN_SKEW_BLEND = 0.25


class BaseTrader(MmSkewBaseTrader):
    def __init__(self, *args, tradeable_products: set[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tradeable_products = None if tradeable_products is None else set(tradeable_products)

    def run(self, state: TradingState):
        orders, conversions, trader_data = super().run(state)
        if self.tradeable_products is None:
            filtered_orders = dict(orders)
        else:
            filtered_orders = {
                product: product_orders
                for product, product_orders in orders.items()
                if product in self.tradeable_products
            }

        residual, synth_fair = _pebbles_group_residual_and_fair(state)
        if residual is None or synth_fair is None or abs(residual) < PEBBLES_TRIGGER:
            return filtered_orders, conversions, trader_data

        # Online replacement of the V4 PEBBLES override. The original blended
        # the basket-residual fair value with the precomputed plan skew via:
        #     reservation = synth_fair[p]
        #                 + PEBBLES_PLAN_SKEW_BLEND * plan.fair_skew[row, col]
        #                 - INVENTORY_LAMBDA * (pos - plan.target_position[row, col])
        # Here we read the same per-tick values from the parent's online
        # state via the hooks. PEBBLES is not in OVR_BASKET_SCALES, so the
        # parent leaves its skew at 0 by default -- meaning the blend term
        # contributes nothing here (matching the common case of the V4 plan
        # whose PEBBLES skew was tiny most of the time).
        for product in PEBBLES_ACTIVE:
            if self.tradeable_products is not None and product not in self.tradeable_products:
                continue
            depth = state.order_depths.get(product)
            if depth is None:
                continue

            position = int(state.position.get(product, 0))
            reservation = (
                synth_fair[product]
                + PEBBLES_PLAN_SKEW_BLEND * self._current_skew(product)
                - self.inventory_lambda * (position - self._current_target(product))
            )
            pebbles_orders = _orders_for_product(
                product=product,
                depth=depth,
                position=position,
                reservation=reservation,
                limit=self.asset_limit,
                passive_edge=self.passive_edge,
                take_edge=self.take_edge,
                take_size=min(self.take_size, PEBBLES_TAKE_SIZE),
            )
            if pebbles_orders:
                filtered_orders[product] = pebbles_orders
            else:
                filtered_orders.pop(product, None)

        return filtered_orders, conversions, trader_data


def _pebbles_group_residual_and_fair(
    state: TradingState,
) -> tuple[float | None, dict[str, float] | None]:
    current_fair: dict[str, float] = {}
    for product in PEBBLES_PRODUCTS:
        depth = state.order_depths.get(product)
        if depth is None:
            return None, None
        fair = _microprice(depth)
        if fair is None:
            return None, None
        current_fair[product] = fair

    total = sum(current_fair.values())
    residual = total - PEBBLES_TARGET_SUM
    synth_fair = {
        product: current_fair[product] - residual
        for product in PEBBLES_PRODUCTS
    }
    return residual, synth_fair


# Names where the direct-product test showed V5 beats the mm_skew baseline.
USE_V5_PRODUCTS: set[str] = {
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "MICROCHIP_OVAL",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "PANEL_1X4",
    "PANEL_2X2",
    "PEBBLES_L",
    "PEBBLES_S",
    "ROBOT_IRONING",
    "UV_VISOR_AMBER",
}

# The blacklist has been explicitly emptied to prevent curve-fitting.
# We now allow the Base Trader to freely market-make across all eligible assets.
DISABLE_PRODUCTS: set[str] = set()


LIMIT = 10
PENNY_SIZE = 5
FAIR_ALPHA = 0.08
FAIR_EDGE = 1.0
INVENTORY_SKEW = 0.55
TREND_GATE = 18.0
FAIR_PRODUCTS = {"PEBBLES_L"}

SELECTED = [
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "MICROCHIP_OVAL",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "PANEL_1X4",
    "PANEL_2X2",
    "PANEL_2X4",
    "PEBBLES_L",
    "PEBBLES_S",
    "PEBBLES_XL",
    "ROBOT_DISHES",
    "ROBOT_IRONING",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_SUEDE",
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_VANILLA",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_VOID_BLUE",
    "UV_VISOR_AMBER",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
]

ML = {
    "OXYGEN_SHAKE_GARLIC": {
        "q": 0.60,
        "mean": [
            0.1818181818,
            0.3631363136,
            0.904490449,
            1.8104310431,
            14.0453045305,
            -0.0034726759,
            18.2591259126,
            -18.2854285429,
        ],
        "scale": [
            11.1317782543,
            15.7824207169,
            25.2089384524,
            35.8075244898,
            1.5074730353,
            0.2978849111,
            4.4281510056,
            4.381330654,
        ],
        "coef": [
            0.0065544877,
            0.0133072666,
            -0.0051592792,
            -0.0042823853,
            -0.0264587087,
            0.0199232035,
            0.1458620188,
            0.1409653857,
        ],
        "intercept": 0.0001534159,
    },
}

BASE_PRODUCTS = [
    product
    for product in SELECTED
    if product not in ML and product != "ROBOT_IRONING" and product not in FAIR_PRODUCTS
]


def best_bid_ask(depth: OrderDepth):
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid = max(depth.buy_orders)
    ask = min(depth.sell_orders)
    if bid >= ask:
        return None
    return bid, ask


def mid(depth: OrderDepth) -> Optional[float]:
    bba = best_bid_ask(depth)
    if bba is None:
        return None
    bid, ask = bba
    return (bid + ask) / 2.0


def load_data(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def dump_data(data: Dict) -> str:
    return json.dumps(data, separators=(",", ":"))


class V5PebblesAndStructureTrader:
    def position(self, state: TradingState, product: str) -> int:
        return int(state.position.get(product, 0))

    def remaining_buy(self, state: TradingState, product: str) -> int:
        return max(0, LIMIT - self.position(state, product))

    def remaining_sell(self, state: TradingState, product: str) -> int:
        return max(0, LIMIT + self.position(state, product))

    def used_order_position(self, orders: List[Order]) -> int:
        return sum(order.quantity for order in orders)

    def add_buy(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        product: str,
        price: int,
        size: int,
    ) -> None:
        orders = orders_by_product[product]
        available = self.remaining_buy(state, product) - max(0, self.used_order_position(orders))
        qty = min(size, available)
        if qty > 0:
            orders.append(Order(product, int(price), int(qty)))

    def add_sell(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        product: str,
        price: int,
        size: int,
    ) -> None:
        orders = orders_by_product[product]
        available = self.remaining_sell(state, product) + min(0, self.used_order_position(orders))
        qty = min(size, available)
        if qty > 0:
            orders.append(Order(product, int(price), -int(qty)))

    def trade_penny_products(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        products: List[str],
    ) -> None:
        for product in products:
            if product not in USE_V5_PRODUCTS:
                continue
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            bba = best_bid_ask(depth)
            if bba is None:
                continue
            bid, ask = bba
            if ask - bid > 1:
                buy_px = bid + 1
                sell_px = ask - 1
            else:
                buy_px = bid
                sell_px = ask
            self.add_buy(state, orders_by_product, product, buy_px, PENNY_SIZE)
            self.add_sell(state, orders_by_product, product, sell_px, PENNY_SIZE)

    def stable_fair(self, data: Dict, product: str, current_mid: float) -> float:
        key = "fair_" + product
        previous = data.get(key)
        fair = current_mid if previous is None else (1.0 - FAIR_ALPHA) * float(previous) + FAIR_ALPHA * current_mid
        data[key] = fair
        history_key = "fair_hist_" + product
        history = data.get(history_key, [])
        history.append(current_mid)
        history = history[-31:]
        data[history_key] = history
        return fair

    def trade_stable_fair_products(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        data: Dict,
        products,
    ) -> None:
        for product in products:
            if product not in USE_V5_PRODUCTS:
                continue
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            bba = best_bid_ask(depth)
            m = mid(depth)
            if bba is None or m is None:
                continue
            bid, ask = bba
            fair = self.stable_fair(data, product, m)
            history = data.get("fair_hist_" + product, [])
            trend = (history[-1] - history[0]) if len(history) >= 31 else 0.0
            pos = self.position(state, product)
            reservation = fair - INVENTORY_SKEW * pos
            buy_px = bid + 1 if ask - bid > 1 else bid
            sell_px = ask - 1 if ask - bid > 1 else ask
            if buy_px <= reservation - FAIR_EDGE and not (trend < -TREND_GATE and pos >= 0):
                self.add_buy(state, orders_by_product, product, buy_px, PENNY_SIZE)
            if sell_px >= reservation + FAIR_EDGE and not (trend > TREND_GATE and pos <= 0):
                self.add_sell(state, orders_by_product, product, sell_px, PENNY_SIZE)

    def features(self, mids: List[float], spread: int, obi: float, bidv: int, askv: int) -> List[float]:
        vals = []
        for lag in (1, 2, 5, 10):
            vals.append(mids[-1] - mids[-1 - lag] if len(mids) > lag else 0.0)
        vals.extend([spread, obi, bidv, askv])
        return vals

    def ml_probability(self, product: str, vals: List[float]) -> float:
        cfg = ML[product]
        z = cfg["intercept"]
        for x, mu, scale, coef in zip(vals, cfg["mean"], cfg["scale"], cfg["coef"]):
            z += ((x - mu) / scale) * coef
        return 1.0 / (1.0 + math.exp(-z))

    def trade_ml_product(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        data: Dict,
        product: str,
    ) -> None:
        if product not in USE_V5_PRODUCTS:
            return
        depth = state.order_depths.get(product)
        if depth is None:
            return
        bba = best_bid_ask(depth)
        m = mid(depth)
        if bba is None or m is None:
            return

        bid, ask = bba
        bidv = max(0, int(depth.buy_orders.get(bid, 0)))
        askv = max(0, -int(depth.sell_orders.get(ask, 0)))
        total = bidv + askv
        obi = (bidv - askv) / total if total > 0 else 0.0

        key = "ml_mid_" + product
        mids = data.get(key, [])
        mids.append(m)
        mids = mids[-11:]
        data[key] = mids
        if len(mids) < 11:
            return

        prob_up = self.ml_probability(product, self.features(mids, ask - bid, obi, bidv, askv))
        threshold = ML[product]["q"]
        buy_px = bid + 1 if ask - bid > 1 else bid
        sell_px = ask - 1 if ask - bid > 1 else ask

        if prob_up > threshold:
            self.add_buy(state, orders_by_product, product, buy_px, LIMIT)
        elif prob_up < 1.0 - threshold:
            self.add_sell(state, orders_by_product, product, sell_px, LIMIT)

    def trade_robot_mean_reversion(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        data: Dict,
    ) -> None:
        product = "ROBOT_IRONING"
        if product not in USE_V5_PRODUCTS:
            return
        depth = state.order_depths.get(product)
        if depth is None:
            return
        bba = best_bid_ask(depth)
        m = mid(depth)
        if bba is None or m is None:
            return

        key = "last_mid_" + product
        prev = data.get(key)
        data[key] = m
        if prev is None:
            return

        last_ret = m - float(prev)
        if abs(last_ret) < 0.5:
            return

        bid, ask = bba
        buy_px = bid + 1 if ask - bid > 1 else bid
        sell_px = ask - 1 if ask - bid > 1 else ask
        if last_ret > 0:
            self.add_sell(state, orders_by_product, product, sell_px, LIMIT)
        elif last_ret < 0:
            self.add_buy(state, orders_by_product, product, buy_px, LIMIT)

    def run(self, state: TradingState):
        data = load_data(state.traderData)
        orders_by_product: Dict[str, List[Order]] = {product: [] for product in state.order_depths}

        self.trade_penny_products(state, orders_by_product, BASE_PRODUCTS)
        self.trade_stable_fair_products(state, orders_by_product, data, FAIR_PRODUCTS)
        for product in ML:
            self.trade_ml_product(state, orders_by_product, data, product)
        self.trade_robot_mean_reversion(state, orders_by_product, data)

        filtered = {product: orders for product, orders in orders_by_product.items() if orders}
        return filtered, 0, dump_data(data)


class Trader:
    def __init__(self, *args, **kwargs) -> None:
        self.base = BaseTrader(*args, **kwargs)
        self.v5 = V5PebblesAndStructureTrader()

    def run(self, state: TradingState):
        raw = state.traderData or ""
        try:
            wrapper_data = json.loads(raw) if raw else {}
            if not isinstance(wrapper_data, dict):
                wrapper_data = {}
        except Exception:
            wrapper_data = {}

        state.traderData = wrapper_data.get("v5", "")
        v5_orders, _, v5_data = self.v5.run(state)
        state.traderData = raw

        base_orders, _, _ = self.base.run(state)

        result: Dict[str, List[Order]] = {}
        for product, orders in base_orders.items():
            if product not in USE_V5_PRODUCTS and product not in DISABLE_PRODUCTS:
                result[product] = orders

        for product in USE_V5_PRODUCTS:
            orders = v5_orders.get(product)
            if orders:
                result[product] = orders
            else:
                result.pop(product, None)

        for product in DISABLE_PRODUCTS:
            result.pop(product, None)

        return result, 0, json.dumps({"v5": v5_data}, separators=(",", ":"))

if __name__ == "__main__":
    pass