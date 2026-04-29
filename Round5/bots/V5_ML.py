import json
import math
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState


LIMIT = 10
PENNY_SIZE = 5

SELECTED = [
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "MICROCHIP_OVAL",
    "MICROCHIP_RECTANGLE",
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
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_VOID_BLUE",
    "UV_VISOR_AMBER",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
]

ML = {
    "OXYGEN_SHAKE_GARLIC": {
        "q": 0.60,
        "mean": [0.1818181818, 0.3631363136, 0.904490449, 1.8104310431, 14.0453045305, -0.0034726759, 18.2591259126, -18.2854285429],
        "scale": [11.1317782543, 15.7824207169, 25.2089384524, 35.8075244898, 1.5074730353, 0.2978849111, 4.4281510056, 4.381330654],
        "coef": [0.0065544877, 0.0133072666, -0.0051592792, -0.0042823853, -0.0264587087, 0.0199232035, 0.1458620188, 0.1409653857],
        "intercept": 0.0001534159,
    },
    "GALAXY_SOUNDS_SOLAR_WINDS": {
        "q": 0.55,
        "mean": [-0.0411541154, -0.0837083708, -0.2203220322, -0.4540454045, 12.8725872587, -0.0034726759, 18.2591259126, -18.2854285429],
        "scale": [10.2305888395, 14.3831422536, 22.4415334881, 31.6282215407, 1.2756685893, 0.2978849111, 4.4281510056, 4.381330654],
        "coef": [0.0054244192, -0.0324359954, 0.0109026501, 0.0090864558, 0.0292366574, 0.0852265855, 0.0296172469, 0.0089935456],
        "intercept": -0.0005584837,
    },
}

BASE_PRODUCTS = [product for product in SELECTED if product not in ML and product != "ROBOT_IRONING"]


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


class Trader:
    def position(self, state: TradingState, product: str) -> int:
        return int(state.position.get(product, 0))

    def remaining_buy(self, state: TradingState, product: str) -> int:
        return max(0, LIMIT - self.position(state, product))

    def remaining_sell(self, state: TradingState, product: str) -> int:
        return max(0, LIMIT + self.position(state, product))

    def used_order_position(self, orders: List[Order]) -> int:
        return sum(order.quantity for order in orders)

    def add_buy(self, state: TradingState, orders_by_product: Dict[str, List[Order]], product: str, price: int, size: int) -> None:
        orders = orders_by_product[product]
        available = self.remaining_buy(state, product) - max(0, self.used_order_position(orders))
        qty = min(size, available)
        if qty > 0:
            orders.append(Order(product, int(price), int(qty)))

    def add_sell(self, state: TradingState, orders_by_product: Dict[str, List[Order]], product: str, price: int, size: int) -> None:
        orders = orders_by_product[product]
        available = self.remaining_sell(state, product) + min(0, self.used_order_position(orders))
        qty = min(size, available)
        if qty > 0:
            orders.append(Order(product, int(price), -int(qty)))

    def trade_penny_products(self, state: TradingState, orders_by_product: Dict[str, List[Order]], products: List[str]) -> None:
        for product in products:
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

    def trade_ml_product(self, state: TradingState, orders_by_product: Dict[str, List[Order]], data: Dict, product: str) -> None:
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

    def trade_robot_mean_reversion(self, state: TradingState, orders_by_product: Dict[str, List[Order]], data: Dict) -> None:
        product = "ROBOT_IRONING"
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
        for product in ML:
            self.trade_ml_product(state, orders_by_product, data, product)
        self.trade_robot_mean_reversion(state, orders_by_product, data)

        return orders_by_product, 0, dump_data(data)