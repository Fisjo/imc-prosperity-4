from __future__ import annotations

import math
from typing import Dict, List

from prosperity4bt.datamodel import Order, OrderDepth, TradingState


PRODUCTS = (
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
)

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

EDGE = 4.0
Z_ENTRY = 1.5
MIN_OBSERVATIONS = 200


class ExpandingZScore:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, price: float) -> None:
        self.n += 1
        delta = price - self.mean
        self.mean += delta / self.n
        delta_after_update = price - self.mean
        self.m2 += delta * delta_after_update

    def ready(self) -> bool:
        return self.n >= MIN_OBSERVATIONS

    def z_score(self, price: float) -> float:
        variance = self.m2 / max(1, self.n - 1)
        return (price - self.mean) / math.sqrt(max(variance, 1e-9))


class Trader:
    def __init__(self) -> None:
        self.estimators = {product: ExpandingZScore() for product in PRODUCTS}

    @staticmethod
    def _all_level_microprice(depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None

        bid_volume = sum(volume for volume in depth.buy_orders.values() if volume > 0)
        ask_volume = sum(-volume for volume in depth.sell_orders.values() if volume < 0)
        if bid_volume <= 0 or ask_volume <= 0:
            return None

        bid_vwap = (
            sum(price * volume for price, volume in depth.buy_orders.items() if volume > 0)
            / bid_volume
        )
        ask_vwap = (
            sum(price * -volume for price, volume in depth.sell_orders.items() if volume < 0)
            / ask_volume
        )
        return (bid_vwap * ask_volume + ask_vwap * bid_volume) / (bid_volume + ask_volume)

    @staticmethod
    def _orders_for_product(
        product: str,
        depth: OrderDepth,
        fair_value: float,
        z_score: float,
        position: int,
    ) -> List[Order]:
        limit = POSITION_LIMITS[product]
        buy_capacity = limit - position
        sell_capacity = limit + position
        orders: List[Order] = []

        if z_score <= -Z_ENTRY:
            for ask_price, ask_volume in sorted(depth.sell_orders.items()):
                if ask_price > fair_value - EDGE or buy_capacity <= 0:
                    break
                quantity = min(-ask_volume, buy_capacity)
                if quantity > 0:
                    orders.append(Order(product, ask_price, quantity))
                    buy_capacity -= quantity

        if z_score >= Z_ENTRY:
            for bid_price, bid_volume in sorted(depth.buy_orders.items(), reverse=True):
                if bid_price < fair_value + EDGE or sell_capacity <= 0:
                    break
                quantity = min(bid_volume, sell_capacity)
                if quantity > 0:
                    orders.append(Order(product, bid_price, -quantity))
                    sell_capacity -= quantity

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        for product in PRODUCTS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue

            microprice = self._all_level_microprice(depth)
            if microprice is None:
                continue

            estimator = self.estimators[product]
            if estimator.ready():
                z_score = estimator.z_score(microprice)
                orders = self._orders_for_product(
                    product,
                    depth,
                    estimator.mean,
                    z_score,
                    state.position.get(product, 0),
                )
                if orders:
                    result[product] = orders

            estimator.update(microprice)

        return result, 0, state.traderData
