from __future__ import annotations

from prosperity4bt.datamodel import Order, OrderDepth, TradingState


DEFAULT_LIMIT = 10
EDGE = 1


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

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        orders: dict[str, list[Order]] = {}
        for product, depth in state.order_depths.items():
            product_orders = _orders_for_product(
                product=product,
                depth=depth,
                position=int(state.position.get(product, 0)),
                limit=self.limit,
                edge=self.edge,
            )
            if product_orders:
                orders[product] = product_orders

        return orders, 0, ""


def _orders_for_product(
    *,
    product: str,
    depth: OrderDepth,
    position: int,
    limit: int,
    edge: int,
) -> list[Order]:
    fair_price = _midprice(depth)
    if fair_price is None:
        return []

    orders: list[Order] = []
    position_after_orders = position
    buy_limit_price = fair_price - edge
    sell_limit_price = fair_price + edge

    for ask_price, ask_volume in sorted(depth.sell_orders.items()):
        if ask_price > buy_limit_price or position_after_orders >= limit:
            break
        quantity = min(-ask_volume, limit - position_after_orders)
        if quantity > 0:
            orders.append(Order(product, ask_price, quantity))
            position_after_orders += quantity

    for bid_price, bid_volume in sorted(depth.buy_orders.items(), reverse=True):
        if bid_price < sell_limit_price or position_after_orders <= -limit:
            break
        quantity = min(bid_volume, limit + position_after_orders)
        if quantity > 0:
            orders.append(Order(product, bid_price, -quantity))
            position_after_orders -= quantity

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
    if not prices:
        return None
    return max(prices)


def _best_ask(depth: OrderDepth) -> int | None:
    prices = [price for price, volume in depth.sell_orders.items() if volume < 0]
    if not prices:
        return None
    return min(prices)
