from datamodel import Order, OrderDepth, TradingState
import json
from typing import List


class Trader:
    """
    Prosperity 4 — Round 1 — v5 (Optimized for Buy and Hold)

    Key fix: Pepper accumulation only sweeps L1 (best ask),
    uses aggressive bids for the rest. 
    Market Making logic for Pepper REMOVED to maximize PnL 
    through a pure Buy and Hold strategy on upward drift.
    """

    OSMIUM = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM_LIMIT = 80
    PEPPER_LIMIT = 80
    OSMIUM_FV = 10000
    PEPPER_SLOPE = 0.001

    def run(self, state: TradingState):
        result = {}
        # Mantenemos la carga de data por si la necesitas para OSMIUM en el futuro
        data = json.loads(state.traderData) if state.traderData else {}

        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)

            if product == self.OSMIUM:
                result[product] = self.trade_osmium(od, pos)
            elif product == self.PEPPER:
                # Pepper ya no necesita modificar 'data', solo el timestamp
                orders = self.trade_pepper(od, pos, state.timestamp)
                result[product] = orders
            else:
                result[product] = []

        return result, 0, json.dumps(data)

    def trade_osmium(self, od: OrderDepth, pos: int) -> List[Order]:
        orders = []
        buy_cap = self.OSMIUM_LIMIT - pos
        sell_cap = self.OSMIUM_LIMIT + pos

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else self.OSMIUM_FV - 5
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else self.OSMIUM_FV + 5

        my_bid = min(self.OSMIUM_FV - 2, best_bid + 1)
        my_ask = max(self.OSMIUM_FV + 2, best_ask - 1)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        # Control de inventario básico (Avellaneda-Stoikov muy ligero)
        if pos > 40:
            my_bid -= 1
            my_ask -= 1
        elif pos < -40:
            my_bid += 1
            my_ask += 1

        if buy_cap > 0:
            orders.append(Order(self.OSMIUM, my_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(self.OSMIUM, my_ask, -sell_cap))

        return orders

    def trade_pepper(self, od: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        orders = []
        buy_cap = self.PEPPER_LIMIT - pos

        fv = self.OSMIUM_FV + timestamp * self.PEPPER_SLOPE
        fv_int = int(round(fv))

        # ── BUY AND HOLD PHASE ──────────────────────────────────
        # El bot simplemente evaluará en todo momento si tiene hueco para comprar.
        # Si tiene espacio, comprará. Si está lleno (pos == 80), no hará nada (Hold).
        if buy_cap > 0:
            # Compra agresiva (Sweep): devorar el L1 si el precio es razonable respecto al Fair Value
            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                if best_ask <= fv_int + 2:
                    vol = min(-od.sell_orders[best_ask], buy_cap)
                    orders.append(Order(self.PEPPER, best_ask, vol))
                    buy_cap -= vol

            # Compra pasiva (Pennying): poner órdenes un tick por encima del mejor comprador
            if buy_cap > 0 and od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                our_bid = min(best_bid + 1, fv_int)
                orders.append(Order(self.PEPPER, our_bid, buy_cap))

        # Fíjate que no hay lógica de venta. Una vez llega a 80, simplemente mantiene.
        return orders