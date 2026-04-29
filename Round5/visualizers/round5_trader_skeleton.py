
# ─── IMC Prosperity 4 — Round 5 Pair Trading Trader ───────────────────────
# Auto-generated skeleton. Fill SELECTED_PAIRS from analysis notebook.
# Position limit per product: 10.

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json, math

# Each tuple: (Y, X, beta, entry_z, exit_z, qty)
SELECTED_PAIRS = [

]

POS_LIMIT   = 10
ROLL_WIN    = 300   # lookback for rolling z-score


class Trader:
    def run(self, state: TradingState):
        result  = {}  # product -> List[Order]
        traderData = json.loads(state.traderData) if state.traderData else {}

        for (Y, X, beta, entry_z, exit_z, qty) in SELECTED_PAIRS:
            # ── Get mid prices ──────────────────────────────────────────────
            y_mid = self._mid(state, Y)
            x_mid = self._mid(state, X)
            if y_mid is None or x_mid is None:
                continue

            # ── Update spread history ────────────────────────────────────────
            key = f"{Y}_{X}_hist"
            hist = traderData.get(key, [])
            spread_now = y_mid - beta * x_mid
            hist.append(spread_now)
            if len(hist) > ROLL_WIN:
                hist = hist[-ROLL_WIN:]
            traderData[key] = hist

            if len(hist) < 30:  # need enough history
                continue

            # ── Compute z-score ──────────────────────────────────────────────
            mu  = sum(hist) / len(hist)
            var = sum((h - mu)**2 for h in hist) / len(hist)
            sig = math.sqrt(var) if var > 0 else 1e-9
            z   = (spread_now - mu) / sig

            # ── Current positions ────────────────────────────────────────────
            y_pos = state.position.get(Y, 0)
            x_pos = state.position.get(X, 0)

            # ── Signal & order generation ────────────────────────────────────
            y_orders, x_orders = [], []

            if z > entry_z and y_pos >= 0 and x_pos <= 0:  # SHORT spread
                # sell Y, buy X
                sell_y = min(qty, POS_LIMIT + y_pos)  # how much we can sell
                buy_x  = min(qty, POS_LIMIT - x_pos)  # but wait: we WANT negative x
                sell_y = min(qty, POS_LIMIT + y_pos)
                buy_x  = min(qty, POS_LIMIT + x_pos)  # x_pos <=0 so this increases |x_pos|
                # Actually: shorting spread = sell Y (y_pos goes negative) + buy X (x_pos goes positive)
                sell_y = min(qty, POS_LIMIT + y_pos)   # room to sell = limit + current pos (y_pos>=0)
                buy_x  = min(qty, POS_LIMIT - x_pos)   # room to buy  = limit - current pos (x_pos<=0, so this is limit+|x_pos|)
                to_trade = min(sell_y, buy_x)
                if to_trade > 0:
                    y_bid = self._best_bid(state, Y)
                    x_ask = self._best_ask(state, X)
                    if y_bid and x_ask:
                        y_orders.append(Order(Y, y_bid, -to_trade))
                        x_orders.append(Order(X, x_ask,  to_trade))

            elif z < -entry_z and y_pos <= 0 and x_pos >= 0:  # LONG spread
                # buy Y, sell X
                buy_y  = min(qty, POS_LIMIT + y_pos)   # room to buy
                sell_x = min(qty, POS_LIMIT - x_pos)   # room to sell
                to_trade = min(buy_y, sell_x)
                if to_trade > 0:
                    y_ask = self._best_ask(state, Y)
                    x_bid = self._best_bid(state, X)
                    if y_ask and x_bid:
                        y_orders.append(Order(Y, y_ask,  to_trade))
                        x_orders.append(Order(X, x_bid, -to_trade))

            elif abs(z) < exit_z:  # EXIT / flatten
                if y_pos > 0:
                    y_bid = self._best_bid(state, Y)
                    if y_bid: y_orders.append(Order(Y, y_bid, -y_pos))
                elif y_pos < 0:
                    y_ask = self._best_ask(state, Y)
                    if y_ask: y_orders.append(Order(Y, y_ask, -y_pos))
                if x_pos > 0:
                    x_bid = self._best_bid(state, X)
                    if x_bid: x_orders.append(Order(X, x_bid, -x_pos))
                elif x_pos < 0:
                    x_ask = self._best_ask(state, X)
                    if x_ask: x_orders.append(Order(X, x_ask, -x_pos))

            if y_orders: result[Y] = result.get(Y, []) + y_orders
            if x_orders: result[X] = result.get(X, []) + x_orders

        return result, 0, json.dumps(traderData)

    @staticmethod
    def _mid(state: TradingState, product: str):
        od = state.order_depths.get(product)
        if od is None: return None
        if not od.buy_orders or not od.sell_orders: return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2

    @staticmethod
    def _best_bid(state: TradingState, product: str):
        od = state.order_depths.get(product)
        if od is None or not od.buy_orders: return None
        return int(max(od.buy_orders))

    @staticmethod
    def _best_ask(state: TradingState, product: str):
        od = state.order_depths.get(product)
        if od is None or not od.sell_orders: return None
        return int(min(od.sell_orders))
