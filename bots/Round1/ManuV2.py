from datamodel import Order, OrderDepth, TradingState
import json
from typing import List


class Trader:
    """
    Prosperity 4 — Round 1 — v5

    Key fix: Pepper accumulation only sweeps L1 (best ask),
    uses aggressive bids for the rest. Saves ~200-400 pts
    on initial spread cost with negligible drift loss.
    """

    OSMIUM = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM_LIMIT = 80
    PEPPER_LIMIT = 80
    OSMIUM_FV = 10000
    PEPPER_SLOPE = 0.001

    # Pepper MM params
    PEPPER_MM_MIN_POS = 40
    PEPPER_MM_MAX_VOL = 20
    PEPPER_MM_ASK_FLOOR = 3

    def run(self, state: TradingState):
        result = {}
        data = json.loads(state.traderData) if state.traderData else {}

        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)

            if product == self.OSMIUM:
                result[product] = self.trade_osmium(od, pos)
            elif product == self.PEPPER:
                orders, data = self.trade_pepper(od, pos, state.timestamp, data)
                result[product] = orders
            else:
                result[product] = []

        return result, 0, json.dumps(data)

    # ══════════════════════════════════════════════════════════════
    #  ASH_COATED_OSMIUM — same Take/Clear/Make as before
    # ══════════════════════════════════════════════════════════════

    def trade_osmium(self, od: OrderDepth, pos: int) -> List[Order]:
        orders = []
        FV = self.OSMIUM_FV
        LIMIT = self.OSMIUM_LIMIT
        buy_cap = LIMIT - pos
        sell_cap = LIMIT + pos

        # TAKE
        for ask in sorted(od.sell_orders.keys()):
            if ask < FV and buy_cap > 0:
                vol = min(-od.sell_orders[ask], buy_cap)
                orders.append(Order(self.OSMIUM, ask, vol))
                buy_cap -= vol

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid > FV and sell_cap > 0:
                vol = min(od.buy_orders[bid], sell_cap)
                orders.append(Order(self.OSMIUM, bid, -vol))
                sell_cap -= vol

        # CLEAR
        if pos > 0 and sell_cap > 0:
            vol = min(pos, sell_cap)
            orders.append(Order(self.OSMIUM, FV, -vol))
            sell_cap -= vol
        elif pos < 0 and buy_cap > 0:
            vol = min(-pos, buy_cap)
            orders.append(Order(self.OSMIUM, FV, vol))
            buy_cap -= vol

        # MAKE
        if od.buy_orders:
            best_bid = max(od.buy_orders.keys())
        else:
            best_bid = FV - 10
        if od.sell_orders:
            best_ask = min(od.sell_orders.keys())
        else:
            best_ask = FV + 10

        our_bid = min(best_bid + 1, FV - 1)
        our_ask = max(best_ask - 1, FV + 1)

        if buy_cap > 0:
            orders.append(Order(self.OSMIUM, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(self.OSMIUM, our_ask, -sell_cap))

        return orders

    # ══════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT — Fixed accumulation
    # ══════════════════════════════════════════════════════════════

    def trade_pepper(self, od: OrderDepth, pos: int, timestamp: int, data: dict):
        orders = []
        LIMIT = self.PEPPER_LIMIT

        # FV tracking
        if "pepper_base" not in data:
            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders.keys())
                best_ask = min(od.sell_orders.keys())
                mid = (best_bid + best_ask) / 2
                data["pepper_base"] = mid - self.PEPPER_SLOPE * timestamp
            else:
                return orders, data

        fv = data["pepper_base"] + self.PEPPER_SLOPE * timestamp
        fv_int = int(round(fv))

        buy_cap = LIMIT - pos
        sell_cap = LIMIT + pos
        accumulated = data.get("pepper_accumulated", False)

        if not accumulated:
            # ── SMART ACCUMULATION ─────────────────────────────
            # OLD: sweep ALL levels → pays 9-11 above FV on L2/L3
            # NEW: only sweep L1 (best ask) + aggressive bid
            #   - Saves ~200-400 pts on spread cost
            #   - Fills in 5-8 ticks instead of 3 (misses <1 pt drift)
            #   - Net: much better entry price

            if buy_cap > 0 and od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                # Only take L1: the best ask price
                vol = min(-od.sell_orders[best_ask], buy_cap)
                orders.append(Order(self.PEPPER, best_ask, vol))
                buy_cap -= vol

            # Aggressive passive bid: penny the best bid
            if buy_cap > 0:
                if od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    orders.append(Order(self.PEPPER, best_bid + 1, buy_cap))
                else:
                    orders.append(Order(self.PEPPER, fv_int, buy_cap))

            if pos >= LIMIT:
                data["pepper_accumulated"] = True

        else:
            # ── HOLDING PHASE ──────────────────────────────────
            if buy_cap > 0:
                # Aggressive rebuy: sweep L1 if near FV
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    if best_ask <= fv_int + 2:
                        vol = min(-od.sell_orders[best_ask], buy_cap)
                        orders.append(Order(self.PEPPER, best_ask, vol))
                        buy_cap -= vol

                # Passive penny bid for remainder
                if buy_cap > 0 and od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    our_bid = min(best_bid + 1, fv_int)
                    orders.append(Order(self.PEPPER, our_bid, buy_cap))

            # Sell-side MM
            if pos >= self.PEPPER_MM_MIN_POS:
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                else:
                    best_ask = fv_int + 8

                mm_ask = max(best_ask - 1, fv_int + self.PEPPER_MM_ASK_FLOOR)
                available = pos - self.PEPPER_MM_MIN_POS
                mm_vol = min(available, self.PEPPER_MM_MAX_VOL)

                if mm_vol > 0 and sell_cap > 0:
                    mm_vol = min(mm_vol, sell_cap)
                    orders.append(Order(self.PEPPER, mm_ask, -mm_vol))

        return orders, data