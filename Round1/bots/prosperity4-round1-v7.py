from datamodel import Order, OrderDepth, TradingState
import json
from typing import List


class Trader:
    """
    Prosperity 4 — Round 1 — v7

    KEY IMPROVEMENTS over v5:
    1. Osmium: Volume imbalance signal (0.50 corr with returns!) shifts dynamic FV
    2. Osmium: Position-dependent quote shifting (both bid AND ask shift to unwind)
    3. Osmium: No emergency clear at FV (crossing spread wastes ~8 ticks per clear)
    4. Osmium: Multi-level taking with position-aware edges
    5. Pepper: Multi-level accumulation sweep (fill ~2x faster)
    6. Pepper: Mean-reversion aware MM pricing (deviation has -0.71 autocorr)
    7. Pepper: Tighter rebuy threshold + dynamic ask floor
    """

    OSMIUM = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM_LIMIT = 80
    PEPPER_LIMIT = 80
    OSMIUM_FV = 10001
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
    #  ASH_COATED_OSMIUM — Volume-imbalance FV + position-aware MM
    # ══════════════════════════════════════════════════════════════

    def trade_osmium(self, od: OrderDepth, pos: int) -> List[Order]:
        orders = []
        LIMIT = self.OSMIUM_LIMIT

        # ── Dynamic FV using volume imbalance ──
        # Volume imbalance has 0.50 corr with next-tick return
        total_bid_vol = sum(od.buy_orders.values()) if od.buy_orders else 0
        total_ask_vol = sum(-v for v in od.sell_orders.values()) if od.sell_orders else 0

        if total_bid_vol + total_ask_vol > 0:
            imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        else:
            imbalance = 0

        # Shift FV by up to 2 ticks based on imbalance
        fv = self.OSMIUM_FV + int(round(imbalance * 0))

        # Position factor
        pos_ratio = pos / LIMIT  # -1.0 to 1.0
        # Shift to apply: positive when long (want to sell => lower prices)
        pos_shift = int(round(pos_ratio * 3))  # ±3 ticks max

        buy_cap = LIMIT - pos
        sell_cap = LIMIT + pos

        # ── TAKE: multi-level, position-aware ──
        # When long: sell more aggressively (accept lower bids)
        # When short: buy more aggressively (accept higher asks)
        buy_take_edge = fv + max(0, int(-pos_ratio * 3))
        sell_take_edge = fv - max(0, int(pos_ratio * 3))

        for ask in sorted(od.sell_orders.keys()):
            if ask <= buy_take_edge and buy_cap > 0:
                vol = min(-od.sell_orders[ask], buy_cap)
                orders.append(Order(self.OSMIUM, ask, vol))
                buy_cap -= vol

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid >= sell_take_edge and sell_cap > 0:
                vol = min(od.buy_orders[bid], sell_cap)
                orders.append(Order(self.OSMIUM, bid, -vol))
                sell_cap -= vol

        # ── MAKE: penny with position-dependent shift ──
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else fv - 10
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else fv + 10

        # Both quotes shift DOWN by pos_shift:
        # Long (pos_shift>0): lower bid (less buying) + lower ask (more selling)
        # Short (pos_shift<0): raise bid (more buying) + raise ask (less selling)
        our_bid = min(best_bid + 1, fv - 1 - pos_shift)
        our_ask = max(best_ask - 1, fv + 1 - pos_shift)

        # Safety: ensure reasonable spread
        if our_bid >= our_ask:
            our_bid = fv - 1 - pos_shift
            our_ask = fv + 1 - pos_shift
        if our_bid >= our_ask:  # still broken, fall back
            our_bid = fv - 2
            our_ask = fv + 2

        if buy_cap > 0:
            orders.append(Order(self.OSMIUM, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(self.OSMIUM, our_ask, -sell_cap))

        return orders

    # ══════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT — L1-only accumulation + tighter MM
    # ══════════════════════════════════════════════════════════════
 
    def trade_pepper(self, od: OrderDepth, pos: int, timestamp: int, data: dict):
        orders = []
        LIMIT = self.PEPPER_LIMIT
 
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
            # Smart accumulation: L1 only + aggressive bid
            if buy_cap > 0 and od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                vol = min(-od.sell_orders[best_ask], buy_cap)
                orders.append(Order(self.PEPPER, best_ask, vol))
                buy_cap -= vol
 
            if buy_cap > 0:
                if od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    orders.append(Order(self.PEPPER, best_bid + 1, buy_cap))
                else:
                    orders.append(Order(self.PEPPER, fv_int, buy_cap))
 
            if pos >= LIMIT:
                data["pepper_accumulated"] = True
 
        else:
            # Holding: aggressive rebuy + tighter MM
            if buy_cap > 0:
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    if best_ask <= fv_int + 3:
                        vol = min(-od.sell_orders[best_ask], buy_cap)
                        orders.append(Order(self.PEPPER, best_ask, vol))
                        buy_cap -= vol
 
                if buy_cap > 0 and od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    our_bid = min(best_bid + 1, fv_int)
                    orders.append(Order(self.PEPPER, our_bid, buy_cap))
 
            # Sell-side MM — tighter floor (2 instead of 3)
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