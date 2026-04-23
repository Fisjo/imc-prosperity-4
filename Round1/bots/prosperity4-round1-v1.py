from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json
import math


class Trader:
    """
    Round 1 strategy for two products:

    ASH_COATED_OSMIUM (stationary, FV = 10000)
      - Take/Clear/Make framework
      - Take: sweep mispriced levels on both sides of FV
      - Clear: 0-EV orders at FV to free position capacity
      - Make: penny the best bot level, never quote inside FV

    INTARIAN_PEPPER_ROOT (deterministic linear trend)
      - FV = base + 0.001 * timestamp (drift of +1000 per day)
      - Phase 1: sweep all asks to accumulate to max position (80) ASAP
      - Phase 2: hold for drift appreciation (~80,000 PnL over full session)
      - Phase 2 also includes optional sell-side MM for extra ~2-3% PnL
    """

    # ── Product names ──────────────────────────────────────────────
    OSMIUM = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"

    # ── Position limits ────────────────────────────────────────────
    OSMIUM_LIMIT = 80
    PEPPER_LIMIT = 80

    # ── Osmium parameters ──────────────────────────────────────────
    OSMIUM_FV = 10000

    # ── Pepper parameters ──────────────────────────────────────────
    PEPPER_SLOPE = 0.001          # price increase per timestamp unit
    PEPPER_MM_ENABLE = True       # toggle sell-side MM on/off
    PEPPER_MM_MIN_POS = 60        # only MM when position >= this
    PEPPER_MM_MAX_VOL = 10        # max units to sell per tick for MM
    PEPPER_MM_ASK_FLOOR = 4       # minimum edge above FV for asks (conservative)

    def run(self, state: TradingState):
        result = {}

        # Load persistent state from previous tick
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
    #  ASH_COATED_OSMIUM  --  Stationary asset, FV = 10,000
    # ══════════════════════════════════════════════════════════════

    def trade_osmium(self, od: OrderDepth, pos: int) -> List[Order]:
        orders = []
        FV = self.OSMIUM_FV
        LIMIT = self.OSMIUM_LIMIT

        buy_cap = LIMIT - pos
        sell_cap = LIMIT + pos

        # ── PHASE 1: TAKE ──────────────────────────────────────
        # Buy everything offered below FV
        for ask in sorted(od.sell_orders.keys()):
            if ask <= FV - 1 and buy_cap > 0:
                vol = min(-od.sell_orders[ask], buy_cap)
                orders.append(Order(self.OSMIUM, ask, vol))
                buy_cap -= vol

        # Sell everything bid above FV
        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid >= FV + 1 and sell_cap > 0:
                vol = min(od.buy_orders[bid], sell_cap)
                orders.append(Order(self.OSMIUM, bid, -vol))
                sell_cap -= vol

        # ── PHASE 2: CLEAR (0-EV at FV) ───────────────────────
        # Free position capacity by placing orders at fair value
        if pos > 0 and sell_cap > 0:
            vol = min(pos, sell_cap)
            orders.append(Order(self.OSMIUM, FV, -vol))
            sell_cap -= vol
        elif pos < 0 and buy_cap > 0:
            vol = min(-pos, buy_cap)
            orders.append(Order(self.OSMIUM, FV, vol))
            buy_cap -= vol

        # ── PHASE 3: MAKE (penny the bots) ────────────────────
        # Penny the best bot level, but never quote inside FV
        if od.buy_orders:
            best_bid = max(od.buy_orders.keys())
        else:
            best_bid = FV - 9  # fallback if book empty

        if od.sell_orders:
            best_ask = min(od.sell_orders.keys())
        else:
            best_ask = FV + 9  # fallback if book empty

        our_bid = min(best_bid + 1, FV - 1)
        our_ask = max(best_ask - 1, FV + 1)

        if buy_cap > 0:
            orders.append(Order(self.OSMIUM, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(self.OSMIUM, our_ask, -sell_cap))

        return orders

    # ══════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT  --  Deterministic linear trend
    # ══════════════════════════════════════════════════════════════

    def trade_pepper(self, od: OrderDepth, pos: int, timestamp: int, data: dict):
        orders = []
        LIMIT = self.PEPPER_LIMIT

        # ── FV TRACKING ────────────────────────────────────────
        # On first tick with a valid book, record the base price.
        # FV(t) = base + 0.001 * t
        if "pepper_base" not in data:
            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders.keys())
                best_ask = min(od.sell_orders.keys())
                mid = (best_bid + best_ask) / 2
                data["pepper_base"] = mid - self.PEPPER_SLOPE * timestamp
            else:
                return orders, data  # no book yet, skip

        fv = data["pepper_base"] + self.PEPPER_SLOPE * timestamp
        fv_int = int(round(fv))

        buy_cap = LIMIT - pos
        sell_cap = LIMIT + pos

        accumulated = data.get("pepper_accumulated", False)

        if not accumulated:
            # ── ACCUMULATION PHASE ─────────────────────────────
            # Goal: get to position 80 as fast as possible.
            # Sweep all ask levels (L1, L2, L3) regardless of price.
            # The drift (~1000/session) makes any buy profitable.

            if buy_cap > 0 and od.sell_orders:
                for ask in sorted(od.sell_orders.keys()):
                    if buy_cap <= 0:
                        break
                    vol = min(-od.sell_orders[ask], buy_cap)
                    orders.append(Order(self.PEPPER, ask, vol))
                    buy_cap -= vol

            # Also post a passive penny bid for any remaining capacity
            if buy_cap > 0 and od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                orders.append(Order(self.PEPPER, best_bid + 1, buy_cap))

            # Transition: mark as accumulated once we reach max position
            if pos >= LIMIT:
                data["pepper_accumulated"] = True

        else:
            # ── HOLDING PHASE ──────────────────────────────────
            # Position should be at or near 80. Drift does the work.
            # Two sub-tasks:
            #   a) Rebuy if position dropped (from MM sells) -- PASSIVE ONLY
            #   b) Optional sell-side MM for extra spread capture

            # a) Passive rebuy to get back to 80
            if buy_cap > 0 and od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                our_bid = min(best_bid + 1, fv_int - 1)
                orders.append(Order(self.PEPPER, our_bid, buy_cap))

            # b) Sell-side market making
            if self.PEPPER_MM_ENABLE and pos >= self.PEPPER_MM_MIN_POS:
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                else:
                    best_ask = fv_int + 8

                mm_ask = max(best_ask - 1, fv_int + self.PEPPER_MM_ASK_FLOOR)
                mm_vol = min(pos - self.PEPPER_MM_MIN_POS, self.PEPPER_MM_MAX_VOL)

                if mm_vol > 0 and sell_cap > 0:
                    mm_vol = min(mm_vol, sell_cap)
                    orders.append(Order(self.PEPPER, mm_ask, -mm_vol))

        return orders, data