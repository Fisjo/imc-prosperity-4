from datamodel import Order, OrderDepth, TradingState
import json
import math


class Trader:
    """
    Prosperity 4 Trader — V22 (V20 structure + V21 aggression in Phase 3)

    Three-phase TOMATOES strategy:
      Phase 1 (ticks 1-50):    EMA warmup — sniper only, no maker/unwind
      Phase 2 (ticks 51-1000): V15.1 mode — conservative maker + simple unwind
                                  as_gamma=7, take_threshold=3.0, unwind_thresh=35
      Phase 3 (ticks 1001+):   Aggressive mode — V21 params + spoof awareness
                                  as_gamma=5, take_threshold=2.0, dynamic_fair sniper
                                  spoof-conditional unwind at pos>15

    Rationale:
      - V20 is stable but leaves edge on the table in phase 3 (conservative params)
      - V21 is aggressive but has a 33% max drawdown due to no real warmup
      - V22: V20's 1000-tick warmup protects the submission period, V21's aggression
        activates only after spoof signals are reliable → best of both worlds

    EMERALDS: identical to V6 — unchanged across all versions.
    Position limits: EMERALDS=80, TOMATOES=80
    """

    POSITION_LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    CONFIG = {
        "EMERALDS": {
            "anchor_fair":    10000.0,
            "ema_span":       20,
            "edge":           2.0,
            "quote_size":     40,
            "inventory_skew": 0.02,
        },
        "TOMATOES": {
            "anchor_fair":        None,
            "ema_span":           50,
            "edge":               2.0,
            "quote_size":         20,
            # Phase 2 params (V15.1 conservative)
            "as_gamma":           7.0,
            "take_threshold":     3.0,
            "unwind_thresh":      35,
            # Phase 3 params (V21 aggressive)
            "as_gamma_p3":        5.0,
            "take_threshold_p3":  2.0,
            # Warmup boundaries
            "ema_warmup_ticks":   50,    # phase 1 → 2
            "spoof_warmup_ticks": 1000,  # phase 2 → 3
            "unwind_size":        10,
        },
    }

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_td(self, state: TradingState) -> dict:
        if not state.traderData:
            return {}
        try:
            return json.loads(state.traderData)
        except Exception:
            return {}

    def _best_quotes(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    def _ema(self, symbol: str, mid: float, td: dict, updates: dict) -> float:
        cfg = self.CONFIG[symbol]
        if cfg["anchor_fair"] is not None:
            return float(cfg["anchor_fair"])
        key   = f"{symbol}_ema"
        prev  = td.get(key, mid)
        alpha = 2.0 / (cfg["ema_span"] + 1.0)
        ema   = alpha * mid + (1.0 - alpha) * prev
        updates[key] = ema
        return float(ema)

    # ── EMERALDS (V6, unchanged) ──────────────────────────────────────────────

    def _emeralds_orders(self, depth: OrderDepth, fair: float,
                         buy_cap: int, sell_cap: int, position: int) -> list:
        orders = []
        cfg    = self.CONFIG["EMERALDS"]
        INT_FAIR = int(fair)
        bv = sv = 0

        for ask_price in sorted(depth.sell_orders):
            if ask_price >= fair or buy_cap - bv <= 0:
                break
            qty = min(abs(depth.sell_orders[ask_price]), buy_cap - bv)
            orders.append(Order("EMERALDS", ask_price, qty))
            bv += qty

        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price <= fair or sell_cap - sv <= 0:
                break
            qty = min(depth.buy_orders[bid_price], sell_cap - sv)
            orders.append(Order("EMERALDS", bid_price, -qty))
            sv += qty

        net = position + bv - sv
        if net < 0 and INT_FAIR in depth.sell_orders:
            qty = min(abs(depth.sell_orders[INT_FAIR]), -net, buy_cap - bv)
            if qty > 0:
                orders.append(Order("EMERALDS", INT_FAIR, qty))
                bv += qty
        if net > 0 and INT_FAIR in depth.buy_orders:
            qty = min(depth.buy_orders[INT_FAIR], net, sell_cap - sv)
            if qty > 0:
                orders.append(Order("EMERALDS", INT_FAIR, -qty))
                sv += qty

        net = position + bv - sv
        bid_edge = ask_edge = cfg["edge"]
        if net > 60:  ask_edge = 1.0
        elif net < -60: bid_edge = 1.0

        best_bid, best_ask = self._best_quotes(depth)
        reservation = fair - cfg["inventory_skew"] * (
            self.POSITION_LIMITS["EMERALDS"] - (buy_cap - bv) - (sell_cap - sv)
        )
        bid_q = min(int(math.floor(reservation - bid_edge)), best_bid + 1)
        ask_q = max(int(math.ceil(reservation + ask_edge)),  best_ask - 1)
        if bid_q >= ask_q:
            bid_q = ask_q - 1

        bqty = min(cfg["quote_size"], buy_cap - bv)
        if bqty > 0:
            orders.append(Order("EMERALDS", bid_q, bqty))
        sqty = min(cfg["quote_size"], sell_cap - sv)
        if sqty > 0:
            orders.append(Order("EMERALDS", ask_q, -sqty))

        return orders

    # ── TOMATOES (V22: phased warmup + aggressive phase 3) ───────────────────

    def _tomatoes_orders(self, depth: OrderDepth, fair: float,
                         buy_cap: int, sell_cap: int, position: int,
                         tick: int) -> list:
        orders = []
        cfg    = self.CONFIG["TOMATOES"]
        limit  = self.POSITION_LIMITS["TOMATOES"]

        best_bid, best_ask = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        bv = sv = 0

        # Phase flags
        ema_warm    = tick <= cfg["ema_warmup_ticks"]    # phase 1
        spoof_ready = tick > cfg["spoof_warmup_ticks"]   # phase 3

        # ── Spoof radar (computed always, used only in phase 3) ──────────────
        sorted_bids = sorted(depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.keys())
        bids_l23 = sum(depth.buy_orders[p] for p in sorted_bids[1:]) if len(sorted_bids) > 1 else 0
        asks_l23 = sum(depth.sell_orders[p] for p in sorted_asks[1:]) if len(sorted_asks) > 1 else 0
        total_l23 = bids_l23 + asks_l23
        spoof_imb = (bids_l23 - asks_l23) / total_l23 if total_l23 > 0 else 0

        spoof_shift = 0.0
        if spoof_ready:
            if spoof_imb <= -0.1:
                spoof_shift = 2.0
            elif spoof_imb >= 0.15:
                spoof_shift = -1.5

        dynamic_fair = fair + spoof_shift

        # ── Sniper ───────────────────────────────────────────────────────────
        # Phase 3: dynamic_fair + aggressive threshold (V21)
        # Phases 1 & 2: raw fair + conservative threshold
        if spoof_ready:
            sniper_fair = dynamic_fair
            sniper_thr  = cfg["take_threshold_p3"]   # 2.0
        else:
            sniper_fair = fair
            sniper_thr  = cfg["take_threshold"]       # 3.0

        if best_ask <= sniper_fair - sniper_thr and buy_cap > 0:
            qty = min(cfg["quote_size"], buy_cap)
            orders.append(Order("TOMATOES", best_ask, qty))
            bv += qty

        if best_bid >= sniper_fair + sniper_thr and sell_cap > 0:
            qty = min(cfg["quote_size"], sell_cap)
            orders.append(Order("TOMATOES", best_bid, -qty))
            sv += qty

        # ── Unwind ───────────────────────────────────────────────────────────
        if not ema_warm:
            if spoof_ready:
                # Phase 3: spoof-conditional unwind (V21 threshold=15)
                if spread <= 8:
                    if position > 15 and spoof_imb >= 0.15 and sell_cap - sv > 0:
                        qty = min(cfg["unwind_size"], sell_cap - sv)
                        orders.append(Order("TOMATOES", best_bid, -qty))
                        sv += qty
                    elif position < -15 and spoof_imb <= -0.10 and buy_cap - bv > 0:
                        qty = min(cfg["unwind_size"], buy_cap - bv)
                        orders.append(Order("TOMATOES", best_ask, qty))
                        bv += qty
            else:
                # Phase 2: simple unwind (V15.1 threshold=35)
                if abs(position) > cfg["unwind_thresh"] and spread <= 8:
                    if position > 0 and sell_cap - sv > 0:
                        qty = min(cfg["unwind_size"], sell_cap - sv)
                        orders.append(Order("TOMATOES", best_bid, -qty))
                        sv += qty
                    elif position < 0 and buy_cap - bv > 0:
                        qty = min(cfg["unwind_size"], buy_cap - bv)
                        orders.append(Order("TOMATOES", best_ask, qty))
                        bv += qty

        # ── Maker ────────────────────────────────────────────────────────────
        if not ema_warm:
            if spoof_ready:
                # Phase 3: dynamic_fair + aggressive gamma (V21)
                reservation = dynamic_fair - cfg["as_gamma_p3"] * (position / limit)
            else:
                # Phase 2: raw fair + conservative gamma (V15.1)
                reservation = fair - cfg["as_gamma"] * (position / limit)

            edge = cfg["edge"]
            bid_q = min(int(math.floor(reservation - edge)), best_bid + 1)
            ask_q = max(int(math.ceil(reservation + edge)),  best_ask - 1)
            if bid_q >= ask_q:
                bid_q = ask_q - 1

            gamma_guard = int(cfg["as_gamma"])
            bid_q = min(bid_q, int(fair) + gamma_guard)
            ask_q = max(ask_q, int(fair) - gamma_guard)

            bqty = min(cfg["quote_size"], buy_cap - bv)
            if bqty > 0:
                orders.append(Order("TOMATOES", bid_q, bqty))
            sqty = min(cfg["quote_size"], sell_cap - sv)
            if sqty > 0:
                orders.append(Order("TOMATOES", ask_q, -sqty))

        return orders

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        td      = self._load_td(state)
        updates = {}

        for symbol in ("EMERALDS", "TOMATOES"):
            if symbol not in state.order_depths:
                continue

            depth    = state.order_depths[symbol]
            position = state.position.get(symbol, 0)
            limit    = self.POSITION_LIMITS[symbol]
            buy_cap  = limit - position
            sell_cap = limit + position

            best_bid, best_ask = self._best_quotes(depth)
            if best_bid is None or best_ask is None:
                result[symbol] = []
                continue

            mid  = 0.5 * (best_bid + best_ask)
            fair = self._ema(symbol, mid, td, updates)

            if symbol == "EMERALDS":
                result[symbol] = self._emeralds_orders(
                    depth, fair, buy_cap, sell_cap, position)
            else:
                tick_key = "TOMATOES_tick"
                tick     = td.get(tick_key, 0) + 1
                updates[tick_key] = tick

                result[symbol] = self._tomatoes_orders(
                    depth, fair, buy_cap, sell_cap, position, tick)

        merged = {**td, **updates}
        return result, conversions, json.dumps(merged)
