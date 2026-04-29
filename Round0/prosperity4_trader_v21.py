from datamodel import Order, OrderDepth, TradingState
import json
import math


class Trader:
    """
    Prosperity 4 Trader — V15.2 (Fusion: Aggressive AS + Spoofing + Warmup Fix)

    Base: V15.1 aggressive (γ=5, take=2, unwind=15) → 17k/day on 10k ticks
    + Spoofing dynamic_fair (from V15.1 spoof-aware) → +360 pts
    + Warmup fix: during first 10 ticks, francotirador uses MID not EMA
      (EMA with span=50 takes ~25 ticks to converge → toxic buys at start)

    Submission PnL should improve by avoiding the initial position bleed.
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
            "anchor_fair":    None,
            "ema_span":       50,
            "edge":           2.0,
            "quote_size":     20,
            "as_gamma":       5.0,
            "take_threshold": 2.0,
            "warmup_ticks":   10,  
            "unwind_thresh":  15,
            "unwind_size":    10,
        },
    }

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

    # ── EMERALDS (unchanged) ─────────────────────────────────────────────────

    def _emeralds_orders(self, depth: OrderDepth, fair: float,
                         buy_cap: int, sell_cap: int, position: int) -> list:
        orders = []
        cfg    = self.CONFIG["EMERALDS"]
        INT_FAIR = int(fair)
        bv = sv = 0

        for ask_price in sorted(depth.sell_orders):
            if ask_price >= fair or buy_cap - bv <= 0: break
            qty = min(abs(depth.sell_orders[ask_price]), buy_cap - bv)
            orders.append(Order("EMERALDS", ask_price, qty)); bv += qty

        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price <= fair or sell_cap - sv <= 0: break
            qty = min(depth.buy_orders[bid_price], sell_cap - sv)
            orders.append(Order("EMERALDS", bid_price, -qty)); sv += qty

        net = position + bv - sv
        if net < 0 and INT_FAIR in depth.sell_orders:
            qty = min(abs(depth.sell_orders[INT_FAIR]), -net, buy_cap - bv)
            if qty > 0: orders.append(Order("EMERALDS", INT_FAIR, qty)); bv += qty
        if net > 0 and INT_FAIR in depth.buy_orders:
            qty = min(depth.buy_orders[INT_FAIR], net, sell_cap - sv)
            if qty > 0: orders.append(Order("EMERALDS", INT_FAIR, -qty)); sv += qty

        net = position + bv - sv
        bid_edge = ask_edge = cfg["edge"]
        if net > 60: ask_edge = 1.0
        elif net < -60: bid_edge = 1.0

        best_bid, best_ask = self._best_quotes(depth)
        reservation = fair - cfg["inventory_skew"] * (
            self.POSITION_LIMITS["EMERALDS"] - (buy_cap - bv) - (sell_cap - sv))
        bid_q = min(int(math.floor(reservation - bid_edge)), best_bid + 1)
        ask_q = max(int(math.ceil(reservation + ask_edge)),  best_ask - 1)
        if bid_q >= ask_q: bid_q = ask_q - 1

        bqty = min(cfg["quote_size"], buy_cap - bv)
        if bqty > 0: orders.append(Order("EMERALDS", bid_q, bqty))
        sqty = min(cfg["quote_size"], sell_cap - sv)
        if sqty > 0: orders.append(Order("EMERALDS", ask_q, -sqty))
        return orders

    # ── TOMATOES (V15.2: aggressive + spoof + warmup fix) ────────────────────

    def _tomatoes_orders(self, depth: OrderDepth, fair: float, mid: float,
                         buy_cap: int, sell_cap: int, position: int,
                         warmup: bool) -> list:
        orders = []
        cfg    = self.CONFIG["TOMATOES"]
        limit  = self.POSITION_LIMITS["TOMATOES"]

        best_bid, best_ask = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        bv = sv = 0

        # === 1. SPOOF DETECTION ===
        sorted_bids = sorted(depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.keys())
        bids_l23 = sum(depth.buy_orders[p] for p in sorted_bids[1:]) if len(sorted_bids) > 1 else 0
        asks_l23 = sum(depth.sell_orders[p] for p in sorted_asks[1:]) if len(sorted_asks) > 1 else 0
        total_l23 = bids_l23 + asks_l23
        spoof_imb = (bids_l23 - asks_l23) / total_l23 if total_l23 > 0 else 0

        # === 2. DYNAMIC FAIR (spoof shift) ===
        spoof_shift = 0.0
        if spoof_imb <= -0.1:
            spoof_shift = 2.0
        elif spoof_imb >= 0.15:
            spoof_shift = -1.5

        # WARMUP FIX: during early ticks, use mid instead of EMA for taking
        # EMA with span=50 lags 3-5 pts behind price during first ~25 ticks
        # This causes toxic buys that create +15 to +20 initial position
        take_fair = mid if warmup else fair
        dynamic_fair = take_fair + spoof_shift

        # === 3. FRANCOTIRADOR (uses dynamic_fair with warmup fix) ===
        if best_ask <= dynamic_fair - cfg["take_threshold"] and buy_cap > 0:
            qty = min(cfg["quote_size"], buy_cap)
            orders.append(Order("TOMATOES", best_ask, qty))
            bv += qty

        if best_bid >= dynamic_fair + cfg["take_threshold"] and sell_cap > 0:
            qty = min(cfg["quote_size"], sell_cap)
            orders.append(Order("TOMATOES", best_bid, -qty))
            sv += qty

        # === 4. UNWIND (spoof-aware, asymmetric) ===
        unwind_size = cfg["unwind_size"]
        if spread <= 8:
            if position > 15 and spoof_imb >= 0.15 and sell_cap - sv > 0:
                qty = min(unwind_size, sell_cap - sv)
                orders.append(Order("TOMATOES", best_bid, -qty))
                sv += qty
            elif position < -15 and spoof_imb <= -0.10 and buy_cap - bv > 0:
                qty = min(unwind_size, buy_cap - bv)
                orders.append(Order("TOMATOES", best_ask, qty))
                bv += qty

        # === 5. MAKING: AS + spoof (only after warmup) ===
        if not warmup:
            # Making uses EMA fair (not mid), shifted by spoof
            making_fair = fair + spoof_shift
            reservation = making_fair - cfg["as_gamma"] * (position / limit)

            edge = cfg["edge"]
            bid_q = min(int(math.floor(reservation - edge)), best_bid + 1)
            ask_q = max(int(math.ceil(reservation + edge)),  best_ask - 1)
            if bid_q >= ask_q: bid_q = ask_q - 1

            bid_q = min(bid_q, int(fair) + int(cfg["as_gamma"]))
            ask_q = max(ask_q, int(fair) - int(cfg["as_gamma"]))

            bqty = min(cfg["quote_size"], buy_cap - bv)
            if bqty > 0: orders.append(Order("TOMATOES", bid_q, bqty))
            sqty = min(cfg["quote_size"], sell_cap - sv)
            if sqty > 0: orders.append(Order("TOMATOES", ask_q, -sqty))

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
                warmup   = tick <= self.CONFIG["TOMATOES"]["warmup_ticks"]

                result[symbol] = self._tomatoes_orders(
                    depth, fair, mid, buy_cap, sell_cap, position, warmup)

        merged = {**td, **updates}
        return result, conversions, json.dumps(merged)