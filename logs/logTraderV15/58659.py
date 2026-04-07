from datamodel import Order, OrderDepth, TradingState
import json
import math


class Trader:
    """
    Prosperity 4 Trader — V15.1 (AS Skew — PnL-optimised)

    Parameter set: γ=7, unwind_thresh=35, unwind_size=10
    Backtest (2 days): PnL=+30,749  Sharpe=6.77  mean|pos|=20.9  atLim=0%

    AS reservation price: r = fair − 7 × (pos / 80)
      pos=+35 → shift = −3.06 ticks  (ask moves well inside mid, fills faster)
      pos=+80 → shift = −7.0  ticks  (ask at fair−5, very aggressive sell)

    unwind_thresh=35: aggressive spread-crossing fires earlier than V15.0 (was 50).
    Keeps mean inventory at ~21 units vs ~33 in V15.0.
    Chosen for best raw PnL in the parameter sweep (plateau across all γ at thresh=35).

    EMERALDS: identical to V6 — near-perfectly neutral (mean pos=−0.07), unchanged.
    Position limits: EMERALDS=80, TOMATOES=80
    """

    POSITION_LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    CONFIG = {
        "EMERALDS": {
            "anchor_fair":    10000.0,
            "ema_span":       20,
            "edge":           2.0,
            "quote_size":     40,
            "inventory_skew": 0.02,   # used only for reservation shift (light)
        },
        "TOMATOES": {
            "anchor_fair":    None,
            "ema_span":       50,
            "edge":           2.0,
            "quote_size":     20,
            "as_gamma":       7.0,    # reservation shift at full limit (ticks)
            "take_threshold": 3.0,
            "warmup_ticks":   50,
            "unwind_thresh":  35,     # |pos| to trigger aggressive spread-crossing
            "unwind_size":    10,     # units per aggressive unwind
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
        cfg   = self.CONFIG[symbol]
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

        # 1. Sweep mispriced asks
        for ask_price in sorted(depth.sell_orders):
            if ask_price >= fair or buy_cap - bv <= 0:
                break
            qty = min(abs(depth.sell_orders[ask_price]), buy_cap - bv)
            orders.append(Order("EMERALDS", ask_price, qty))
            bv += qty

        # 2. Sweep mispriced bids
        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price <= fair or sell_cap - sv <= 0:
                break
            qty = min(depth.buy_orders[bid_price], sell_cap - sv)
            orders.append(Order("EMERALDS", bid_price, -qty))
            sv += qty

        # 3. At-fair unwinding
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

        # 4. Inventory tightening
        net = position + bv - sv
        bid_edge = ask_edge = cfg["edge"]
        if net > 60:  ask_edge = 1.0
        elif net < -60: bid_edge = 1.0

        # 5. Passive quotes with reservation skew
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

    # ── TOMATOES (V15 SUPREMO: Reversion + Spoof-Aware AS Skew) ───────────────

    def _tomatoes_orders(self, depth: OrderDepth, fair: float,
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

        # === 1. FRANCOTIRADOR CLÁSICO (Para salvar el PnL del Día 1) ===
        # Si el mercado se desvía de la media por ineficiencia pura, ataca.
        if best_ask <= fair - cfg["take_threshold"] and buy_cap > 0:
            qty = min(cfg["quote_size"], buy_cap)
            orders.append(Order("TOMATOES", best_ask, qty))
            bv += qty

        if best_bid >= fair + cfg["take_threshold"] and sell_cap > 0:
            qty = min(cfg["quote_size"], sell_cap)
            orders.append(Order("TOMATOES", best_bid, -qty))
            sv += qty

        # === 2. DETECCIÓN DE SPOOFING (El Radar) ===
        sorted_bids = sorted(depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.keys())
        bids_l23 = sum(depth.buy_orders[p] for p in sorted_bids[1:]) if len(sorted_bids) > 1 else 0
        asks_l23 = sum(depth.sell_orders[p] for p in sorted_asks[1:]) if len(sorted_asks) > 1 else 0
        total_l23 = bids_l23 + asks_l23
        spoof_imb = (bids_l23 - asks_l23) / total_l23 if total_l23 > 0 else 0

        # === 3. UNWIND INTELIGENTE (Cortar la sangría del Avellaneda) ===
        # El AS original cruzaba el spread perdiendo dinero solo porque abs(pos) > 35.
        # Ahora SOLO evacuamos pagando el spread si el spoofing nos ataca EN CONTRA.
        unwind_size = cfg["unwind_size"]

        if spread <= 8:
            # Si estamos LARGOS (> 20) y el mercado va a CAER (Señal >= 0.15) -> Evacuamos rápido
            if position > 20 and spoof_imb >= 0.15 and sell_cap - sv > 0:
                qty = min(unwind_size, sell_cap - sv)
                orders.append(Order("TOMATOES", best_bid, -qty))
                sv += qty
            # Si estamos CORTOS (< -20) y el mercado va a SUBIR (Señal <= -0.15) -> Evacuamos rápido
            elif position < -20 and spoof_imb <= -0.15 and buy_cap - bv > 0:
                qty = min(unwind_size, buy_cap - bv)
                orders.append(Order("TOMATOES", best_ask, qty))
                bv += qty

        # === 4. MAKING: AVELLANEDA-STOIKOV "SPOOF-AWARE" (Unwind Rentable) ===
        if not warmup:
            # MAGIA AQUÍ: Desplazamos el "fair" basándonos en la predicción del spoofing.
            # En tu V8 esto lo llamabas "Spoof shift +2.0 on fair_making".
            spoof_shift = 0.0
            if spoof_imb <= -0.10:
                spoof_shift = 2.0  # El mercado subirá: Subimos nuestros precios (evita malvender)
            elif spoof_imb >= 0.10:
                spoof_shift = -2.0 # El mercado bajará: Bajamos nuestros precios (evita malcomprar)

            # Reservation price combina la predicción (spoof_shift) y la defensa AS (gamma)
            reservation = (fair + spoof_shift) - cfg["as_gamma"] * (position / limit)

            edge = cfg["edge"]
            bid_q = min(int(math.floor(reservation - edge)), best_bid + 1)
            ask_q = max(int(math.ceil(reservation + edge)),  best_ask - 1)

            if bid_q >= ask_q:
                bid_q = ask_q - 1

            # Cortafuegos basado en el fair real (sin el shift) para no cotizar locuras
            bid_q = min(bid_q, int(fair) + int(cfg["as_gamma"]))
            ask_q = max(ask_q, int(fair) - int(cfg["as_gamma"]))

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
                warmup   = tick <= self.CONFIG["TOMATOES"]["warmup_ticks"]

                result[symbol] = self._tomatoes_orders(
                    depth, fair, buy_cap, sell_cap, position, warmup)

        merged = {**td, **updates}
        return result, conversions, json.dumps(merged)