from datamodel import Order, OrderDepth, TradingState
import json
import math


class Trader:
    """
    Prosperity 4 — Round 1: Intara

    EMERALDS + TOMATOES: V21 sin cambios.

    INTARIAN_PEPPER_ROOT — "quite steady" (como EMERALDS del tutorial)
        - Fair value fijo, predecible. El trend en datos históricos era
          convergencia al fair value real.
        - Estrategia: market making anclado al EMA con edge estrecho.
        - OFI corr=0.313 → shift al fair para mejorar timing.
        - Position limit: 80.

    ASH_COATED_OSMIUM — "volatile with hidden pattern"
        - Mean reverting alrededor de 10.000 (confirmado en datos).
        - Half-life=2.9 ticks, ACF=-0.495 → reversión muy rápida.
        - Spread=16 → edge=5, grandes márgenes por trade.
        - OFI corr=0.297 → shift útil para taking.
        - Position limit: 80.
    """

    POSITION_LIMITS = {
        "EMERALDS":             80,
        "TOMATOES":             80,
        "ASH_COATED_OSMIUM":    80,   # confirmado en descripción del round
        "INTARIAN_PEPPER_ROOT": 80,   # confirmado en descripción del round
    }

    CONFIG = {
        # ── Tutorial products (V21 sin cambios) ──────────────────────────────
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

        # ── INTARIAN_PEPPER_ROOT — steady (análogo a EMERALDS) ───────────────
        # "quite steady" → anchor-based making, no directional.
        # No sabemos el anchor exacto → usamos EMA con span largo
        # para seguir el fair sin sobrereaccionar.
        "INTARIAN_PEPPER_ROOT": {
            "anchor_fair":    None,    # desconocido → EMA
            "ema_span":       50,      # moderado: el fair es estable
            "edge":           2.0,     # spread=13 → cotizamos dentro
            "quote_size":     30,
            "inventory_skew": 0.02,
            "take_threshold": 3.0,     # tomamos si hay >3pts de edge
            "ofi_alpha":      1.5,     # shift OFI (corr=0.313)
        },

        # ── ASH_COATED_OSMIUM — volatile con patrón oculto ───────────────────
        # Anchor = 10.000 (confirmado en datos: mean price 10000.20)
        # Half-life=2.9t → taking agresivo
        # Spread=16 → edge=5 nos pone como mejor precio con margen
        "ASH_COATED_OSMIUM": {
            "anchor_fair":    10000.0, # confirmado en microestructura
            "edge":           5,
            "quote_size":     25,
            "take_threshold": 6,       # agresivo por half-life corta
            "inventory_skew": 0.04,
            "ofi_alpha":      2.0,     # shift OFI (corr=0.297)
        },
    }

    # ── utilidades ────────────────────────────────────────────────────────────

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
        if cfg.get("anchor_fair") is not None:
            return float(cfg["anchor_fair"])
        key   = f"{symbol}_ema"
        prev  = td.get(key, mid)
        alpha = 2.0 / (cfg["ema_span"] + 1.0)
        ema   = alpha * mid + (1.0 - alpha) * prev
        updates[key] = ema
        return float(ema)

    def _ofi(self, depth: OrderDepth) -> float:
        """OFI L1: corr(OFI, ret_t+1) = 0.30 en ambos productos."""
        best_bid, best_ask = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return 0.0
        bv  = depth.buy_orders.get(best_bid, 0)
        av  = abs(depth.sell_orders.get(best_ask, 0))
        tot = bv + av
        return (bv - av) / tot if tot > 0 else 0.0

    def _tick(self, symbol: str, td: dict, updates: dict) -> int:
        key = f"{symbol}_tick"
        t   = td.get(key, 0) + 1
        updates[key] = t
        return t

    # ── EMERALDS (V21 sin cambios) ────────────────────────────────────────────

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

    # ── TOMATOES (V21 sin cambios) ────────────────────────────────────────────

    def _tomatoes_orders(self, depth: OrderDepth, fair: float, mid: float,
                         buy_cap: int, sell_cap: int, position: int,
                         warmup: bool) -> list:
        orders = []
        cfg   = self.CONFIG["TOMATOES"]
        limit = self.POSITION_LIMITS["TOMATOES"]

        best_bid, best_ask = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        bv = sv = 0

        sorted_bids = sorted(depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.keys())
        bids_l23 = sum(depth.buy_orders[p] for p in sorted_bids[1:]) \
                   if len(sorted_bids) > 1 else 0
        asks_l23 = sum(depth.sell_orders[p] for p in sorted_asks[1:]) \
                   if len(sorted_asks) > 1 else 0
        total_l23 = bids_l23 + asks_l23
        spoof_imb = (bids_l23 - asks_l23) / total_l23 if total_l23 > 0 else 0

        spoof_shift = 0.0
        if spoof_imb <= -0.1:   spoof_shift =  2.0
        elif spoof_imb >= 0.15: spoof_shift = -1.5

        take_fair    = mid if warmup else fair
        dynamic_fair = take_fair + spoof_shift

        if best_ask <= dynamic_fair - cfg["take_threshold"] and buy_cap > 0:
            qty = min(cfg["quote_size"], buy_cap)
            orders.append(Order("TOMATOES", best_ask, qty)); bv += qty
        if best_bid >= dynamic_fair + cfg["take_threshold"] and sell_cap > 0:
            qty = min(cfg["quote_size"], sell_cap)
            orders.append(Order("TOMATOES", best_bid, -qty)); sv += qty

        if spread <= 8:
            if position > 15 and spoof_imb >= 0.15 and sell_cap - sv > 0:
                qty = min(cfg["unwind_size"], sell_cap - sv)
                orders.append(Order("TOMATOES", best_bid, -qty)); sv += qty
            elif position < -15 and spoof_imb <= -0.10 and buy_cap - bv > 0:
                qty = min(cfg["unwind_size"], buy_cap - bv)
                orders.append(Order("TOMATOES", best_ask, qty)); bv += qty

        if not warmup:
            making_fair = fair + spoof_shift
            reservation = making_fair - cfg["as_gamma"] * (position / limit)
            edge  = cfg["edge"]
            bid_q = min(int(math.floor(reservation - edge)), best_bid + 1)
            ask_q = max(int(math.ceil(reservation + edge)),  best_ask - 1)
            if bid_q >= ask_q: bid_q = ask_q - 1
            bid_q = min(bid_q, int(fair) + int(cfg["as_gamma"]))
            ask_q = max(ask_q, int(fair) - int(cfg["as_gamma"]))
            bqty = min(cfg["quote_size"], buy_cap - bv)
            if bqty > 0: orders.append(Order("TOMATOES", bid_q,  bqty))
            sqty = min(cfg["quote_size"], sell_cap - sv)
            if sqty > 0: orders.append(Order("TOMATOES", ask_q, -sqty))

        return orders

    # ── INTARIAN_PEPPER_ROOT — steady, análogo a EMERALDS ────────────────────

    def _pepper_orders(self, depth: OrderDepth, fair: float,
                       buy_cap: int, sell_cap: int, position: int,
                       ofi: float) -> list:
        """
        Fair value estable → market making simétrico centrado en EMA.
        OFI shift pequeño para mejorar el timing de los quotes.
        Taking cuando el precio se aleja más de take_threshold del fair.
        """
        orders = []
        cfg    = self.CONFIG["INTARIAN_PEPPER_ROOT"]
        sym    = "INTARIAN_PEPPER_ROOT"
        limit  = self.POSITION_LIMITS[sym]
        INT_FAIR = int(fair)
        bv = sv  = 0

        best_bid, best_ask = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return orders

        # Fair dinámico con OFI
        dynamic_fair = fair + ofi * cfg["ofi_alpha"]

        # === TAKING — igual que EMERALDS =====================================
        for ask_price in sorted(depth.sell_orders):
            if ask_price >= dynamic_fair - cfg["take_threshold"] \
                    or buy_cap - bv <= 0: break
            qty = min(abs(depth.sell_orders[ask_price]), buy_cap - bv)
            orders.append(Order(sym, ask_price, qty)); bv += qty

        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price <= dynamic_fair + cfg["take_threshold"] \
                    or sell_cap - sv <= 0: break
            qty = min(depth.buy_orders[bid_price], sell_cap - sv)
            orders.append(Order(sym, bid_price, -qty)); sv += qty

        # Limpieza al precio exacto
        net = position + bv - sv
        if net < 0 and INT_FAIR in depth.sell_orders:
            qty = min(abs(depth.sell_orders[INT_FAIR]), -net, buy_cap - bv)
            if qty > 0: orders.append(Order(sym, INT_FAIR, qty)); bv += qty
        if net > 0 and INT_FAIR in depth.buy_orders:
            qty = min(depth.buy_orders[INT_FAIR], net, sell_cap - sv)
            if qty > 0: orders.append(Order(sym, INT_FAIR, -qty)); sv += qty

        # === MAKING — simétrico con inventory skew ===========================
        net = position + bv - sv
        reservation = dynamic_fair - cfg["inventory_skew"] * net

        edge  = cfg["edge"]
        bid_q = min(int(math.floor(reservation - edge)), best_bid + 1)
        ask_q = max(int(math.ceil(reservation + edge)),  best_ask - 1)
        if bid_q >= ask_q: bid_q = ask_q - 1

        bqty = min(cfg["quote_size"], buy_cap - bv)
        if bqty > 0: orders.append(Order(sym, bid_q,  bqty))
        sqty = min(cfg["quote_size"], sell_cap - sv)
        if sqty > 0: orders.append(Order(sym, ask_q, -sqty))

        return orders

    # ── ASH_COATED_OSMIUM — mean reverting a 10.000 ──────────────────────────

    def _osmium_orders(self, depth: OrderDepth, fair: float,
                       buy_cap: int, sell_cap: int, position: int,
                       ofi: float) -> list:
        """
        Anchor = 10.000, spread=16, half-life=2.9t.
        Edge = 5 → somos el mejor precio del libro con ~3pts de margen.
        Taking agresivo en todos los niveles (no solo best).
        """
        orders = []
        cfg    = self.CONFIG["ASH_COATED_OSMIUM"]
        sym    = "ASH_COATED_OSMIUM"
        INT_FAIR = int(fair)
        bv = sv  = 0

        best_bid, best_ask = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return orders

        dynamic_fair = fair + ofi * cfg["ofi_alpha"]

        # === TAKING — walk the book ==========================================
        for ask_price in sorted(depth.sell_orders):
            if ask_price >= dynamic_fair - cfg["take_threshold"] \
                    or buy_cap - bv <= 0: break
            qty = min(abs(depth.sell_orders[ask_price]), buy_cap - bv)
            orders.append(Order(sym, ask_price, qty)); bv += qty

        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price <= dynamic_fair + cfg["take_threshold"] \
                    or sell_cap - sv <= 0: break
            qty = min(depth.buy_orders[bid_price], sell_cap - sv)
            orders.append(Order(sym, bid_price, -qty)); sv += qty

        net = position + bv - sv
        if net < 0 and INT_FAIR in depth.sell_orders:
            qty = min(abs(depth.sell_orders[INT_FAIR]), -net, buy_cap - bv)
            if qty > 0: orders.append(Order(sym, INT_FAIR, qty)); bv += qty
        if net > 0 and INT_FAIR in depth.buy_orders:
            qty = min(depth.buy_orders[INT_FAIR], net, sell_cap - sv)
            if qty > 0: orders.append(Order(sym, INT_FAIR, -qty)); sv += qty

        # === MAKING ==========================================================
        net = position + bv - sv
        reservation = dynamic_fair - cfg["inventory_skew"] * net

        edge  = cfg["edge"]
        bid_q = min(int(math.floor(reservation - edge)), best_bid + 1)
        ask_q = max(int(math.ceil(reservation + edge)),  best_ask - 1)
        if bid_q >= ask_q: bid_q = ask_q - 1

        bqty = min(cfg["quote_size"], buy_cap - bv)
        if bqty > 0: orders.append(Order(sym, bid_q,  bqty))
        sqty = min(cfg["quote_size"], sell_cap - sv)
        if sqty > 0: orders.append(Order(sym, ask_q, -sqty))

        return orders

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        td      = self._load_td(state)
        updates = {}

        for symbol in ("EMERALDS", "TOMATOES",
                        "ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"):
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
            ofi  = self._ofi(depth)

            if symbol == "EMERALDS":
                result[symbol] = self._emeralds_orders(
                    depth, fair, buy_cap, sell_cap, position)

            elif symbol == "TOMATOES":
                tick   = self._tick(symbol, td, updates)
                warmup = tick <= self.CONFIG["TOMATOES"]["warmup_ticks"]
                result[symbol] = self._tomatoes_orders(
                    depth, fair, mid, buy_cap, sell_cap, position, warmup)

            elif symbol == "INTARIAN_PEPPER_ROOT":
                result[symbol] = self._pepper_orders(
                    depth, fair, buy_cap, sell_cap, position, ofi)

            elif symbol == "ASH_COATED_OSMIUM":
                result[symbol] = self._osmium_orders(
                    depth, fair, buy_cap, sell_cap, position, ofi)

        merged = {**td, **updates}
        return result, conversions, json.dumps(merged)