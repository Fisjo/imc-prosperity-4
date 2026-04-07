from datamodel import Order, OrderDepth, TradingState
import math
from typing import Dict, List

class Trader:
    """
    Prosperity 4 Trader — Hybrid Version
    - EMERALDS: V15 Logic (Anchored Market Making + Sweeping)
    - TOMATOES: Regression Predator (Spoofing L2/L3 + L1 Imbalance + Avellaneda-Stoikov)
    - Memoryless: Fast execution, 0 dependencies on traderData.
    """

    POSITION_LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    CONFIG = {
        "EMERALDS": {
            "fair":           10000.0,
            "edge":           2.0,
            "quote_size":     40,
            "inventory_skew": 0.02,
        },
        "TOMATOES": {
            "edge":           2.0,
            "as_gamma":       7.0,
            "quote_size":     20,
        }
    }

    def _best_quotes(self, depth: OrderDepth):
        """Devuelve el mejor bid (compra) y ask (venta) del Nivel 1."""
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    # ── EMERALDS (Lógica Original V15) ────────────────────────────────────────

    def _emeralds_orders(self, depth: OrderDepth, fair: float, buy_cap: int, sell_cap: int, position: int) -> list:
        orders = []
        cfg    = self.CONFIG["EMERALDS"]
        INT_FAIR = int(fair)
        bv = sv = 0

        # 1. Sweep mispriced asks (Francotirador de compras)
        for ask_price in sorted(depth.sell_orders):
            if ask_price >= fair or buy_cap - bv <= 0:
                break
            qty = min(abs(depth.sell_orders[ask_price]), buy_cap - bv)
            orders.append(Order("EMERALDS", ask_price, qty))
            bv += qty

        # 2. Sweep mispriced bids (Francotirador de ventas)
        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price <= fair or sell_cap - sv <= 0:
                break
            qty = min(depth.buy_orders[bid_price], sell_cap - sv)
            orders.append(Order("EMERALDS", bid_price, -qty))
            sv += qty

        # 3. At-fair unwinding (Limpieza de stock a coste 0)
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

        # 4. Inventory tightening (Reducir margen si nos acercamos a 80)
        net = position + bv - sv
        bid_edge = ask_edge = cfg["edge"]
        if net > 60:  ask_edge = 1.0
        elif net < -60: bid_edge = 1.0

        # 5. Passive quotes with reservation skew (Market Making)
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

    # ── TOMATOES (Lógica Nacho V2 - Regression Predator) ──────────────────────

    def _tomatoes_orders(self, depth: OrderDepth, best_bid: int, best_ask: int, buy_cap: int, sell_cap: int, position: int) -> list:
        orders: List[Order] = []
        cfg = self.CONFIG["TOMATOES"]
        limit = self.POSITION_LIMITS["TOMATOES"]

        mid_price = (best_ask + best_bid) / 2.0

        # 1. Volumen del Nivel 1 (Inmediatez)
        bid_vol_1 = depth.buy_orders[best_bid]
        ask_vol_1 = abs(depth.sell_orders[best_ask])
        total_l1 = bid_vol_1 + ask_vol_1
        l1_imb = (bid_vol_1 - ask_vol_1) / total_l1 if total_l1 > 0 else 0

        # 2. Volumen Profundo (L2 y L3 - Detección de Spoofing)
        sorted_bids = sorted(depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.keys())
        
        bids_l23 = sum(depth.buy_orders[p] for p in sorted_bids[1:]) if len(sorted_bids) > 1 else 0
        asks_l23 = sum(abs(depth.sell_orders[p]) for p in sorted_asks[1:]) if len(sorted_asks) > 1 else 0
        total_l23 = bids_l23 + asks_l23
        spoof_imb = (bids_l23 - asks_l23) / total_l23 if total_l23 > 0 else 0

        # 3. Ecuación de Fair Value (Regresión Lineal extraída del CSV)
        predicted_fair = mid_price - (20.0 * spoof_imb) - (1.5 * l1_imb)

        # 4. Avellaneda-Stoikov: Control de Inventario
        reservation_price = predicted_fair - cfg["as_gamma"] * (position / limit)

        my_bid = min(int(math.floor(reservation_price - cfg["edge"])), best_bid + 1)
        my_ask = max(int(math.ceil(reservation_price + cfg["edge"])), best_ask - 1)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        # 5. Enviar órdenes pasivas al mercado
        if buy_cap > 0:
            orders.append(Order("TOMATOES", my_bid, min(cfg["quote_size"], buy_cap)))
        if sell_cap > 0:
            orders.append(Order("TOMATOES", my_ask, -min(cfg["quote_size"], sell_cap)))

        return orders

    # ── EJECUCIÓN PRINCIPAL ───────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = "" # ¡Bot sin estado! Inmune a errores de serialización

        for symbol in state.order_depths:
            depth = state.order_depths[symbol]
            position = state.position.get(symbol, 0)
            limit = self.POSITION_LIMITS.get(symbol, 80)
            
            # Calcular capacidades para no exceder los límites
            buy_cap = limit - position
            sell_cap = limit + position

            best_bid, best_ask = self._best_quotes(depth)
            if best_bid is None or best_ask is None:
                result[symbol] = []
                continue

            # Enrutador de Estrategias
            if symbol == "EMERALDS":
                fair = self.CONFIG["EMERALDS"]["fair"]
                result[symbol] = self._emeralds_orders(depth, fair, buy_cap, sell_cap, position)
                
            elif symbol == "TOMATOES":
                result[symbol] = self._tomatoes_orders(depth, best_bid, best_ask, buy_cap, sell_cap, position)

        return result, conversions, traderData