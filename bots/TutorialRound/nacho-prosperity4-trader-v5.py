from datamodel import Order, OrderDepth, TradingState, Trade
import json
import math
from typing import Dict, List, Any

class Trader:
    """
    Nacho V5 - Tape Reader & Micro-Pennying
    - EMERALDS: V15 Logic (Market Making anclado de alta eficiencia)
    - TOMATOES: Order Flow Toxicity + Whale Fingerprinting + Pennying
    """

    POSITION_LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    # Configuraciones base
    CONFIG = {
        "EMERALDS": {
            "fair": 10000.0,
            "edge": 2.0,
            "quote_size": 40,
            "inventory_skew": 0.02,
        },
        "TOMATOES": {
            "as_gamma": 5.0,
            "whale_footprint": 21,  # EL NÚMERO MÁGICO (Ajustar tras analizar CSV)
            "tfi_memory_ticks": 5   # Cuántos ticks recordamos el flujo de mercado
        }
    }

    def _load_state(self, state: TradingState) -> dict:
        if not state.traderData:
            return {
                "tomatoes_tfi_history": [],
                "tomatoes_last_pos": 0
            }
        try:
            return json.loads(state.traderData)
        except Exception:
            return {"tomatoes_tfi_history": [], "tomatoes_last_pos": 0}

    def _best_quotes(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    # ── EMERALDS (Intacto - Máxima Eficiencia) ────────────────────────────────

    def _emeralds_orders(self, depth: OrderDepth, fair: float, buy_cap: int, sell_cap: int, position: int) -> list:
        orders = []
        cfg = self.CONFIG["EMERALDS"]
        INT_FAIR = int(fair)
        bv = sv = 0

        for ask_price in sorted(depth.sell_orders):
            if ask_price >= fair or buy_cap - bv <= 0: break
            qty = min(abs(depth.sell_orders[ask_price]), buy_cap - bv)
            orders.append(Order("EMERALDS", ask_price, qty))
            bv += qty

        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price <= fair or sell_cap - sv <= 0: break
            qty = min(depth.buy_orders[bid_price], sell_cap - sv)
            orders.append(Order("EMERALDS", bid_price, -qty))
            sv += qty

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
        reservation = fair - cfg["inventory_skew"] * (self.POSITION_LIMITS["EMERALDS"] - (buy_cap - bv) - (sell_cap - sv))
        bid_q = min(int(math.floor(reservation - bid_edge)), best_bid + 1)
        ask_q = max(int(math.ceil(reservation + ask_edge)),  best_ask - 1)
        
        if bid_q >= ask_q: bid_q = ask_q - 1

        bqty = min(cfg["quote_size"], buy_cap - bv)
        if bqty > 0: orders.append(Order("EMERALDS", bid_q, bqty))
        sqty = min(cfg["quote_size"], sell_cap - sv)
        if sqty > 0: orders.append(Order("EMERALDS", ask_q, -sqty))

        return orders

    # ── TOMATOES (Tape Reader & Pennying) ─────────────────────────────────────

    def _tomatoes_orders(self, state: TradingState, best_bid: int, best_ask: int, buy_cap: int, sell_cap: int, position: int, td: dict) -> list:
        orders: List[Order] = []
        cfg = self.CONFIG["TOMATOES"]
        depth = state.order_depths["TOMATOES"]
        limit = self.POSITION_LIMITS["TOMATOES"]
        mid_price = (best_ask + best_bid) / 2.0

        # === 1. TAPE READING: Analizar transacciones reales del mercado ===
        market_trades = state.market_trades.get("TOMATOES", [])
        tick_buy_vol = 0
        tick_sell_vol = 0
        whale_detected = False

        for trade in market_trades:
            # Si el precio del trade fue mayor o igual al ask, alguien compró agresivamente
            if trade.price >= best_ask:
                tick_buy_vol += trade.quantity
            # Si el precio fue menor o igual al bid, alguien vendió agresivamente
            elif trade.price <= best_bid:
                tick_sell_vol += trade.quantity

            # FINGERPRINTING: Detectar si este trade coincide con el tamaño exacto de la Ballena
            if trade.quantity == cfg["whale_footprint"]:
                whale_detected = True

        # Calcular el desbalance de transacciones de este tick
        tick_tfi = tick_buy_vol - tick_sell_vol
        td["tomatoes_tfi_history"].append(tick_tfi)
        if len(td["tomatoes_tfi_history"]) > cfg["tfi_memory_ticks"]:
            td["tomatoes_tfi_history"].pop(0)

        # Imbalance acumulado (Momentum real)
        running_tfi = sum(td["tomatoes_tfi_history"])

        # === 2. CALCULAR PRECIO JUSTO (Basado en Momentum Real, no en Spoofing) ===
        # El flujo de órdenes reales empuja el precio. 10 unidades netas reales mueven el precio ~1 tick.
        predicted_fair = mid_price + (running_tfi * 0.1)

        if whale_detected:
            # Si vemos la huella de la ballena comprando, disparamos el precio justo hacia arriba
            if tick_tfi > 0: predicted_fair += 5.0 
            else: predicted_fair -= 5.0

        # === 3. TAKER LOGIC (Francotirador Basado en Cinta) ===
        take_qty_buy = 0
        take_qty_sell = 0

        if predicted_fair >= best_ask + 1.5 and buy_cap > 0:
            take_qty_buy = min(15, buy_cap)
            orders.append(Order("TOMATOES", best_ask, take_qty_buy))
            buy_cap -= take_qty_buy
            position += take_qty_buy
            
        elif predicted_fair <= best_bid - 1.5 and sell_cap > 0:
            take_qty_sell = min(15, sell_cap)
            orders.append(Order("TOMATOES", best_bid, -take_qty_sell))
            sell_cap -= take_qty_sell
            position -= take_qty_sell

        # === 4. MICRO-PENNYING (Búsqueda de Muros) ===
        # En lugar de precios estáticos, buscamos dónde están los muros grandes en el L1/L2
        biggest_bid_price = best_bid
        max_bid_vol = 0
        for price, vol in depth.buy_orders.items():
            if vol > max_bid_vol:
                max_bid_vol = vol
                biggest_bid_price = price

        biggest_ask_price = best_ask
        max_ask_vol = 0
        for price, vol in depth.sell_orders.items():
            if abs(vol) > max_ask_vol:
                max_ask_vol = abs(vol)
                biggest_ask_price = price

        # === 5. MAKER LOGIC ===
        # Nos ponemos 1 tick por delante del muro más grande (Pennying), ajustado por inventario
        reservation_shift = cfg["as_gamma"] * (position / limit)
        
        # Nuestro bid pasivo será justo por encima del mayor muro de compradores, menos el ajuste de inventario
        my_bid = min(int(math.floor(biggest_bid_price + 1 - reservation_shift)), best_bid + 1)
        # Nuestro ask pasivo será justo por debajo del mayor muro de vendedores, más el ajuste de inventario
        my_ask = max(int(math.ceil(biggest_ask_price - 1 - reservation_shift)), best_ask - 1)

        # Cortafuegos para no cruzar nuestros propios precios y perder dinero
        if my_bid >= my_ask:
            my_bid = my_ask - 1

        # Enviar órdenes pasivas (quote size es 20)
        if buy_cap > 0:
            orders.append(Order("TOMATOES", my_bid, min(20, buy_cap)))
        if sell_cap > 0:
            orders.append(Order("TOMATOES", my_ask, -min(20, sell_cap)))

        td["tomatoes_last_pos"] = position
        return orders

    # ── EJECUCIÓN PRINCIPAL ───────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        td = self._load_state(state)

        for symbol in state.order_depths:
            depth = state.order_depths[symbol]
            position = state.position.get(symbol, 0)
            limit = self.POSITION_LIMITS.get(symbol, 80)
            
            buy_cap = limit - position
            sell_cap = limit + position

            best_bid, best_ask = self._best_quotes(depth)
            if best_bid is None or best_ask is None:
                result[symbol] = []
                continue

            if symbol == "EMERALDS":
                fair = self.CONFIG["EMERALDS"]["fair"]
                result[symbol] = self._emeralds_orders(depth, fair, buy_cap, sell_cap, position)
                
            elif symbol == "TOMATOES":
                # Le pasamos el 'state' completo a tomatoes porque necesitamos state.market_trades
                result[symbol] = self._tomatoes_orders(state, best_bid, best_ask, buy_cap, sell_cap, position, td)

        return result, conversions, json.dumps(td)