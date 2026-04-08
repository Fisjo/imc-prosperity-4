from datamodel import Order, OrderDepth, TradingState
import json
import math
from typing import Dict, List, Any

class Trader:
    """
    Nacho V4 - The Apex Predator
    - EMERALDS: V15 Logic (Market Making anclado de alta eficiencia)
    - TOMATOES: Machine Learning HFT
        1. Taker Logic (Agresividad)
        2. Dynamic Edge (Optimización de spread)
        3. Adverse Selection Defense (Pánico controlado)
        4. Dynamic Coefficients (Auto-ajuste de regresión)
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
            "as_gamma": 7.0,
            "quote_size": 20,
            "learning_rate": 0.1 # Velocidad a la que el bot aprende de sus errores
        }
    }

    def _load_state(self, state: TradingState) -> dict:
        if not state.traderData:
            return {
                "tomatoes": {
                    "spoof_coef": 20.0,   # Empezamos con el valor de tu CSV
                    "l1_coef": 1.5,       # Empezamos con el valor de tu CSV
                    "last_mid": None,
                    "last_spoof_imb": 0.0,
                    "last_l1_imb": 0.0,
                    "last_position": 0,
                    "panic_ticks": 0      # Contador de defensa
                }
            }
        try:
            return json.loads(state.traderData)
        except Exception:
            return self._load_state(TradingState("", 0, {}, {}, {}, {}, {}, "")) # Fallback recursivo de seguridad

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
        reservation = fair - cfg["inventory_skew"] * (self.POSITION_LIMITS["EMERALDS"] - (buy_cap - bv) - (sell_cap - sv))
        bid_q = min(int(math.floor(reservation - bid_edge)), best_bid + 1)
        ask_q = max(int(math.ceil(reservation + ask_edge)),  best_ask - 1)
        
        if bid_q >= ask_q: bid_q = ask_q - 1

        bqty = min(cfg["quote_size"], buy_cap - bv)
        if bqty > 0: orders.append(Order("EMERALDS", bid_q, bqty))
        sqty = min(cfg["quote_size"], sell_cap - sv)
        if sqty > 0: orders.append(Order("EMERALDS", ask_q, -sqty))

        return orders

    # ── TOMATOES (Apex Predator con Auto-Ajuste) ──────────────────────────────

    def _tomatoes_orders(self, depth: OrderDepth, best_bid: int, best_ask: int, buy_cap: int, sell_cap: int, position: int, td: dict) -> list:
        orders: List[Order] = []
        cfg = self.CONFIG["TOMATOES"]
        limit = self.POSITION_LIMITS["TOMATOES"]
        td_tom = td["tomatoes"]

        mid_price = (best_ask + best_bid) / 2.0
        spread = best_ask - best_bid

        # --- 1. DYNAMIC COEFFICIENTS (Machine Learning Básico) ---
        # Si tenemos datos del tick anterior, vemos si nos equivocamos y ajustamos los coeficientes.
        if td_tom["last_mid"] is not None:
            actual_diff = mid_price - td_tom["last_mid"]
            # Lo que predijimos en el tick anterior
            predicted_diff = -(td_tom["spoof_coef"] * td_tom["last_spoof_imb"]) - (td_tom["l1_coef"] * td_tom["last_l1_imb"])
            
            # Error de predicción
            error = actual_diff - predicted_diff
            
            # Ajustamos el multiplicador del spoofing. Si el error es positivo (subió más de lo esperado), 
            # y el spoof_imb era negativo, incrementamos el coeficiente para predecir mejor la próxima vez.
            td_tom["spoof_coef"] += cfg["learning_rate"] * error * td_tom["last_spoof_imb"]
            
            # Limitamos los coeficientes para que el bot no se vuelva loco
            td_tom["spoof_coef"] = max(10.0, min(35.0, td_tom["spoof_coef"]))

        # --- 2. CÁLCULO DE SENSORES ACTUALES ---
        bid_vol_1 = depth.buy_orders[best_bid]
        ask_vol_1 = abs(depth.sell_orders[best_ask])
        total_l1 = bid_vol_1 + ask_vol_1
        l1_imb = (bid_vol_1 - ask_vol_1) / total_l1 if total_l1 > 0 else 0

        sorted_bids = sorted(depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.keys())
        bids_l23 = sum(depth.buy_orders[p] for p in sorted_bids[1:]) if len(sorted_bids) > 1 else 0
        asks_l23 = sum(abs(depth.sell_orders[p]) for p in sorted_asks[1:]) if len(sorted_asks) > 1 else 0
        total_l23 = bids_l23 + asks_l23
        spoof_imb = (bids_l23 - asks_l23) / total_l23 if total_l23 > 0 else 0

        # Predecimos el Fair Value usando los coeficientes dinámicos
        predicted_fair = mid_price - (td_tom["spoof_coef"] * spoof_imb) - (td_tom["l1_coef"] * l1_imb)

        # --- 3. ADVERSE SELECTION DEFENSE (Pánico) ---
        filled_qty = abs(position - td_tom["last_position"])
        if filled_qty > 15:
            td_tom["panic_ticks"] = 3 # Entramos en pánico durante 3 ticks
        
        # --- 4. DYNAMIC EDGE (Optimización de Margen) ---
        # Pedimos el 35% del spread como margen base. Si el spread es 14, pedimos ~5 ticks.
        dynamic_edge = max(1.5, spread * 0.35) 
        
        if td_tom["panic_ticks"] > 0:
            dynamic_edge *= 2.0 # Si estamos en pánico, duplicamos el margen para alejarnos del fuego
            td_tom["panic_ticks"] -= 1

        # --- 5. TAKER LOGIC (Francotirador Agresivo) ---
        # Si el precio justo predicho es muy superior al precio de venta actual, ¡compramos ya!
        if predicted_fair >= best_ask + 1.5 and buy_cap > 0:
            take_qty = min(15, buy_cap) # Tomamos un bocado agresivo pero no todo
            orders.append(Order("TOMATOES", best_ask, take_qty))
            buy_cap -= take_qty # Actualizamos capacidades para el Market Making posterior
            position += take_qty
            
        elif predicted_fair <= best_bid - 1.5 and sell_cap > 0:
            take_qty = min(15, sell_cap)
            orders.append(Order("TOMATOES", best_bid, -take_qty))
            sell_cap -= take_qty
            position -= take_qty

        # --- 6. MAKER LOGIC (Avellaneda-Stoikov con Dynamic Edge) ---
        reservation_price = predicted_fair - cfg["as_gamma"] * (position / limit)

        my_bid = min(int(math.floor(reservation_price - dynamic_edge)), best_bid + 1)
        my_ask = max(int(math.ceil(reservation_price + dynamic_edge)), best_ask - 1)

        if my_bid >= my_ask: my_bid = my_ask - 1

        if buy_cap > 0:
            orders.append(Order("TOMATOES", my_bid, min(cfg["quote_size"], buy_cap)))
        if sell_cap > 0:
            orders.append(Order("TOMATOES", my_ask, -min(cfg["quote_size"], sell_cap)))

        # Guardar estado para el siguiente tick
        td_tom["last_mid"] = mid_price
        td_tom["last_spoof_imb"] = spoof_imb
        td_tom["last_l1_imb"] = l1_imb
        td_tom["last_position"] = position

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
                result[symbol] = self._tomatoes_orders(depth, best_bid, best_ask, buy_cap, sell_cap, position, td)

        return result, conversions, json.dumps(td)