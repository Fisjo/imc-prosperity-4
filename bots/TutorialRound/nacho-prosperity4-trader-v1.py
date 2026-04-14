from datamodel import Order, OrderDepth, TradingState
import json
import math
from typing import Dict, List, Any

class Trader:
    
    PRODUCT = "TOMATOES"
    POSITION_LIMIT = 80
    
    # Parámetros ajustables para que puedas experimentar
    MIN_VOL_FILTER = 15     # Ignorar órdenes con menos de este volumen (filtrar a los "peces pequeños")
    PRESSURE_TICKS = 50     # Cuántos ticks miramos hacia atrás para calcular la presión
    VOLUME_BAR = 1.0        # Volumen promedio ejecutado que consideramos "peligroso" (para huir)

    def _load_td(self, state: TradingState) -> dict:
        """Carga la memoria del bot del tick anterior."""
        if not state.traderData:
            # Estado inicial por defecto en el Tick 0
            return {
                "last_best_bid": None,
                "last_best_ask": None,
                "last_best_bid_volume": 0,
                "last_best_ask_volume": 0,
                "pressure_history": [],
                "mid_history": [],
                "upper_edge": 3,
                "lower_edge": 3,
                "volume_history": [],
                "last_position": 0,
                "last_signal": None,
                "optimized": False
            }
        try:
            return json.loads(state.traderData)
        except Exception:
            return {}

    def get_running_pressure(self, order_depth: OrderDepth, td: dict) -> float:
        """Calcula hacia dónde está empujando el mercado filtrando el ruido."""
        best_bid = None
        best_ask = None
        best_bid_volume = 0
        best_ask_volume = 0

        # Encontrar el mejor bid "real" (ignorando volumen pequeño)
        for bid, volume in order_depth.buy_orders.items():
            if volume >= self.MIN_VOL_FILTER:
                if best_bid is None or bid > best_bid:
                    best_bid = bid
                    best_bid_volume = volume

        # Encontrar el mejor ask "real" (ignorando volumen pequeño)
        for ask, volume in order_depth.sell_orders.items():
            if abs(volume) >= self.MIN_VOL_FILTER:
                if best_ask is None or ask < best_ask:
                    best_ask = ask
                    best_ask_volume = abs(volume)

        if best_bid is None or best_ask is None:
            return 0.0

        buy_pressure = 0
        sell_pressure = 0

        # Comparar con el tick anterior para ver la "aceleración"
        if best_bid is not None and td["last_best_bid"] is not None:
            if best_bid > td["last_best_bid"]:
                buy_pressure = best_bid_volume
            elif best_bid == td["last_best_bid"]:
                buy_pressure = best_bid_volume - td["last_best_bid_volume"]
            else:
                buy_pressure = -td["last_best_bid_volume"]

        if best_ask is not None and td["last_best_ask"] is not None:
            if best_ask < td["last_best_ask"]:
                sell_pressure = best_ask_volume
            elif best_ask == td["last_best_ask"]:
                sell_pressure = best_ask_volume - td["last_best_ask_volume"]
            else:
                sell_pressure = -td["last_best_ask_volume"]

        # Diferencial de presión
        pressure_difference = buy_pressure - sell_pressure
        td["pressure_history"].append(pressure_difference)
        
        if len(td["pressure_history"]) > self.PRESSURE_TICKS:
            td["pressure_history"].pop(0)

        # Solo devolvemos la presión si ya tenemos suficientes datos (50 ticks)
        running_pressure = sum(td["pressure_history"]) if len(td["pressure_history"]) == self.PRESSURE_TICKS else 0

        # Guardar estado actual para el siguiente tick
        td["last_best_bid"] = best_bid
        td["last_best_ask"] = best_ask
        td["last_best_bid_volume"] = best_bid_volume
        td["last_best_ask_volume"] = best_ask_volume

        return running_pressure

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        td = self._load_td(state)

        if self.PRODUCT not in state.order_depths:
            return result, conversions, json.dumps(td)

        order_depth = state.order_depths[self.PRODUCT]
        position = state.position.get(self.PRODUCT, 0)
        orders: List[Order] = []

        # 1. Calcular el Fair Value filtrado
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
        if best_ask is None or best_bid is None:
             return result, conversions, json.dumps(td)

        # Filtramos como los ganadores: buscamos bloques grandes
        filtered_ask = [p for p in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[p]) >= self.MIN_VOL_FILTER]
        filtered_bid = [p for p in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[p]) >= self.MIN_VOL_FILTER]
        
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
        
        mmmid_price = (mm_ask + mm_bid) / 2
        td["mid_history"].append(mmmid_price)
        
        # Guardamos solo los últimos 2 para suavizar un poco
        td["mid_history"] = td["mid_history"][-2:]
        if len(td["mid_history"]) >= 2:
            fair_value = round((td["mid_history"][-1] + td["mid_history"][-2]) / 2)
        else:
            fair_value = mmmid_price

        # 2. Leer la Presión del Mercado
        running_pressure = self.get_running_pressure(order_depth, td)
        
        # Generar Señal Direccional
        if (0 < running_pressure < 30) or (running_pressure < -30):
            signal = "b" # Comprar
        elif (-30 < running_pressure < 0) or (running_pressure > 30):
            signal = "s" # Vender
        else:
            signal = "h" # Mantener (Hold)

        # 3. Calcular volumen ejecutado y evaluar peligro (Dynamic Edge)
        filled_volume = abs(position - td["last_position"])
        td["last_position"] = position

        if state.timestamp > 0:
            td["volume_history"].append(filled_volume)
            if len(td["volume_history"]) > 5:
                td["volume_history"].pop(0)
            
            # Resetear la optimización si la señal cambió
            if td["last_signal"] != signal:
                td["optimized"] = False
                td["volume_history"] = []
            
            # Ajustar nuestro margen pasivo basado en cuánto nos están comprando/vendiendo
            if len(td["volume_history"]) >= 3 and not td["optimized"]:
                volume_avg = sum(td["volume_history"]) / len(td["volume_history"])
                
                if signal == "b":
                    curr_edge = td["lower_edge"]
                    if volume_avg > self.VOLUME_BAR:
                        # Nos están vendiendo mucho, el precio cae. ¡Ensanchar el margen!
                        td["lower_edge"] = min(curr_edge + 1, 5)
                        td["volume_history"] = []
                    elif volume_avg < self.VOLUME_BAR * 0.7:
                        # Nadie nos vende, acercarnos al precio.
                        td["lower_edge"] = max(curr_edge - 1, -2) # Dejo que sea negativo para ser agresivo
                        td["volume_history"] = []
                elif signal == "s":
                    curr_edge = td["upper_edge"]
                    if volume_avg > self.VOLUME_BAR:
                        td["upper_edge"] = min(curr_edge + 1, 5)
                        td["volume_history"] = []
                    elif volume_avg < self.VOLUME_BAR * 0.7:
                        td["upper_edge"] = max(curr_edge - 1, -2)
                        td["volume_history"] = []

        td["last_signal"] = signal
        upper_edge = td["upper_edge"]
        lower_edge = td["lower_edge"]

        # 4. Enviar las órdenes al mercado
        if signal == "b":
            target_volume = self.POSITION_LIMIT - position
            if target_volume > 0:
                orders.append(Order(self.PRODUCT, fair_value - lower_edge, int(round(target_volume))))
        
        elif signal == "s":
            target_volume = -self.POSITION_LIMIT - position
            if target_volume < 0:
                orders.append(Order(self.PRODUCT, fair_value + upper_edge, int(round(target_volume))))

        result[self.PRODUCT] = orders

        return result, conversions, json.dumps(td)