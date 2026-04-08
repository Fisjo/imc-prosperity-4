from datamodel import Order, OrderDepth, TradingState
import json
import math
from typing import Dict, List

class Trader:
    
    PRODUCT = "TOMATOES"
    POSITION_LIMIT = 80
    
    # Parámetros Científicos (Extraídos de tu CSV de la Ronda 0)
    EDGE = 2.0        # Ganancia que buscamos por operación (estando dentro del spread de 13)
    AS_GAMMA = 7.0    # Agresividad para descargar inventario (Igual que tus amigos)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        if self.PRODUCT not in state.order_depths:
            return result, conversions, traderData

        order_depth = state.order_depths[self.PRODUCT]
        position = state.position.get(self.PRODUCT, 0)
        orders: List[Order] = []

        # 1. Obtener los precios actuales
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
        if best_ask is None or best_bid is None:
             return result, conversions, traderData

        mid_price = (best_ask + best_bid) / 2.0

        # =========================================================================
        # 2. MODELO DE REGRESIÓN: EL PREDICTOR DE SPOOFING
        # =========================================================================
        
        # A. Volumen del Nivel 1 (Inmediatez)
        bid_vol_1 = order_depth.buy_orders[best_bid]
        ask_vol_1 = abs(order_depth.sell_orders[best_ask])
        total_l1 = bid_vol_1 + ask_vol_1
        l1_imb = (bid_vol_1 - ask_vol_1) / total_l1 if total_l1 > 0 else 0

        # B. Volumen Profundo (L2 y L3 - Detección de Spoofing)
        sorted_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(order_depth.sell_orders.keys())
        
        bids_l23 = sum(order_depth.buy_orders[p] for p in sorted_bids[1:]) if len(sorted_bids) > 1 else 0
        asks_l23 = sum(abs(order_depth.sell_orders[p]) for p in sorted_asks[1:]) if len(sorted_asks) > 1 else 0
        total_l23 = bids_l23 + asks_l23
        spoof_imb = (bids_l23 - asks_l23) / total_l23 if total_l23 > 0 else 0

        # C. Ecuación de Fair Value (Sustituye a la lenta EMA de tus amigos)
        # Los coeficientes -20.0 y -1.5 salen directamente del CSV que subiste.
        predicted_fair = mid_price - (20.0 * spoof_imb) - (1.5 * l1_imb)

        # =========================================================================
        # 3. AVELLANEDA-STOIKOV: CONTROL DE INVENTARIO
        # =========================================================================
        
        # Penalizamos o beneficiamos nuestro precio dependiendo de qué tan llenos estamos
        reservation_price = predicted_fair - self.AS_GAMMA * (position / self.POSITION_LIMIT)

        # 4. Generar precios de cotización (Market Making pasivo)
        # Usamos math.floor/ceil para ser enteros, y restamos/sumamos el EDGE (margen)
        my_bid = min(int(math.floor(reservation_price - self.EDGE)), best_bid + 1)
        my_ask = max(int(math.ceil(reservation_price + self.EDGE)), best_ask - 1)

        # Evitar cruces accidentales en nuestro propio código
        if my_bid >= my_ask:
            my_bid = my_ask - 1

        # 5. Calcular capacidades de volumen (¡No pasarse de 80!)
        buy_cap = self.POSITION_LIMIT - position
        sell_cap = self.POSITION_LIMIT + position

        # Enviar órdenes (Cotizamos 20 unidades, que es un buen bocado)
        if buy_cap > 0:
            orders.append(Order(self.PRODUCT, my_bid, min(20, buy_cap)))
        
        if sell_cap > 0:
            orders.append(Order(self.PRODUCT, my_ask, -min(20, sell_cap)))

        result[self.PRODUCT] = orders

        return result, conversions, traderData