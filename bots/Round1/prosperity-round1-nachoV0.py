from datamodel import Order, OrderDepth, TradingState
import json
import math

class Trader:
    """
    Nacho R1 - The Specialist (Official Round 1 Bot)
    - ASH_COATED_OSMIUM: Mean Reversion MM (Anchored 10k, Spread Capture).
    - INTARIAN_PEPPER_ROOT: Momentum Sniper (Imbalance-driven, Trend following).
    """

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80
    }

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        traderData = "" # No necesitamos memoria para esta estrategia

        # Procesamos solo los activos de la Ronda 1
        for symbol in ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]:
            if symbol not in state.order_depths:
                continue
            
            depth = state.order_depths[symbol]
            pos = state.position.get(symbol, 0)
            limit = self.POSITION_LIMITS[symbol]
            
            if symbol == "ASH_COATED_OSMIUM":
                result[symbol] = self._osmium_strategy(depth, pos, limit)
            
            elif symbol == "INTARIAN_PEPPER_ROOT":
                result[symbol] = self._pepper_strategy(depth, pos, limit)

        return result, conversions, traderData

    def _osmium_strategy(self, depth: OrderDepth, pos: int, limit: int) -> list[Order]:
        """
        Lógica para ASH_COATED_OSMIUM:
        Market Making de reversión a la media anclado en 10,000.
        Aprovecha el spread de ~16 ticks.
        """
        fair = 10000.0
        orders = []
        
        # 1. Calculamos el Precio de Reserva (Ajuste por inventario)
        # Si estamos muy largos, bajamos nuestros precios para vender.
        # Coeficiente 0.2: si tenemos 80 unidades, bajamos 16 ticks nuestro centro.
        reservation_price = fair - (pos * 0.2)
        
        # 2. Identificamos los mejores precios del mercado para hacer Pennying
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else fair - 10
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else fair + 10
        
        # 3. Definimos nuestros precios (Edge de 4 ticks para asegurar captura de spread)
        my_bid = min(int(math.floor(reservation_price - 4)), best_bid + 1)
        my_ask = max(int(math.ceil(reservation_price + 4)), best_ask - 1)
        
        if my_bid >= my_ask:
            my_bid = my_ask - 1

        # 4. Enviamos órdenes (Consolidadas según el consejo del Auction)
        buy_qty = limit - pos
        sell_qty = limit + pos # Posición negativa

        if buy_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_bid, buy_qty))
        if sell_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_ask, -sell_qty))
            
        return orders

    def _pepper_strategy(self, depth: OrderDepth, pos: int, limit: int) -> list[Order]:
        """
        Lógica para INTARIAN_PEPPER_ROOT:
        Sniper de tendencia basado en el Order Book Imbalance (OBI).
        No hace market making para evitar quedar atrapado en tendencias largas.
        """
        orders = []
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        
        # 1. Cálculo del Imbalance (V_bid / V_total)
        v_bid = depth.buy_orders[best_bid]
        v_ask = abs(depth.sell_orders[best_ask])
        total_v = v_bid + v_ask
        
        imbalance = (v_bid - v_ask) / total_v if total_v > 0 else 0
        
        # 2. Señal de Sniper (Umbral de confianza del 60%)
        # "One carefully considered order": disparamos bloques de tamaño moderado
        trade_size = 20 

        if imbalance > 0.6:
            # Momentum alcista detectado -> Atacamos el spread (Buy)
            qty = min(trade_size, limit - pos)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, qty))
                
        elif imbalance < -0.6:
            # Momentum bajista detectado -> Atacamos el spread (Sell)
            qty = min(trade_size, limit + pos)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -qty))

        # 3. Gestión de salida (Mean-Reversion de seguridad)
        # Si la tendencia se agota (imbalance vuelve a neutral), cerramos un poco de posición
        if abs(imbalance) < 0.1:
            if pos > 20:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -10))
            elif pos < -20:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, 10))

        return orders