from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List

class Trader:
    
    # Límite de posición genérico para no pasarnos de la raya
    POSITION_LIMIT = 20 

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = "" # No usaremos memoria (estado) por ahora

        # Iteramos sobre todos los productos que el simulador nos envíe
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # 1. Obtener los mejores precios actuales (Nivel 1 del libro)
            if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
                # Si el libro de órdenes está vacío por un lado, saltamos este turno por seguridad
                continue
            
            best_ask = min(order_depth.sell_orders.keys()) # Lo más barato a lo que alguien vende
            best_bid = max(order_depth.buy_orders.keys())  # Lo más caro a lo que alguien compra

            # 2. Calcular el Mid-Price (Precio Medio)
            mid_price = (best_ask + best_bid) / 2

            # 3. Definir nuestros precios de cotización (1 tick de margen)
            my_bid_price = int(mid_price) - 1 # Queremos comprar 1 unidad más barato que la media
            my_ask_price = int(mid_price) + 1 # Queremos vender 1 unidad más caro que la media

            # 4. Comprobar nuestro inventario actual
            # Si no tenemos nada de este producto, state.position devuelve 0
            current_position = state.position.get(product, 0)

            # 5. Calcular cuánto podemos comprar y vender sin pasarnos del límite
            # En IMC Prosperity, las compras son positivas y las ventas son negativas
            buy_volume = self.POSITION_LIMIT - current_position
            sell_volume = -self.POSITION_LIMIT - current_position 

            # 6. Enviar nuestras órdenes al mercado
            if buy_volume > 0:
                orders.append(Order(product, my_bid_price, buy_volume))
            
            if sell_volume < 0:
                orders.append(Order(product, my_ask_price, sell_volume))

            # Guardamos las órdenes de este producto en el resultado final
            result[product] = orders

        return result, conversions, traderData