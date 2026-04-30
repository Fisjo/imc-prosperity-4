import json
from typing import Dict, List, Optional
from datamodel import Order, OrderDepth, TradingState

class Trader:
    def __init__(self):
        # Parámetros del modelo dinámico (Ajustables)
        self.K = 0.003
        self.entry_z = 1.75
        self.exit_z = 0.5
        self.max_pos = 10
        
        # Definición de los Spreads (Activo A - Activo B)
        self.spread_defs = {
            'CHOC_VAN': ('CHOCOLATE', 'VANILLA'),
            'PIST_STRAW': ('PISTACHIO', 'STRAWBERRY'),
            'RASP_STRAW': ('RASPBERRY', 'STRAWBERRY')
        }
        
        # Volatilidad Congelada de la Innovación (Debes calibrar esto con días previos)
        self.frozen_stds = {
            'CHOC_VAN': 1.25, 
            'PIST_STRAW': 1.10,
            'RASP_STRAW': 1.30
        }

    def get_mid_price(self, product: str, state: TradingState) -> Optional[float]:
        """Calcula el precio medio basado en el mejor bid y ask."""
        depth = state.order_depths.get(product)
        if not depth:
            return None
        if depth.buy_orders and depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            best_bid = max(depth.buy_orders.keys())
            return (best_ask + best_bid) / 2.0
        return None

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        """
        Punto de entrada principal llamado por el motor de IMC en cada tick.
        """
        result: dict[str, list[Order]] = {}
        conversions = 0 
        
        # 1. RECUPERAR ESTADO INTERNO (traderData) de forma segura
        if state.traderData:
            internal_state = json.loads(state.traderData)
        else:
            # Inicializamos en el tick 0
            internal_state = {
                "fair_values": {spread: None for spread in self.spread_defs},
                "spread_positions": {spread: 0 for spread in self.spread_defs}
            }

        # 2. OBTENER PRECIOS MEDIOS
        mid_prices: Dict[str, float] = {}
        for spread_id, (asset_a, asset_b) in self.spread_defs.items():
            mid_a = self.get_mid_price(asset_a, state)
            mid_b = self.get_mid_price(asset_b, state)
            if mid_a is not None and mid_b is not None:
                mid_prices[asset_a] = mid_a
                mid_prices[asset_b] = mid_b

        # 3. ACTUALIZAR MODELO CAUSAL Y GENERAR SEÑALES (Z-SCORES)
        raw_spread_targets = {}
        
        for spread_id, (asset_a, asset_b) in self.spread_defs.items():
            if asset_a not in mid_prices or asset_b not in mid_prices:
                raw_spread_targets[spread_id] = internal_state["spread_positions"][spread_id]
                continue
                
            current_spread = mid_prices[asset_a] - mid_prices[asset_b]
            m_t_minus_1 = internal_state["fair_values"][spread_id]
            
            # Inicialización en el primer tick válido
            if m_t_minus_1 is None:
                internal_state["fair_values"][spread_id] = current_spread
                raw_spread_targets[spread_id] = 0
                continue
                
            # Calcular Innovación y Z-Score
            innovation = current_spread - m_t_minus_1
            z_score = innovation / self.frozen_stds[spread_id]
            
            # Actualizar Fair Value para el próximo tick
            internal_state["fair_values"][spread_id] = m_t_minus_1 + self.K * innovation
            
            # Evaluar reglas de Entrada / Salida
            current_spread_pos = internal_state["spread_positions"][spread_id]
            
            if abs(z_score) > self.entry_z:
                # spread caro (vender A, comprar B) -> pos negativa
                direction = -1 if z_score > 0 else 1
                raw_spread_targets[spread_id] = self.max_pos * direction
            elif abs(z_score) < self.exit_z:
                raw_spread_targets[spread_id] = 0 # Reversión: Aplanar
            else:
                raw_spread_targets[spread_id] = current_spread_pos # Mantener

        # 4. NETTING Y RECORTE PROPORCIONAL (CLIPPING)
        target_inventory = {asset: 0 for asset in ['CHOCOLATE', 'VANILLA', 'PISTACHIO', 'RASPBERRY', 'STRAWBERRY']}
        
        for spread_id, (asset_a, asset_b) in self.spread_defs.items():
            if spread_id in raw_spread_targets:
                pos = raw_spread_targets[spread_id]
                target_inventory[asset_a] += pos
                target_inventory[asset_b] -= pos

        # Comprobar límite estricto en STRAWBERRY (activo solapado)
        straw_target = target_inventory['STRAWBERRY']
        if abs(straw_target) > self.max_pos:
            scale_factor = self.max_pos / abs(straw_target)
            
            # Escalar SOLO los spreads que comparten STRAWBERRY
            for spread_id in ['PIST_STRAW', 'RASP_STRAW']:
                if spread_id in raw_spread_targets:
                    # Truncar hacia cero usando int() para no romper límite
                    raw_spread_targets[spread_id] = int(raw_spread_targets[spread_id] * scale_factor)
                    
            # Recalcular inventario con los spreads escalados
            target_inventory = {asset: 0 for asset in target_inventory.keys()}
            for spread_id, (asset_a, asset_b) in self.spread_defs.items():
                if spread_id in raw_spread_targets:
                    pos = raw_spread_targets[spread_id]
                    target_inventory[asset_a] += pos
                    target_inventory[asset_b] -= pos

        # Guardar posiciones de spread efectivas en el estado interno
        for spread_id in self.spread_defs:
            if spread_id in raw_spread_targets:
                internal_state["spread_positions"][spread_id] = raw_spread_targets[spread_id]

        # 5. TRADUCCIÓN A ÓRDENES DEL EXCHANGE
        for asset, target in target_inventory.items():
            # Obtener inventario actual del asset
            current_pos = state.position.get(asset, 0)
            delta = target - current_pos
            
            if delta != 0 and asset in state.order_depths:
                order_depth = state.order_depths[asset]
                orders = []
                
                # IMPORTANTÍSIMO para el backtester: Forzar int() en precio y delta
                delta = int(delta)
                
                if delta > 0: # Comprar
                    if order_depth.sell_orders:
                        best_ask = int(min(order_depth.sell_orders.keys()))
                        orders.append(Order(asset, best_ask, delta))
                        
                elif delta < 0: # Vender
                    if order_depth.buy_orders:
                        best_bid = int(max(order_depth.buy_orders.keys()))
                        orders.append(Order(asset, best_bid, delta))
                
                if orders:
                    result[asset] = orders

        # 6. SERIALIZAR ESTADO PARA EL PRÓXIMO TICK
        # (Usamos separadores compactos para optimizar como en la v0)
        traderData = json.dumps(internal_state, separators=(",", ":"))
        
        return result, conversions, traderData