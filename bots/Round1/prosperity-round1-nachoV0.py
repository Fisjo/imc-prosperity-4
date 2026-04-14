from datamodel import Order, OrderDepth, TradingState
import json
import math

class Trader:
    """
    Prosperity 4 - Round 1 (V21 Architecture Base)
    
    Activos:
    - ASH_COATED_OSMIUM: Mean Reversion Market Making. Anclado en 10,000. 
      Usa spread ancho (Edge) y Avellaneda-Stoikov para control de inventario.
    - INTARIAN_PEPPER_ROOT: Trend Following Sniper. 
      Cruza la EMA con el L1 Imbalance (64% correlación).
      Utiliza "Auction Awareness" para no empujar el Clearing Price en contra.
    """

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80
    }

    # Configuraciones parametrizadas estilo V21
    CONFIG = {
        "ASH_COATED_OSMIUM": {
            "anchor_fair": 10000.0,
            "edge": 6.0,          # Margen para capturar el spread de ~16 ticks
            "as_gamma": 0.2,      # Sensibilidad al inventario
            "quote_size": 20
        },
        "INTARIAN_PEPPER_ROOT": {
            "anchor_fair": None,  # Activo de tendencia, no tiene ancla fija
            "ema_span": 20,       # Periodos para definir la tendencia lenta
            "imbalance_threshold": 0.6, # Correlación alta
            "take_size": 20       # "One carefully considered order"
        }
    }

    # ── Helpers (Estructura V21) ──────────────────────────────────────────────

    def _load_td(self, state: TradingState) -> dict:
        if not state.traderData:
            return {}
        try:
            return json.loads(state.traderData)
        except Exception:
            return {}

    def _best_quotes(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        return best_bid, best_ask

    def _ema(self, symbol: str, mid: float, td: dict, updates: dict) -> float:
        cfg = self.CONFIG[symbol]
        if cfg.get("anchor_fair") is not None:
            return float(cfg["anchor_fair"])
        
        key = f"{symbol}_ema"
        prev = td.get(key, mid)
        alpha = 2.0 / (cfg["ema_span"] + 1.0)
        ema = alpha * mid + (1.0 - alpha) * prev
        updates[key] = ema
        return float(ema)

    def _simulate_clearing_price(self, depth: OrderDepth, test_price: int, test_vol: int):
        """
        Simulador del motor de Subasta (Call Auction).
        Calcula el precio donde se cruza el mayor volumen.
        """
        all_bids = depth.buy_orders.copy()
        all_asks = depth.sell_orders.copy()

        if test_vol > 0:
            all_bids[test_price] = all_bids.get(test_price, 0) + test_vol
        elif test_vol < 0:
            all_asks[test_price] = all_asks.get(test_price, 0) + abs(test_vol)

        prices = sorted(set(list(all_bids.keys()) + list(all_asks.keys())))
        max_vol = 0
        best_p = 0

        for p in prices:
            demand = sum(v for k, v in all_bids.items() if k >= p)
            supply = sum(v for k, v in all_asks.items() if k <= p)
            traded = min(demand, supply)
            if traded >= max_vol:
                max_vol = traded
                best_p = p
                
        return best_p, max_vol

    # ── ASH_COATED_OSMIUM (El nuevo Emeralds) ─────────────────────────────────

    def _osmium_orders(self, depth: OrderDepth, pos: int) -> list:
        orders = []
        cfg = self.CONFIG["ASH_COATED_OSMIUM"]
        limit = self.POSITION_LIMITS["ASH_COATED_OSMIUM"]
        fair = cfg["anchor_fair"]

        best_bid, best_ask = self._best_quotes(depth)
        if not best_bid or not best_ask: 
            return []

        # Avellaneda-Stoikov simplificado para Osmium
        reservation = fair - cfg["as_gamma"] * pos

        # Calculamos nuestros precios pidiendo nuestro "edge" (margen)
        # Hacemos Pennying limitándonos a best_bid + 1 / best_ask - 1
        my_bid = min(int(math.floor(reservation - cfg["edge"])), best_bid + 1)
        my_ask = max(int(math.ceil(reservation + cfg["edge"])), best_ask - 1)

        # Cortafuegos
        if my_bid >= my_ask:
            my_bid = my_ask - 1

        buy_cap = limit - pos
        sell_cap = limit + pos

        if buy_cap > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_bid, min(cfg["quote_size"], buy_cap)))
        if sell_cap > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_ask, -min(cfg["quote_size"], sell_cap)))

        return orders

    # ── INTARIAN_PEPPER_ROOT (Trend & Imbalance Sniper) ───────────────────────

    def _pepper_orders(self, depth: OrderDepth, pos: int, ema: float) -> list:
        orders = []
        cfg = self.CONFIG["INTARIAN_PEPPER_ROOT"]
        limit = self.POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]

        best_bid, best_ask = self._best_quotes(depth)
        if not best_bid or not best_ask: 
            return []

        mid = (best_bid + best_ask) / 2.0

        # Cálculo de Imbalance del Nivel 1
        v_bid = depth.buy_orders[best_bid]
        v_ask = abs(depth.sell_orders[best_ask])
        total_v = v_bid + v_ask
        imbalance = (v_bid - v_ask) / total_v if total_v > 0 else 0

        buy_cap = limit - pos
        sell_cap = limit + pos
        size = min(cfg["take_size"], limit)

        # Lógica de Sniper + Trend Following
        # Si el imbalance es muy alto y el precio está subiendo (mid > ema)
        if imbalance > cfg["imbalance_threshold"] and mid > ema:
            qty = min(size, buy_cap)
            if qty > 0:
                # REGLA DE LA SUBASTA: "A well-placed offer does not shout"
                # Compramos EXACTAMENTE al best_ask para no alterar el clearing price.
                # Si pusiéramos best_ask + 10, empujaríamos la subasta hacia arriba.
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, qty))
                
        elif imbalance < -cfg["imbalance_threshold"] and mid < ema:
            qty = min(size, sell_cap)
            if qty > 0:
                # Vendemos EXACTAMENTE al best_bid
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -qty))

        return orders

    # ── Motor Principal (Run) ─────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        td = self._load_td(state)
        updates = {}

        # Contador global (Warmup) heredado de la V21
        tick_key = "tick_count"
        tick = td.get(tick_key, 0) + 1
        updates[tick_key] = tick

        for symbol in ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]:
            if symbol not in state.order_depths:
                continue

            depth = state.order_depths[symbol]
            position = state.position.get(symbol, 0)

            if symbol == "ASH_COATED_OSMIUM":
                result[symbol] = self._osmium_orders(depth, position)
                
            elif symbol == "INTARIAN_PEPPER_ROOT":
                best_bid, best_ask = self._best_quotes(depth)
                if best_bid and best_ask:
                    mid = (best_bid + best_ask) / 2.0
                    # Actualizamos la memoria (EMA) en cada tick
                    ema = self._ema(symbol, mid, td, updates)
                    
                    # Esperamos 50 ticks de "warmup" (para que la EMA sea fiable) antes de disparar
                    if tick > 50:
                        result[symbol] = self._pepper_orders(depth, position, ema)
                    else:
                        result[symbol] = []

        # Fusionar el diccionario antiguo con las nuevas actualizaciones y guardar
        merged_td = {**td, **updates}
        return result, conversions, json.dumps(merged_td)