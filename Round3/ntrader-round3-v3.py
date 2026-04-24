"""
Round 3 Trading Strategy - GOAT v3.0 (Clean & Adaptive)
Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_* vouchers (options)

IMPLEMENTED IMPROVEMENTS (No internal PnL tracking for max stability):
1. Dynamic IV estimation (blends 0.24 prior with rolling realized vol).
2. Max delta-hedge size per tick (capped to 20 units).
3. Faster EMA during high volatility (alpha bumps to 0.5 on >10 tick jumps).
4. Volatility-based Circuit Breaker (widens spreads and halves sizes on vol spikes).
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle
import math

# ─────────────────────────────────────────────
# BLACK-SCHOLES HELPERS (no scipy needed)
# ─────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    approx = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return approx if x >= 0 else 1.0 - approx

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0: return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0: return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return norm_cdf(d1)

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────

BASE_IV = 0.24
HYDROGEL_FAIR = 10000

VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
}
ACTIVE_VOUCHERS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]

LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{v: 300 for v in VOUCHER_STRIKES},
}

HYDROGEL_SPREAD = 4       
VEV_SPREAD = 3            
VOUCHER_SPREAD = 1        
ORDER_FRACTION = 0.4

# New Defense Parameters
MAX_HEDGE_PER_TICK = 20
VOL_WINDOW = 20
CIRCUIT_BREAKER_VOL_THRESHOLD = 0.35 # Triggers defense mode if realized vol exceeds 35%

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def mid_price(order_depth: OrderDepth) -> float | None:
    if not order_depth.buy_orders or not order_depth.sell_orders: return None
    return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2.0

def best_bid(od: OrderDepth) -> float | None: return max(od.buy_orders.keys()) if od.buy_orders else None
def best_ask(od: OrderDepth) -> float | None: return min(od.sell_orders.keys()) if od.sell_orders else None

def available_volume(od: OrderDepth, side: str, price: float) -> int:
    if side == "buy": return abs(sum(v for p, v in od.sell_orders.items() if p <= price))
    else: return sum(v for p, v in od.buy_orders.items() if p >= price)

# ─────────────────────────────────────────────
# MAIN TRADER CLASS
# ─────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        # ── 1. Load persistent state ──────────────────────────────────
        trader_data = {}
        if state.traderData and state.traderData != "":
            try: trader_data = jsonpickle.decode(state.traderData)
            except Exception: pass

        vev_ema = trader_data.get("vev_ema", None)
        vev_history = trader_data.get("vev_history", [])
        step = trader_data.get("step", 0)

        round_progress = min(state.timestamp / 1_000_000, 1.0)
        TTE_days = 5.0 - round_progress
        TTE = max(TTE_days / 365.0, 1e-6)

        orders: Dict[str, List[Order]] = {}
        positions = state.position

        # ── 2. Dynamics: Fast EMA, IV & Circuit Breaker ───────────────
        vev_spot = None
        current_iv = BASE_IV
        circuit_breaker = False

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vev_od = state.order_depths["VELVETFRUIT_EXTRACT"]
            vev_mid = mid_price(vev_od)
            
            if vev_mid is not None:
                # Rolling History for Volatility
                vev_history.append(vev_mid)
                if len(vev_history) > VOL_WINDOW:
                    vev_history.pop(0)

                if len(vev_history) >= 2:
                    returns = [math.log(vev_history[i]/vev_history[i-1]) for i in range(1, len(vev_history))]
                    mean_ret = sum(returns) / len(returns)
                    variance = sum((r - mean_ret)**2 for r in returns) / len(returns)
                    realized_vol = math.sqrt(variance) * 100 # Scaling factor proxy
                    
                    # Blend 70% Prior / 30% Realized
                    current_iv = 0.7 * BASE_IV + 0.3 * min(max(realized_vol, 0.15), 0.45)
                    
                    # Trigger Circuit Breaker on extreme volatility spikes
                    if current_iv > CIRCUIT_BREAKER_VOL_THRESHOLD:
                        circuit_breaker = True

                # Fast EMA on price jumps > 10 ticks
                if vev_ema is None:
                    vev_ema = vev_mid
                else:
                    alpha = 0.5 if abs(vev_mid - vev_ema) > 10 else 0.1
                    vev_ema = alpha * vev_mid + (1 - alpha) * vev_ema
                
                vev_spot = vev_ema

        # Apply defense multipliers
        spread_mult = 2 if circuit_breaker else 1
        size_mult = 0.5 if circuit_breaker else 1.0

        # ── 3. HYDROGEL_PACK market making ────────────────────────────
        hyd_orders = self._market_make(
            product="HYDROGEL_PACK", fair_value=HYDROGEL_FAIR,
            half_spread=HYDROGEL_SPREAD * spread_mult, state=state, size_mult=size_mult
        )
        if hyd_orders: orders["HYDROGEL_PACK"] = hyd_orders

        # ── 4. VELVETFRUIT_EXTRACT market making ──────────────────────
        if vev_spot is not None:
            vev_orders = self._market_make(
                product="VELVETFRUIT_EXTRACT", fair_value=vev_spot,
                half_spread=VEV_SPREAD * spread_mult, state=state, size_mult=size_mult
            )
            if vev_orders: orders["VELVETFRUIT_EXTRACT"] = vev_orders

        # ── 5. Options market making + Capped Delta Hedging ───────────
        if vev_spot is not None:
            net_delta = 0.0

            for voucher in ACTIVE_VOUCHERS:
                K = VOUCHER_STRIKES[voucher]
                if voucher not in state.order_depths: continue

                fair = bs_call_price(vev_spot, K, TTE, current_iv)
                delta = bs_delta(vev_spot, K, TTE, current_iv)

                v_orders = self._options_market_make(
                    product=voucher, fair_value=fair,
                    half_spread=max(VOUCHER_SPREAD * spread_mult, 1), state=state, size_mult=size_mult
                )
                if v_orders: orders[voucher] = v_orders

                pos = positions.get(voucher, 0)
                net_delta += pos * delta

            # ── Delta Hedge ──
            hedge_target = -round(net_delta)
            vev_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
            vev_limit = LIMITS["VELVETFRUIT_EXTRACT"]

            hedge_delta = hedge_target - vev_pos
            hedge_delta = max(-vev_limit - vev_pos, min(vev_limit - vev_pos, hedge_delta))

            # CAP THE HEDGE (Max 20 units per tick to prevent over-buying the dip)
            hedge_delta = max(-MAX_HEDGE_PER_TICK, min(MAX_HEDGE_PER_TICK, hedge_delta))

            if abs(hedge_delta) >= 1 and vev_spot is not None and "VELVETFRUIT_EXTRACT" in state.order_depths:
                vev_od = state.order_depths["VELVETFRUIT_EXTRACT"]
                hedge_orders = orders.get("VELVETFRUIT_EXTRACT", [])

                if hedge_delta > 0:
                    ask = best_ask(vev_od)
                    if ask is not None: hedge_orders.append(Order("VELVETFRUIT_EXTRACT", ask, int(hedge_delta)))
                else:
                    bid = best_bid(vev_od)
                    if bid is not None: hedge_orders.append(Order("VELVETFRUIT_EXTRACT", bid, int(hedge_delta)))

                orders["VELVETFRUIT_EXTRACT"] = hedge_orders

        # ── 6. Save state ─────────────────────────────────────────────
        trader_data["vev_ema"] = vev_ema
        trader_data["vev_history"] = vev_history
        trader_data["step"] = step + 1
        
        return orders, 0, jsonpickle.encode(trader_data)

    # ──────────────────────────────────────────
    # MARKET MAKING HELPERS (Updated with size_mult)
    # ──────────────────────────────────────────

    def _market_make(self, product: str, fair_value: float, half_spread: int, state: TradingState, size_mult: float) -> List[Order]:
        if product not in state.order_depths: return []

        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = LIMITS[product]
        orders: List[Order] = []
        bb = best_bid(od)
        ba = best_ask(od)

        if ba is not None and ba < fair_value - 1:
            buy_qty = min(limit - pos, available_volume(od, "buy", fair_value - 1))
            if buy_qty > 0:
                orders.append(Order(product, ba, buy_qty))
                pos += buy_qty

        if bb is not None and bb > fair_value + 1:
            sell_qty = min(pos + limit, available_volume(od, "sell", fair_value + 1))
            if sell_qty > 0:
                orders.append(Order(product, bb, -sell_qty))
                pos -= sell_qty

        skew = int(pos / limit * half_spread)
        our_bid = int(fair_value) - half_spread - skew
        our_ask = int(fair_value) + half_spread - skew

        if bb is not None: our_bid = min(our_bid, bb)
        if ba is not None: our_ask = max(our_ask, ba)

        buy_cap = limit - pos
        sell_cap = limit + pos

        # Apply defense size modifier
        bid_size = max(1, int(buy_cap * ORDER_FRACTION * size_mult))
        ask_size = max(1, int(sell_cap * ORDER_FRACTION * size_mult))

        if buy_cap > 0 and our_bid > 0: orders.append(Order(product, our_bid, bid_size))
        if sell_cap > 0: orders.append(Order(product, our_ask, -ask_size))

        return orders

    def _options_market_make(self, product: str, fair_value: float, half_spread: int, state: TradingState, size_mult: float) -> List[Order]:
        if product not in state.order_depths: return []

        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = LIMITS[product]
        orders: List[Order] = []
        bb = best_bid(od)
        ba = best_ask(od)

        if ba is not None and ba < fair_value - half_spread:
            qty = min(limit - pos, abs(sum(v for p, v in od.sell_orders.items() if p <= fair_value - half_spread)))
            if qty > 0:
                orders.append(Order(product, ba, qty))
                pos += qty

        if bb is not None and bb > fair_value + half_spread:
            qty = min(pos + limit, sum(v for p, v in od.buy_orders.items() if p >= fair_value + half_spread))
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                pos -= qty

        skew = int(pos / limit * half_spread * 0.5)
        our_bid = math.floor(fair_value) - half_spread - skew
        our_ask = math.ceil(fair_value) + half_spread - skew

        if our_bid <= 0: our_bid = 1
        if our_ask <= our_bid: our_ask = our_bid + 1

        buy_cap = limit - pos
        sell_cap = limit + pos

        # Apply defense size modifier
        bid_size = max(1, int(buy_cap * ORDER_FRACTION * size_mult))
        ask_size = max(1, int(sell_cap * ORDER_FRACTION * size_mult))

        if buy_cap > 0: orders.append(Order(product, our_bid, bid_size))
        if sell_cap > 0: orders.append(Order(product, our_ask, -ask_size))

        return orders