# ============================================================================
# V53 - SHIPPING STRATEGY (V50 + VFE 10/5 + Mark 22 alpha)
# ============================================================================
# Stack vs V50 baseline ($933,596 on R4):
#
#   1. PASSIVE_VFE_BID_EDGE: 4 -> 10  (filter adverse-selected passive bid fills)
#   2. PASSIVE_VFE_ASK_EDGE: 4 -> 5
#   3. Mark 22 asymmetric buy edge on {VEV_5200, VEV_5500}, red=1.5, floor=0.5
#
# Backtest results:
#   R4 (in-sample 3 days): $952,569  (+$18,973 vs V50)
#     D1: $357,955 (+$11,324)   D2: $208,682 (-$1,673)   D3: $385,932 (+$9,322)
#   R3 D0 (hold-out):     $237,548  (+$586 vs V50)
#
# What we KEPT from V50 (parked for now):
#   - V9 conditional IV regime detection
#   - V10 deep ITM module for VEV_4000 / VEV_4500
#   - V11 HGP micro-buy edge=4 (note: hold-out evidence suggests this is
#     overfit to R4 D1 specifically - costs $18K on R3, but contributes
#     $28K on R4 from D1+D2 dominance. User chose to keep for now.)
#   - M67/M49 informed-flow suppression (neutral effect, no harm)
#
# What we REJECTED from colleague's V12_Hybrid:
#   - VEV_5400 in mask:    HURTS R4 by ~$10K (HIGH_GAMMA_EDGE too tight already)
#   - VEV_6000/6500 mask:  no effect (dead markets)
#   - Leash mechanism:     tested, near-neutral, sometimes negative
#   - red=2.5 default:     red=1.5 beats it by ~$900
#
# Validation: V53 beats V12_Hybrid by $39,289 on R4 in our backtester.
# (V12: $913,280 on R4, V53: $952,569 on R4)
# ============================================================================

"""V11 FINAL: V10 + Hydrogel micro-take logic.

Strategy stack:
  V1 (baseline):           OU mean-reversion + option market making     $746,459 (SHIP)
  V9 (regime detection):   conditional IV sampling + low-vol override   +$120,780
  V10 (ITM module):        structural fair + inv-skew on VEV_4000/4500  +$37,730
  V11 (HGP micro-buy):     asymmetric take edge buy=4 sell=12 on HGP    +$28,627
                                                                        ━━━━━━━━━━
                                                                        $933,596

Per-day verification:
  Day 1: 346,631
  Day 2: 210,355
  Day 3: 376,610
"""

import math
import json
from datamodel import Order, OrderDepth, TradingState

# ============================================================================
# Products and limits
# ============================================================================
UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"
OPTION_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
}
POSITION_LIMITS = {UNDERLYING: 200, HYDROGEL: 200}
for _p in [*OPTION_STRIKES, "VEV_6000", "VEV_6500"]:
    POSITION_LIMITS[_p] = 300

# ============================================================================
# Model parameters
# ============================================================================
FINAL_TIMESTAMP = 1_000_000.0
UNDERLYING_MU = 5250.0
HYDROGEL_MU = 10000.0
UNDERLYING_KAPPA = 12.0
OPTION_KAPPA = 12.0

# Edges
UNDERLYING_EDGE = 12.0
HYDROGEL_EDGE = 12.0
OPTION_EDGE = 9.5
HIGH_GAMMA_EDGE = 1.0
PASSIVE_OPTION_EDGE = 1.0
PASSIVE_VFE_BID_EDGE = 10
PASSIVE_VFE_ASK_EDGE = 5
PASSIVE_HGP_BID_EDGE = 1
PASSIVE_HGP_ASK_EDGE = 12
PASSIVE_SIZE = 10
OPTION_POSITION_PENALTY = 1.0
DELTA_HEDGE_BETA = 0.02
HIGH_GAMMA_OPTIONS = {"VEV_5300", "VEV_5400", "VEV_5500"}

# Mark 67 / Mark 49 informed-flow suppression on VVF
M67_BUYER = "Mark 67"
M49_SELLER = "Mark 49"
SIGNAL_WINDOW = 20 * 100  # 20 updates of 100ts each

# ============================================================================
# V9: Conditional IV sampling for vol regime detection
# ============================================================================
IV_SOURCE = "VEV_5300"
IV_STRIKE = 5300
SPOT_NEAR_MEAN_THRESHOLD = 10.0   # only sample IV when |spot - 5250| < this
N_SAMPLES_TO_LOCK = 50            # need this many good samples before locking
VOL_FLOOR = 130.0                 # never lock below this
VOL_CAP = 170.0                   # never lock above this
OVERRIDE_THRESHOLD = 162.0        # if median < this, DON'T lock (low-vol regime)
OPTION_VOL_DEFAULT = 130.0        # default vol when not locked

# ============================================================================
# V10: Deep ITM module for VEV_4000 / VEV_4500
# ============================================================================
ITM_STRIKES = (4000, 4500)
ITM_STRUCTURAL_FV = {k: UNDERLYING_MU - k for k in ITM_STRIKES}  # {4000:1250, 4500:750}
ITM_TAKE_EDGE = 1
ITM_INV_SCALE = 19
ITM_PASSIVE_CAP = 150
ITM_PASSIVE_CLIP = 50
ITM_DELTA_WEIGHT = 0.5

# ============================================================================
# V11: Hydrogel micro-take (asymmetric, exploits upward drift)
# ============================================================================
MICRO_HGP_BUY_EDGE = 4      # tight take buy edge (replaces HYDROGEL_EDGE=12 for buys)
MICRO_HGP_SELL_EDGE = None  # None = keep default HYDROGEL_EDGE for sells

# ============================================================================
# V53: Mark 22 vulnerable-strike mask (asymmetric buy edge)
# ============================================================================
# Mark 22 is a heavy OTM seller pressing market prices below theoretical fair.
# We buy aggressively at the strikes where this pressure shows up most.
# Validated mask: {VEV_5200, VEV_5500}. VEV_5400 was tested and HURT R4 by $10K
# when included (HIGH_GAMMA_EDGE=1.0 means reducing further triggers
# adversely-selected fills). VEV_6000/6500 are dead markets, no effect.
# Reduction tuned to 1.5 (their V12 used 2.5, we measured 1.5 wins by ~$900).
# Leash mechanism (V12's 0.25x delta weight on vulnerable strikes) was tested
# and does not add value in our backtester, so omitted.
MARK22_VULNERABLE_STRIKES = {"VEV_5200", "VEV_5500"}
OPTION_BUY_EDGE_REDUCTION = 1.5
OPTION_BUY_EDGE_FLOOR = 0.5

# ============================================================================
# Math helpers
# ============================================================================
def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def time_left(ts):
    return max(0.0, (FINAL_TIMESTAMP - float(ts)) / FINAL_TIMESTAMP)

def ou_terminal_mean(spot, mean, kappa, ts):
    return mean + (spot - mean) * math.exp(-kappa * time_left(ts))


def call_fair_with_vol(spot, strike, ts, vol):
    tau = time_left(ts)
    rho = math.exp(-OPTION_KAPPA * tau)
    tm = UNDERLYING_MU + (spot - UNDERLYING_MU) * rho
    tsd = vol * math.sqrt(max(0.0, 1.0 - rho * rho))
    if tsd < 1e-9:
        return max(tm - strike, 0.0)
    d = (tm - strike) / tsd
    return (tm - strike) * normal_cdf(d) + tsd * normal_pdf(d)


def call_delta_with_vol(spot, strike, ts, vol):
    tau = time_left(ts)
    rho = math.exp(-OPTION_KAPPA * tau)
    tm = UNDERLYING_MU + (spot - UNDERLYING_MU) * rho
    tsd = vol * math.sqrt(max(0.0, 1.0 - rho * rho))
    if tsd < 1e-9:
        return rho if tm > strike else 0.0
    d = (tm - strike) / tsd
    return rho * normal_cdf(d)


def implied_vol_bisect(market_price, spot, strike, ts, lo=80.0, hi=250.0):
    intrinsic = max(spot - strike, 0.0)
    if market_price <= intrinsic + 0.5:
        return None
    if call_fair_with_vol(spot, strike, ts, hi) < market_price:
        return hi
    if call_fair_with_vol(spot, strike, ts, lo) > market_price:
        return lo
    for _ in range(35):
        mid = (lo + hi) / 2.0
        if call_fair_with_vol(spot, strike, ts, mid) < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.1:
            break
    return (lo + hi) / 2.0


# ============================================================================
# Trader class
# ============================================================================
class Trader:

    # ---- Order book helpers ----
    def best_mid(self, d):
        if d is None or not d.buy_orders or not d.sell_orders:
            return None
        return (max(d.buy_orders) + min(d.sell_orders)) / 2.0

    def sell_available(self, p, d, minp, cap):
        """Aggregate-sell against bids >= minp, posting at the BEST bid as price."""
        q, op = 0, None
        for pr in sorted(d.buy_orders, reverse=True):
            if pr < minp or q >= cap:
                break
            t = min(d.buy_orders[pr], cap - q)
            if t > 0:
                q += t
                op = pr
        if q <= 0 or op is None:
            return None
        return Order(p, int(op), -int(q))

    def buy_available(self, p, d, maxp, cap):
        """Aggregate-buy against asks <= maxp, posting at the BEST ask as price."""
        q, op = 0, None
        for pr in sorted(d.sell_orders):
            if pr > maxp or q >= cap:
                break
            t = min(-d.sell_orders[pr], cap - q)
            if t > 0:
                q += t
                op = pr
        if q <= 0 or op is None:
            return None
        return Order(p, int(op), int(q))

    # ---- Passive quoting ----
    def add_passive_quotes_split(self, p, d, fair, pos, lim, bid_edge, ask_edge,
                                   orders, suppress_sell=False):
        """Post passive quotes 1 inside best bid/ask, with separate bid/ask edges."""
        if not d.buy_orders or not d.sell_orders:
            return pos
        bb = max(d.buy_orders)
        ba = min(d.sell_orders)
        bp = bb + 1
        if bp < ba and bp <= fair - bid_edge:
            q = min(PASSIVE_SIZE, max(0, lim - pos))
            if q > 0:
                orders.append(Order(p, int(bp), int(q)))
                pos += q
        if not suppress_sell:
            sp = ba - 1
            if sp > bb and sp >= fair + ask_edge:
                q = min(PASSIVE_SIZE, max(0, lim + pos))
                if q > 0:
                    orders.append(Order(p, int(sp), -int(q)))
                    pos -= q
        return pos

    def add_passive_quotes(self, p, d, fair, pos, lim, edge, orders):
        return self.add_passive_quotes_split(p, d, fair, pos, lim, edge, edge, orders)

    def option_edge(self, p):
        return HIGH_GAMMA_EDGE if p in HIGH_GAMMA_OPTIONS else OPTION_EDGE

    # ---- Delta exposure (V10: half-weight ITM) ----
    def delta_exposure(self, state, spot, vol):
        e = float(state.position.get(UNDERLYING, 0))
        for p, k in OPTION_STRIKES.items():
            delta = call_delta_with_vol(spot, k, state.timestamp, vol)
            weight = ITM_DELTA_WEIGHT if k in ITM_STRIKES else 1.0
            e += weight * state.position.get(p, 0) * delta
        return e

    # ---- Mean-reversion underlying / hydrogel trader (V11: asymmetric edges) ----
    def trade_ou_split(self, p, d, pos, mean, edge, bid_pe, ask_pe, ts, fs=0.0,
                        suppress_sell=False, take_buy_edge=None, take_sell_edge=None):
        """OU mean-reversion trade with optional asymmetric take edges.
        take_buy_edge/take_sell_edge override the default `edge` for active takes."""
        spot = self.best_mid(d)
        if spot is None:
            return []
        lim = POSITION_LIMITS[p]
        fair = ou_terminal_mean(spot, mean, UNDERLYING_KAPPA, ts) + fs
        sell_edge = take_sell_edge if take_sell_edge is not None else edge
        buy_edge  = take_buy_edge  if take_buy_edge  is not None else edge
        orders = []
        if not suppress_sell:
            sc = lim + pos
            if sc > 0:
                o = self.sell_available(p, d, fair + sell_edge, sc)
                if o is not None:
                    orders.append(o)
                    pos += o.quantity
        bc = lim - pos
        if bc > 0:
            o = self.buy_available(p, d, fair - buy_edge, bc)
            if o is not None:
                orders.append(o)
                pos += o.quantity
        self.add_passive_quotes_split(p, d, fair, pos, lim, bid_pe, ask_pe, orders,
                                        suppress_sell=suppress_sell)
        return orders

    # ---- Generic option trader (used for VEV_5000-5500) ----
    def trade_option(self, p, d, pos, spot, ts, vol):
        lim = POSITION_LIMITS[p]
        fair = call_fair_with_vol(spot, OPTION_STRIKES[p], ts, vol)
        fair -= OPTION_POSITION_PENALTY * pos / lim
        sell_edge = self.option_edge(p)
        # V53: asymmetric buy edge on Mark 22 vulnerable strikes only
        if p in MARK22_VULNERABLE_STRIKES:
            buy_edge = max(OPTION_BUY_EDGE_FLOOR,
                           sell_edge - OPTION_BUY_EDGE_REDUCTION)
        else:
            buy_edge = sell_edge
        orders = []
        if spot >= UNDERLYING_MU:
            sc = lim + pos
            if sc > 0:
                o = self.sell_available(p, d, fair + sell_edge, sc)
                if o is not None:
                    orders.append(o)
                    pos += o.quantity
        bc = lim - pos
        if bc > 0:
            o = self.buy_available(p, d, fair - buy_edge, bc)
            if o is not None:
                orders.append(o)
                pos += o.quantity
        self.add_passive_quotes(p, d, fair, pos, lim, PASSIVE_OPTION_EDGE, orders)
        return orders

    # ---- V10: Deep ITM module for VEV_4000 / VEV_4500 ----
    def trade_itm_option(self, sym, depth, pos, strike,
                          take_edge=ITM_TAKE_EDGE, scale=ITM_INV_SCALE,
                          passive_cap=ITM_PASSIVE_CAP, passive_clip=ITM_PASSIVE_CLIP):
        """Specialized deep-ITM logic with structural fair, inventory skew,
        aggressive passive market making."""
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return []
        fair = ITM_STRUCTURAL_FV[strike]
        limit = POSITION_LIMITS[sym]
        inv_offset = int(abs(pos) / limit * scale)

        # Inventory-skewed thresholds
        if pos < 0:
            buy_thresh = fair - take_edge - inv_offset
            sell_thresh = fair + take_edge
        elif pos > 0:
            buy_thresh = fair - take_edge
            sell_thresh = fair + take_edge + inv_offset
        else:
            buy_thresh = fair - take_edge
            sell_thresh = fair + take_edge

        orders = []

        # Active sell: sweep bids >= sell_thresh, post at best bid
        sc = limit + pos
        if sc > 0 and max(depth.buy_orders) >= sell_thresh:
            q = 0
            for pr in sorted(depth.buy_orders, reverse=True):
                if pr < sell_thresh or q >= sc:
                    break
                q += min(depth.buy_orders[pr], sc - q)
            if q > 0:
                orders.append(Order(sym, int(max(depth.buy_orders)), -int(q)))
                pos -= q

        # Active buy: sweep asks <= buy_thresh, post at best ask
        bc = limit - pos
        if bc > 0 and min(depth.sell_orders) <= buy_thresh:
            q = 0
            for pr in sorted(depth.sell_orders):
                if pr > buy_thresh or q >= bc:
                    break
                q += min(-depth.sell_orders[pr], bc - q)
            if q > 0:
                orders.append(Order(sym, int(min(depth.sell_orders)), int(q)))
                pos += q

        # Passive market making (only when not too inventory-saturated)
        if abs(pos) < passive_cap:
            bb = max(depth.buy_orders)
            ba = min(depth.sell_orders)
            spread = ba - bb
            edge = max(1, spread // 2)
            buy_quote = int(round(fair - edge))
            sell_quote = int(round(fair + edge))
            if buy_quote <= bb:
                buy_quote = bb + 1
            if sell_quote >= ba:
                sell_quote = ba - 1
            if buy_quote < ba:
                q = min(passive_clip, max(0, limit - pos))
                if q > 0:
                    orders.append(Order(sym, buy_quote, int(q)))
            if sell_quote > bb:
                q = min(passive_clip, max(0, limit + pos))
                if q > 0:
                    orders.append(Order(sym, sell_quote, -int(q)))

        return orders

    # ---- M67/M49 informed flow detector ----
    def update_signal_state(self, state, ss):
        last_ts = ss.get("last_signal_ts", -10**9)
        for t in state.market_trades.get(UNDERLYING, []):
            if t.buyer == M67_BUYER or t.seller == M49_SELLER:
                if t.timestamp > last_ts:
                    last_ts = t.timestamp
        ss["last_signal_ts"] = last_ts
        return ss

    # ---- V9: Conditional IV regime detection ----
    def update_iv_state(self, state, ss, spot):
        """Sample IV only when spot is near mean. After N samples, lock vol
        IF median > OVERRIDE_THRESHOLD (high-vol regime). Otherwise stay at default."""
        if ss.get("vol_locked", False):
            return ss, ss["vol"]
        if spot is None or abs(spot - UNDERLYING_MU) >= SPOT_NEAR_MEAN_THRESHOLD:
            return ss, OPTION_VOL_DEFAULT
        depth = state.order_depths.get(IV_SOURCE)
        opt_mid = self.best_mid(depth) if depth else None
        if opt_mid is None:
            return ss, OPTION_VOL_DEFAULT
        iv = implied_vol_bisect(opt_mid, spot, IV_STRIKE, state.timestamp)
        if iv is None:
            return ss, OPTION_VOL_DEFAULT
        samples = ss.get("iv_samples", [])
        samples.append(iv)
        ss["iv_samples"] = samples
        if len(samples) >= N_SAMPLES_TO_LOCK:
            sorted_s = sorted(samples)
            median = sorted_s[len(sorted_s) // 2]
            # OVERRIDE: low-vol regime detected, don't lock - stay at default
            if median < OVERRIDE_THRESHOLD:
                ss["vol_locked"] = False
                ss["iv_samples"] = []
                return ss, OPTION_VOL_DEFAULT
            # High-vol regime: lock at median, capped/floored
            locked = max(VOL_FLOOR, min(VOL_CAP, median))
            ss["vol"] = locked
            ss["vol_locked"] = True
            ss["iv_samples"] = []
            return ss, locked
        return ss, OPTION_VOL_DEFAULT

    # ---- Main loop ----
    def run(self, state):
        try:
            ss = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ss = {}
        ss = self.update_signal_state(state, ss)

        result = {}
        vvf_depth = state.order_depths.get(UNDERLYING)
        vvf_spot = self.best_mid(vvf_depth)

        # V9: get current vol (locked or default)
        ss, cur_vol = self.update_iv_state(state, ss, vvf_spot)

        last_ts = ss.get("last_signal_ts", -10**9)
        signal_active = (state.timestamp - last_ts) <= SIGNAL_WINDOW and last_ts >= 0

        # --- VVF (with M67 sells-suppression) ---
        if UNDERLYING in state.order_depths:
            fs = 0.0
            if vvf_spot is not None:
                fs = -DELTA_HEDGE_BETA * self.delta_exposure(state, vvf_spot, cur_vol)
            position = state.position.get(UNDERLYING, 0)
            orders = self.trade_ou_split(
                UNDERLYING, state.order_depths[UNDERLYING], position,
                UNDERLYING_MU, UNDERLYING_EDGE,
                PASSIVE_VFE_BID_EDGE, PASSIVE_VFE_ASK_EDGE,
                state.timestamp, fs,
                suppress_sell=signal_active,
            )
            if orders:
                result[UNDERLYING] = orders

        # --- Hydrogel (V11: micro-buy at edge=4, sell at default 12) ---
        if HYDROGEL in state.order_depths:
            position = state.position.get(HYDROGEL, 0)
            orders = self.trade_ou_split(
                HYDROGEL, state.order_depths[HYDROGEL], position,
                HYDROGEL_MU, HYDROGEL_EDGE,
                PASSIVE_HGP_BID_EDGE, PASSIVE_HGP_ASK_EDGE,
                state.timestamp,
                take_buy_edge=MICRO_HGP_BUY_EDGE,    # 4
                take_sell_edge=MICRO_HGP_SELL_EDGE,  # None = default 12
            )
            if orders:
                result[HYDROGEL] = orders

        # --- All options (generic logic, will be overridden for ITM strikes below) ---
        if vvf_spot is not None:
            for product in OPTION_STRIKES:
                depth = state.order_depths.get(product)
                if depth is None:
                    continue
                position = state.position.get(product, 0)
                orders = self.trade_option(product, depth, position, vvf_spot,
                                            state.timestamp, cur_vol)
                if orders:
                    result[product] = orders

        # --- V10: Override ITM strikes (VEV_4000 / VEV_4500) with specialized logic ---
        for strike in ITM_STRIKES:
            sym = f"VEV_{strike}"
            result.pop(sym, None)
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            position = state.position.get(sym, 0)
            orders = self.trade_itm_option(sym, depth, position, strike)
            if orders:
                result[sym] = orders

        return result, 0, json.dumps(ss)