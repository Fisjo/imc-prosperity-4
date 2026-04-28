"""V64b_A6 - FINAL SHIP STRATEGY for Round 4
================================================================================
Backtest results:
  R3 D0 (HOLD-OUT):  $   272,934
  R4 D1:             $   358,296
  R4 D2:             $   212,936
  R4 D3:             $   399,082
  R4 Total:          $   970,314
  ALL Total:         $ 1,243,248

Strategy stack (each layer additive on top of previous):

  V1 baseline ($746K R4):
    - OU mean-reversion model for VFE (UNDERLYING_MU=5250) and HGP (HYDROGEL_MU=10000)
    - Black-Scholes-style call fairs for options (using OU terminal mean)
    - Position-aware passive market making with edge-based take logic

  V9 layer (+$120K R4): Conditional IV regime detection
    - Sample VEV_5300 implied vol when spot near mean
    - Lock at high vol (170) for high-IV days, stay at default (130) for low-IV days
    - Captures bimodal IV distribution (152 vs 178 across observed days)

  V10 layer (+$37K R4): Deep ITM specialized module
    - VEV_4000 / VEV_4500 use structural FV = MU - K
    - Inventory-aware buy/sell thresholds with passive quoting
    - Captures consistent edge on deep ITM strikes

  V11 layer (+$28K R4): HGP micro-buy aggressive edge
    - Tighter buy edge (4) when conditions met
    - Symmetric default edge (12) for safety

  V53 (+$18K R4): Mark 22 vulnerable strikes alpha
    - Asymmetric buy edge on {VEV_5200, VEV_5500} (reduction = 1.5)
    - Captures Mark 22 informed-flow signal in options

  V58 (+$18K R3 vs V53): M14 gating
    - V11 aggressive buy mode gated on Mark 14 recent activity
    - Auto-fallback to safe mode in anonymized data (R3) where M14 doesn't exist

  V61_sent (+$24K vs V53): HGP opening regime classifier
    - Three regimes by spot at day open: low (<9985), mid, high (>10010)
    - Different (mean, buy_edge) per regime: (10000, 1), (10000, 9), (10006, 9)

  V63 (+$5K): HGP_ID alpha (M14 buy minus M38 buy flow)
    - Continuous flow signal at HGP_ID_SCALE = 0.15

  V64b improvement over V63 ($1,230,382 -> $1,243,225, +$13K):
    - Online OU EWMA kappa estimation for both VFE and HGP
    - Replaces hardcoded kappa=12 with empirically-derived dynamic kappa
    - Fitted defaults from R4 D1-D3: VFE kappa=19.37, HGP kappa=19.40
    - Adapts within-day if process changes

  A6 improvement on V64b (interpretable IV lock):
    - Replaces magic threshold "median < 162" with "p5(samples) < 160"
    - Same PnL, cleaner mechanism: "lock only when 5th percentile of IV
      samples is clearly above 160 (bottom 10% in high regime)"
    - Threshold 160 sits in 10-vol-point insensitivity plateau [160, 170]

================================================================================
"""

import json
import math
from datamodel import Order, OrderDepth, TradingState

# ============================================================================
# V64 - V63 + Online OU estimation for HGP kappa
# ============================================================================
# Backtester results:
#   Round 4: 981,094
#   Round 3: 827,707
#
# Delta vs original v62:
#   1. Hydrogel opening regime retuned:
#      - low_open buy edge: 2 -> 1
#      - high_open mean: 10005 -> 10006
#   2. Hydrogel ID alpha:
#      - Mark 14 flow is mildly bullish
#      - Mark 38 flow is mildly bearish
#   3. VEV_5500 ID alpha:
#      - Mark 22 flow is mildly bullish
#      - Mark 01 flow is mildly bearish
# ============================================================================

# ============================================================================
# Products and limits
# ============================================================================
UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"
OPTION_STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
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

# ============================================================================
# V9: Conditional IV sampling for vol regime detection
# ============================================================================
IV_SOURCE = "VEV_5300"
IV_STRIKE = 5300
SPOT_NEAR_MEAN_THRESHOLD = 10.0
N_SAMPLES_TO_LOCK = 50
VOL_FLOOR = 130.0
VOL_CAP = 170.0
OVERRIDE_THRESHOLD = 162.0
OPTION_VOL_DEFAULT = 130.0

# ============================================================================
# V10: Deep ITM module for VEV_4000 / VEV_4500
# ============================================================================
ITM_STRIKES = (4000, 4500)
ITM_STRUCTURAL_FV = {k: UNDERLYING_MU - k for k in ITM_STRIKES}
ITM_TAKE_EDGE = 1
ITM_INV_SCALE = 19
ITM_PASSIVE_CAP = 150
ITM_PASSIVE_CLIP = 50
ITM_DELTA_WEIGHT = 0.5

# ============================================================================
# Hydrogel opening regime
# ============================================================================
MICRO_HGP_BUY_EDGE = 4
MICRO_HGP_SELL_EDGE = None
HGP_LOW_OPEN_TH = 9985.0
HGP_HIGH_OPEN_TH = 10010.0
HGP_LOW_OPEN_MEAN = 10000.0
HGP_LOW_OPEN_BUY_EDGE = 1
HGP_MID_OPEN_MEAN = 10000.0
HGP_MID_OPEN_BUY_EDGE = 9
HGP_HIGH_OPEN_MEAN = 10006.0
HGP_HIGH_OPEN_BUY_EDGE = 9

# ============================================================================
# Voucher alpha
# ============================================================================
MARK22_VULNERABLE_STRIKES = {"VEV_5200", "VEV_5500"}
OPTION_BUY_EDGE_REDUCTION = 1.5
OPTION_BUY_EDGE_FLOOR = 0.5
HGP_ID_SCALE = 0.15
VEV5500_ID_SCALE = 0.05

# ============================================================================
# V64: Online OU EWMA for HGP kappa estimation (from V13 idea)
# ============================================================================
VFE_DEFAULT_PHI = 0.998064
VFE_DEFAULT_KAPPA = 19.3741
VFE_DEFAULT_SIGMA = 18.2827
HGP_DEFAULT_PHI = 0.998062
HGP_DEFAULT_KAPPA = 19.4017  # empirically fit on R4 D1-D3
HGP_DEFAULT_SIGMA = 34.8349
HGP_OU_EWMA_LAMBDA = 0.999
HGP_OU_MIN_OBSERVATIONS = 50
HGP_OU_MIN_PHI = 1e-6
HGP_OU_MAX_PHI = 0.999999
HGP_OU_MIN_SXX = 1e-9
HGP_OU_MIN_DT = 1e-12



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


class _IdSignalMixin:
    def signed_flow(self, trades, buyer_id=None, seller_id=None):
        signal = 0.0
        for t in trades:
            q = abs(t.quantity)
            if buyer_id is not None and t.buyer == buyer_id:
                signal += q
            if seller_id is not None and t.seller == seller_id:
                signal -= q
        return signal

    def hgp_id_alpha(self, state):
        trades = state.market_trades.get(HYDROGEL, [])
        return self.signed_flow(
            trades, buyer_id="Mark 14", seller_id="Mark 14"
        ) - self.signed_flow(trades, buyer_id="Mark 38", seller_id="Mark 38")

    def vev5500_id_alpha(self, state):
        trades = state.market_trades.get("VEV_5500", [])
        return self.signed_flow(
            trades, buyer_id="Mark 22", seller_id="Mark 22"
        ) - self.signed_flow(trades, buyer_id="Mark 01", seller_id="Mark 01")


class Trader(_IdSignalMixin):
    def best_mid(self, d):
        if d is None or not d.buy_orders or not d.sell_orders:
            return None
        return (max(d.buy_orders) + min(d.sell_orders)) / 2.0

    def sell_available(self, p, d, minp, cap):
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

    def add_passive_quotes_split(self, p, d, fair, pos, lim, bid_edge, ask_edge, orders, suppress_sell=False):
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

    def delta_exposure(self, state, spot, vol):
        e = float(state.position.get(UNDERLYING, 0))
        for p, k in OPTION_STRIKES.items():
            delta = call_delta_with_vol(spot, k, state.timestamp, vol)
            weight = ITM_DELTA_WEIGHT if k in ITM_STRIKES else 1.0
            e += weight * state.position.get(p, 0) * delta
        return e

    def trade_ou_split(
        self,
        p,
        d,
        pos,
        mean,
        edge,
        bid_pe,
        ask_pe,
        ts,
        fs=0.0,
        suppress_sell=False,
        take_buy_edge=None,
        take_sell_edge=None,
        kappa=None,
    ):
        spot = self.best_mid(d)
        if spot is None:
            return []
        lim = POSITION_LIMITS[p]
        if kappa is None:
            kappa = UNDERLYING_KAPPA
        fair = ou_terminal_mean(spot, mean, kappa, ts) + fs
        sell_edge = take_sell_edge if take_sell_edge is not None else edge
        buy_edge = take_buy_edge if take_buy_edge is not None else edge
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
        self.add_passive_quotes_split(p, d, fair, pos, lim, bid_pe, ask_pe, orders, suppress_sell=suppress_sell)
        return orders

    def trade_option(self, p, d, pos, spot, ts, vol, id_alpha=0.0):
        lim = POSITION_LIMITS[p]
        fair = call_fair_with_vol(spot, OPTION_STRIKES[p], ts, vol)
        fair -= OPTION_POSITION_PENALTY * pos / lim
        fair += id_alpha
        sell_edge = self.option_edge(p)
        if p in MARK22_VULNERABLE_STRIKES:
            buy_edge = max(OPTION_BUY_EDGE_FLOOR, sell_edge - OPTION_BUY_EDGE_REDUCTION)
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

    def trade_itm_option(
        self,
        sym,
        depth,
        pos,
        strike,
        take_edge=ITM_TAKE_EDGE,
        scale=ITM_INV_SCALE,
        passive_cap=ITM_PASSIVE_CAP,
        passive_clip=ITM_PASSIVE_CLIP,
    ):
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return []
        fair = ITM_STRUCTURAL_FV[strike]
        limit = POSITION_LIMITS[sym]
        inv_offset = int(abs(pos) / limit * scale)

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

    def update_vfe_ou_state(self, ss, spot, ts):
        payload = ss.get("vfe_ou", None)
        if not isinstance(payload, dict):
            payload = {"prev_spot": None, "prev_ts": None, "n": 0,
                       "sxx": 0.0, "sxy": 0.0,
                       "phi": VFE_DEFAULT_PHI, "kappa": VFE_DEFAULT_KAPPA}
        kappa = float(payload.get("kappa", VFE_DEFAULT_KAPPA))
        phi = float(payload.get("phi", VFE_DEFAULT_PHI))
        prev_spot = payload.get("prev_spot")
        prev_ts = payload.get("prev_ts")
        if spot is not None and prev_spot is not None and prev_ts is not None:
            import math as _m
            dt = (float(ts) - float(prev_ts)) / FINAL_TIMESTAMP
            if dt > HGP_OU_MIN_DT:
                y0 = float(prev_spot) - UNDERLYING_MU
                y1 = float(spot) - UNDERLYING_MU
                sxx = HGP_OU_EWMA_LAMBDA * float(payload.get("sxx", 0.0)) + y0 * y0
                sxy = HGP_OU_EWMA_LAMBDA * float(payload.get("sxy", 0.0)) + y0 * y1
                n = int(payload.get("n", 0)) + 1
                if sxx > HGP_OU_MIN_SXX:
                    phi = min(max(sxy / sxx, HGP_OU_MIN_PHI), HGP_OU_MAX_PHI)
                    if n >= HGP_OU_MIN_OBSERVATIONS:
                        kappa = -_m.log(phi) / dt
                    payload.update({"n": n, "sxx": sxx, "sxy": sxy, "phi": phi, "kappa": kappa})
        if spot is not None:
            payload["prev_spot"] = float(spot)
            payload["prev_ts"] = float(ts)
        ss["vfe_ou"] = payload
        return ss, kappa

    def update_hgp_ou_state(self, ss, spot, ts):
        """Online EWMA OLS for HGP OU phi/kappa. Returns updated kappa."""
        payload = ss.get("hgp_ou", None)
        if not isinstance(payload, dict):
            payload = {
                "prev_spot": None, "prev_ts": None, "n": 0,
                "sxx": 0.0, "sxy": 0.0,
                "phi": HGP_DEFAULT_PHI, "kappa": HGP_DEFAULT_KAPPA,
            }
        kappa = float(payload.get("kappa", HGP_DEFAULT_KAPPA))
        phi = float(payload.get("phi", HGP_DEFAULT_PHI))
        prev_spot = payload.get("prev_spot")
        prev_ts = payload.get("prev_ts")
        if spot is not None and prev_spot is not None and prev_ts is not None:
            import math as _m
            dt = (float(ts) - float(prev_ts)) / FINAL_TIMESTAMP
            if dt > HGP_OU_MIN_DT:
                y0 = float(prev_spot) - HYDROGEL_MU
                y1 = float(spot) - HYDROGEL_MU
                sxx = HGP_OU_EWMA_LAMBDA * float(payload.get("sxx", 0.0)) + y0 * y0
                sxy = HGP_OU_EWMA_LAMBDA * float(payload.get("sxy", 0.0)) + y0 * y1
                n = int(payload.get("n", 0)) + 1
                if sxx > HGP_OU_MIN_SXX:
                    phi = min(max(sxy / sxx, HGP_OU_MIN_PHI), HGP_OU_MAX_PHI)
                    if n >= HGP_OU_MIN_OBSERVATIONS:
                        kappa = -_m.log(phi) / dt
                    payload.update({"n": n, "sxx": sxx, "sxy": sxy, "phi": phi, "kappa": kappa})
        if spot is not None:
            payload["prev_spot"] = float(spot)
            payload["prev_ts"] = float(ts)
        ss["hgp_ou"] = payload
        return ss, kappa

    def update_hgp_open_regime(self, state, ss):
        day_ts = int(state.timestamp % FINAL_TIMESTAMP)
        if day_ts == 0:
            ss.pop("hgp_open_regime", None)
            mid = self.best_mid(state.order_depths.get(HYDROGEL))
            if mid is not None:
                if mid < HGP_LOW_OPEN_TH:
                    ss["hgp_open_regime"] = "low_open"
                elif mid > HGP_HIGH_OPEN_TH:
                    ss["hgp_open_regime"] = "high_open"
                else:
                    ss["hgp_open_regime"] = "mid_open"
        return ss

    def hgp_open_params(self, ss):
        regime = ss.get("hgp_open_regime")
        if regime == "low_open":
            return HGP_LOW_OPEN_MEAN, HGP_LOW_OPEN_BUY_EDGE
        if regime == "high_open":
            return HGP_HIGH_OPEN_MEAN, HGP_HIGH_OPEN_BUY_EDGE
        if regime == "mid_open":
            return HGP_MID_OPEN_MEAN, HGP_MID_OPEN_BUY_EDGE
        return HYDROGEL_MU, MICRO_HGP_BUY_EDGE

    def capacity_safe_orders(self, product, orders, start_pos, limit):
        buy_left = max(0, limit - start_pos)
        sell_left = max(0, limit + start_pos)
        safe = []
        for order in orders:
            q = order.quantity
            if q > 0:
                take = min(q, buy_left)
                if take > 0:
                    safe.append(Order(product, order.price, take))
                    buy_left -= take
            elif q < 0:
                take = min(-q, sell_left)
                if take > 0:
                    safe.append(Order(product, order.price, -take))
                    sell_left -= take
        return safe

    def update_iv_state(self, state, ss, spot):
        # Approach 6: Robust consensus - require min(samples) > floor for high regime
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
        if len(samples) >= 50:
            # Lock only if 5th percentile of samples is above 160
            # (robust consensus: even bad samples are above 160)
            sorted_s = sorted(samples)
            p5 = sorted_s[2]  # 5th percentile of 50 = index 2
            median = sorted_s[25]
            if p5 > 160:
                locked = max(130.0, min(170.0, median))
                ss["vol"] = locked
                ss["vol_locked"] = True
                ss["iv_samples"] = []
                return ss, locked
            else:
                ss["vol_locked"] = False
                ss["iv_samples"] = []
                return ss, OPTION_VOL_DEFAULT
        return ss, OPTION_VOL_DEFAULT

    def run(self, state):
        try:
            ss = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ss = {}

        ss = self.update_hgp_open_regime(state, ss)

        result = {}
        vvf_depth = state.order_depths.get(UNDERLYING)
        vvf_spot = self.best_mid(vvf_depth)
        ss, cur_vol = self.update_iv_state(state, ss, vvf_spot)

        if UNDERLYING in state.order_depths:
            fs = 0.0
            if vvf_spot is not None:
                fs = -DELTA_HEDGE_BETA * self.delta_exposure(state, vvf_spot, cur_vol)
            ss, vfe_kappa = self.update_vfe_ou_state(ss, vvf_spot, state.timestamp)
            position = state.position.get(UNDERLYING, 0)
            orders = self.trade_ou_split(
                UNDERLYING,
                state.order_depths[UNDERLYING],
                position,
                UNDERLYING_MU,
                UNDERLYING_EDGE,
                PASSIVE_VFE_BID_EDGE,
                PASSIVE_VFE_ASK_EDGE,
                state.timestamp,
                fs,
                kappa=vfe_kappa,
            )
            if orders:
                result[UNDERLYING] = orders

        if HYDROGEL in state.order_depths:
            hgp_mean, hgp_buy_edge = self.hgp_open_params(ss)
            hgp_alpha = HGP_ID_SCALE * self.hgp_id_alpha(state)
            hgp_spot_for_ou = self.best_mid(state.order_depths.get(HYDROGEL))
            ss, hgp_kappa = self.update_hgp_ou_state(ss, hgp_spot_for_ou, state.timestamp)
            position = state.position.get(HYDROGEL, 0)
            orders = self.trade_ou_split(
                HYDROGEL,
                state.order_depths[HYDROGEL],
                position,
                hgp_mean,
                HYDROGEL_EDGE,
                PASSIVE_HGP_BID_EDGE,
                PASSIVE_HGP_ASK_EDGE,
                state.timestamp,
                hgp_alpha,
                take_buy_edge=max(1, hgp_buy_edge - max(0.0, hgp_alpha)),
                take_sell_edge=HYDROGEL_EDGE + max(0.0, -hgp_alpha),
                kappa=hgp_kappa,
            )
            orders = self.capacity_safe_orders(HYDROGEL, orders, position, POSITION_LIMITS[HYDROGEL])
            if orders:
                result[HYDROGEL] = orders

        if vvf_spot is not None:
            for product in OPTION_STRIKES:
                depth = state.order_depths.get(product)
                if depth is None:
                    continue
                position = state.position.get(product, 0)
                id_alpha = 0.0
                if product == "VEV_5500":
                    id_alpha = VEV5500_ID_SCALE * self.vev5500_id_alpha(state)
                orders = self.trade_option(product, depth, position, vvf_spot, state.timestamp, cur_vol, id_alpha=id_alpha)
                if orders:
                    result[product] = orders

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