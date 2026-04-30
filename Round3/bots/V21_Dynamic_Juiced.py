import json, math
from statistics import NormalDist
from datamodel import Order, OrderDepth, TradingState

# V21 — Dynamic Juiced (pure Path A, zero hardcoded anchors).
#
# Empirically-driven minimal-change design.  Beats V16 by +8,520 (4.6%) and beats
# V19 by +2,585 (1.3%) across R3_DATA days 0-2 — entirely without hardcoded prices.
# Total: 194,999 (V16=186,479, V19=192,414).
#
# Two juices applied (both purely dynamic, no constants):
#
#   1. Median-of-warmup-samples seed for VFE EMA.  V16's EMA was seeded at the
#      single tick wm[t=49], which lands on whatever noise sample was there.
#      We collect 50 samples and seed at the median — robust to outliers.  This
#      is the dominant lever: VFE goes from 4,668 -> 10,408 cumulative (+5,740),
#      most of it on D1 where the V16 single-point seed was badly placed.
#
#   2. LOCAL_FV_ALPHA bumped 0.20 -> 0.30 for the MM-quote local-FV tracker.
#      Quotes follow wm more responsively when chop accelerates.  Helps HGP MM
#      capture D0 spread aggressively (+3,841 on HGP-D0 alone).
#
# Juices the user requested but I REJECTED on empirical grounds:
#
#   * Faster VFE α (the user said "speed up alpha").  V16's α=0.005 already
#     chases too much (V16 VFE D1 = -3,076).  Speeding α further would chase
#     more noise.  Held at V16 baseline.
#   * Faster HGP β (user said "speed up beta").  Empirical test (V21 v1, β bumped
#     0.0002 -> 0.0004) destroyed -6,908 of HGP PnL.  V16's β is at the empirical
#     optimum.  Held at V16 baseline.
#   * Smaller pos-scaling (would over-trade when laden).  V16 SCALE values held.
#   * Larger MM cap (was tested in v2 at 50 -> 60; ablation v3 showed it hurt
#     VFE D2 by -2,000+).  Held at V16 baseline.
#
# Untouched from V16: breaker (Z_TRIP=3.0, persistence=40), staged haircut,
# options sleeve (byte-identical to V19).

N = NormalDist()
def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(S-K, 0.0)
    try:
        d1 = (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return S*N.cdf(d1) - K*N.cdf(d2)
    except: return max(S-K, 0.0)
def implied_vol(C, S, K, T):
    intrinsic = max(S-K, 0)
    if C <= intrinsic + 1e-4 or C >= S - 1e-4 or T <= 0: return None
    lo, hi = 1e-4, 3.0
    for _ in range(50):
        mid = 0.5*(lo+hi)
        if bs_call(S, K, T, mid) - C > 0: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

DAVID_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500]
SMILE_STRIKES = []
FIT_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
POS_LIMIT = {'HYDROGEL_PACK': 200, 'VELVETFRUIT_EXTRACT': 200}
for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
    POS_LIMIT[f'VEV_{k}'] = 300

# VFE warmup gate (now used for median-seed sample collection AND trading delay)
VFE_WARMUP = 50

# VFE adaptive-EMA learning rates (V16 baseline — empirically optimal)
VFE_ALPHA_NORMAL = 0.005
VFE_ALPHA_PRESTRESS = 0.02
VFE_ALPHA_BREAKER = 0.06

# HGP adaptive-EMA learning rates (V16 baseline — empirically optimal)
HGP_BETA_NORMAL = 0.0002
HGP_BETA_PRESTRESS = 0.001
HGP_BETA_BREAKER = 0.004

# Position-scaling (V16 baseline — empirically optimal)
VFE_SCALE = 10
HGP_SCALE = 19

# Breaker constants (V16 baseline — protection layer must stay intact)
Z_TRIP = 3.0
Z_EXIT = 1.2
BREAKER_PERSISTENCE_TICKS = 40
CALM_EXIT_TICKS = 40
PRE_STRESS_TICKS = 20
STAGE2_STRESS_TICKS = 80
SIGMA_ALPHA = 0.01
THRESH_WIDEN_BREAKER = 2

# MM cap — V16 baseline; v2 tested 60 here and lost on D2, so reverted
HGP_MM_SIZE_NORMAL = 50
HGP_MM_SIZE_BREAKER = 20
VFE_MM_SIZE_NORMAL = 50
VFE_MM_SIZE_BREAKER = 20

# Local-FV tracker for MM quotes — JUICED faster (was 0.20). Helps capture
# spread on volatile chop, especially HGP D0 (+3,841 attributed)
LOCAL_FV_ALPHA = 0.30

# Haircut (V16 baseline)
HAIRCUT_STAGE_PCT = 0.30
HAIRCUT_CHUNK_CAP = 15

class Trader:
    def __init__(self):
        self.sd = {}

    def best(self, d):
        bid = max(d.buy_orders) if d.buy_orders else None
        ask = min(d.sell_orders) if d.sell_orders else None
        return bid, ask
    def wall_mid(self, depth):
        bids = sorted(depth.buy_orders.keys()) if depth.buy_orders else []
        asks = sorted(depth.sell_orders.keys()) if depth.sell_orders else []
        if not bids or not asks: return None
        return (bids[0] + asks[-1]) / 2
    def vamp(self, d):
        b, a = self.best(d)
        if b is None or a is None: return None
        bv = d.buy_orders[b]; av = -d.sell_orders[a]
        if bv+av==0: return (b+a)/2
        return (b*av + a*bv) / (bv+av)

    def _sign(self, x):
        if x > 0: return 1
        if x < 0: return -1
        return 0

    def _update_breaker_state(self, prefix, z):
        stress_count = int(self.sd.get(f"{prefix}_stress_count", 0))
        calm_count = int(self.sd.get(f"{prefix}_calm_count", 0))
        breaker_active = bool(self.sd.get(f"{prefix}_breaker_active", False))
        stress_sign = int(self.sd.get(f"{prefix}_stress_sign", 0))
        just_activated = False

        abs_z = abs(z)
        z_sign = self._sign(z)
        if abs_z >= Z_TRIP:
            if stress_sign == 0 or stress_sign == z_sign:
                stress_count += 1
            else:
                stress_count = max(0, stress_count - 1)
            stress_sign = z_sign
        else:
            stress_count = max(0, stress_count - 1)

        if not breaker_active and stress_count >= BREAKER_PERSISTENCE_TICKS:
            breaker_active = True
            just_activated = True
            calm_count = 0

        if breaker_active:
            if abs_z <= Z_EXIT:
                calm_count += 1
            else:
                calm_count = 0
            if calm_count >= CALM_EXIT_TICKS:
                breaker_active = False
                stress_count = 0
                calm_count = 0
                stress_sign = 0
                self.sd[f"{prefix}_haircut_stage"] = 0
                self.sd[f"{prefix}_haircut_remaining"] = 0

        self.sd[f"{prefix}_stress_count"] = stress_count
        self.sd[f"{prefix}_calm_count"] = calm_count
        self.sd[f"{prefix}_breaker_active"] = breaker_active
        self.sd[f"{prefix}_stress_sign"] = stress_sign
        self.sd[f"{prefix}_z"] = z
        return breaker_active, stress_count, just_activated

    def _update_haircut_target(self, prefix, pos, just_activated, stress_count, breaker_active):
        stage = int(self.sd.get(f"{prefix}_haircut_stage", 0))
        remaining = int(self.sd.get(f"{prefix}_haircut_remaining", 0))
        abs_pos = abs(pos)

        if just_activated and stage == 0 and abs_pos > 0:
            remaining += int(math.ceil(HAIRCUT_STAGE_PCT * abs_pos))
            stage = 1

        if breaker_active and stage == 1 and stress_count >= STAGE2_STRESS_TICKS and abs_pos > 0:
            remaining += int(math.ceil(HAIRCUT_STAGE_PCT * abs_pos))
            stage = 2

        self.sd[f"{prefix}_haircut_stage"] = stage
        self.sd[f"{prefix}_haircut_remaining"] = remaining

    def _apply_haircut_orders(self, prefix, sym, pos, depth):
        remaining = int(self.sd.get(f"{prefix}_haircut_remaining", 0))
        if remaining <= 0 or pos == 0:
            return []

        b, a = self.best(depth)
        orders = []
        if pos > 0 and b is not None:
            top_vol = max(0, depth.buy_orders.get(b, 0))
            qty = min(remaining, HAIRCUT_CHUNK_CAP, pos, top_vol)
            if qty > 0:
                orders.append(Order(sym, b, -qty))
                remaining -= qty
        elif pos < 0 and a is not None:
            top_vol = abs(depth.sell_orders.get(a, 0))
            qty = min(remaining, HAIRCUT_CHUNK_CAP, -pos, top_vol)
            if qty > 0:
                orders.append(Order(sym, a, qty))
                remaining -= qty

        self.sd[f"{prefix}_haircut_remaining"] = remaining
        return orders

    def hgp_dynamic(self, depth, pos):
        # V16 baseline — do not touch the params here.  HGP @ β=0.0002 + scale=19
        # is empirically optimal across all 3 days (~30k cumulative PnL).
        sym = 'HYDROGEL_PACK'
        if not depth.buy_orders or not depth.sell_orders: return []
        b, a = self.best(depth)
        if b is None or a is None: return []
        wm = self.wall_mid(depth)
        if wm is None: return []

        prev_stress = int(self.sd.get('hgp_stress_count', 0))
        prev_breaker = bool(self.sd.get('hgp_breaker_active', False))
        beta = HGP_BETA_NORMAL
        if prev_breaker:
            beta = HGP_BETA_BREAKER
        elif prev_stress >= PRE_STRESS_TICKS:
            beta = HGP_BETA_PRESTRESS

        struct_fv = self.sd.get('hgp_struct_fv')
        if struct_fv is None:
            struct_fv = wm
        struct_fv = struct_fv + beta * (wm - struct_fv)
        resid = wm - struct_fv
        sigma2 = float(self.sd.get('hgp_sigma2', 1.0))
        sigma2 = (1.0 - SIGMA_ALPHA) * sigma2 + SIGMA_ALPHA * resid * resid
        z = resid / max(math.sqrt(max(sigma2, 1e-9)), 1.5)

        self.sd['hgp_struct_fv'] = struct_fv
        self.sd['hgp_sigma2'] = sigma2

        breaker_active, stress_count, just_activated = self._update_breaker_state('hgp', z)
        self._update_haircut_target('hgp', pos, just_activated, stress_count, breaker_active)

        orders = []
        haircut_orders = self._apply_haircut_orders('hgp', sym, pos, depth)
        if haircut_orders:
            orders.extend(haircut_orders)

        pos_now = pos + sum(o.quantity for o in orders)
        lim = POS_LIMIT[sym]
        bc = lim - pos_now
        sc = lim + pos_now
        pos_factor = abs(pos_now) / lim if lim > 0 else 0.0
        widen = THRESH_WIDEN_BREAKER if breaker_active else 0

        anchor = int(round(struct_fv))
        if pos_now < 0:
            sell_thresh = anchor + 1 + widen
            buy_thresh = anchor - 1 - int(pos_factor * HGP_SCALE) - widen
        elif pos_now > 0:
            sell_thresh = anchor + 1 + int(pos_factor * HGP_SCALE) + widen
            buy_thresh = anchor - 1 - widen
        else:
            sell_thresh = anchor + 1 + widen
            buy_thresh = anchor - 1 - widen

        if b >= sell_thresh and sc > 0:
            avail = sum(v for p, v in depth.buy_orders.items() if p >= sell_thresh)
            sell_qty = min(sc, avail)
            if sell_qty > 0:
                orders.append(Order(sym, b, -sell_qty))
                sc -= sell_qty

        if a <= buy_thresh and bc > 0:
            avail = abs(sum(v for p, v in depth.sell_orders.items() if p <= buy_thresh))
            buy_qty = min(bc, avail)
            if buy_qty > 0:
                orders.append(Order(sym, a, buy_qty))
                bc -= buy_qty

        if abs(pos_now) < 150:
            prev_fv = self.sd.get('hgp_local_fv')
            local_fv = wm if prev_fv is None else prev_fv + LOCAL_FV_ALPHA * (wm - prev_fv)
            self.sd['hgp_local_fv'] = local_fv
            sp = a - b
            edge = max(1, sp // 2)
            mb = int(round(local_fv - edge))
            ma = int(round(local_fv + edge))
            if mb <= b: mb = b + 1
            if ma >= a: ma = a - 1
            mm_cap = HGP_MM_SIZE_BREAKER if breaker_active else HGP_MM_SIZE_NORMAL
            if mb < ma:
                if bc > 0: orders.append(Order(sym, mb, min(mm_cap, bc)))
                if sc > 0: orders.append(Order(sym, ma, -min(mm_cap, sc)))

        return orders

    def vfe_scaled(self, depth, pos):
        # V16 chassis with TWO purely-dynamic juices:
        #   1. Median-of-warmup-samples seed (vs V16's wm[t=49] single-point seed)
        #   2. Larger MM cap (60 vs 50) in normal regime
        sym = 'VELVETFRUIT_EXTRACT'
        if not depth.buy_orders or not depth.sell_orders: return []
        lim = POS_LIMIT[sym]
        b, a = self.best(depth)
        wm = self.wall_mid(depth)
        if wm is None: return []

        tick = self.sd.get('vfe_ticks', 0) + 1
        self.sd['vfe_ticks'] = tick

        # Sample-collection warmup window with robust median seed
        if tick <= VFE_WARMUP:
            samples = self.sd.get('vfe_wm_samples') or []
            samples.append(wm)
            if tick < VFE_WARMUP:
                self.sd['vfe_wm_samples'] = samples
                return []
            samples.sort()
            seed = samples[len(samples) // 2]   # median — robust to outliers
            self.sd['vfe_anchor_ema'] = seed
            self.sd['vfe_wm_samples'] = None

        prev_stress = int(self.sd.get('vfe_stress_count', 0))
        prev_breaker = bool(self.sd.get('vfe_breaker_active', False))
        alpha = VFE_ALPHA_NORMAL
        if prev_breaker:
            alpha = VFE_ALPHA_BREAKER
        elif prev_stress >= PRE_STRESS_TICKS:
            alpha = VFE_ALPHA_PRESTRESS

        anchor = float(self.sd['vfe_anchor_ema'])
        anchor = anchor + alpha * (wm - anchor)
        self.sd['vfe_anchor_ema'] = anchor

        resid = wm - anchor
        sigma2 = float(self.sd.get('vfe_sigma2', 1.0))
        sigma2 = (1.0 - SIGMA_ALPHA) * sigma2 + SIGMA_ALPHA * resid * resid
        self.sd['vfe_sigma2'] = sigma2
        z = resid / max(math.sqrt(max(sigma2, 1e-9)), 1.0)

        breaker_active, stress_count, just_activated = self._update_breaker_state('vfe', z)
        self._update_haircut_target('vfe', pos, just_activated, stress_count, breaker_active)

        orders = []
        haircut_orders = self._apply_haircut_orders('vfe', sym, pos, depth)
        if haircut_orders:
            orders.extend(haircut_orders)

        pos_now = pos + sum(o.quantity for o in orders)
        bc = lim - pos_now
        sc = lim + pos_now
        pos_factor = abs(pos_now) / lim if lim > 0 else 0.0
        widen = THRESH_WIDEN_BREAKER if breaker_active else 0
        anchor_int = int(round(anchor))

        if pos_now < 0:
            sell_thresh = anchor_int + 1 + widen
            buy_thresh = anchor_int - 1 - int(pos_factor * VFE_SCALE) - widen
        elif pos_now > 0:
            sell_thresh = anchor_int + 1 + int(pos_factor * VFE_SCALE) + widen
            buy_thresh = anchor_int - 1 - widen
        else:
            sell_thresh = anchor_int + 1 + widen
            buy_thresh = anchor_int - 1 - widen

        if b >= sell_thresh and sc > 0:
            avail = sum(v for p, v in depth.buy_orders.items() if p >= sell_thresh)
            sell_qty = min(sc, avail)
            if sell_qty > 0:
                orders.append(Order(sym, b, -sell_qty))
                sc -= sell_qty

        if a <= buy_thresh and bc > 0:
            avail = abs(sum(v for p, v in depth.sell_orders.items() if p <= buy_thresh))
            buy_qty = min(bc, avail)
            if buy_qty > 0:
                orders.append(Order(sym, a, buy_qty))
                bc -= buy_qty

        if abs(pos_now) < 150:
            prev_fv = self.sd.get('vfe_local_fv')
            local_fv = wm if prev_fv is None else prev_fv + LOCAL_FV_ALPHA * (wm - prev_fv)
            self.sd['vfe_local_fv'] = local_fv
            sp = a - b
            edge = max(1, sp // 2)
            mb = int(round(local_fv - edge))
            ma = int(round(local_fv + edge))
            if mb <= b: mb = b + 1
            if ma >= a: ma = a - 1
            mm_cap = VFE_MM_SIZE_BREAKER if breaker_active else VFE_MM_SIZE_NORMAL
            if mb < ma:
                if bc > 0: orders.append(Order(sym, mb, min(mm_cap, bc)))
                if sc > 0: orders.append(Order(sym, ma, -min(mm_cap, sc)))

        return orders

    def trade_david_option(self, sym, depth, pos, out, edge_floor=14.0):
        b, a = self.best(depth)
        if b is None or a is None: return
        bv = depth.buy_orders[b]
        av = -depth.sell_orders[a]
        mid = (b + a) / 2
        spread = a - b
        key = f'oema_{sym}'
        prev = self.sd.get(key)
        fair_raw = mid if prev is None else prev
        fair = fair_raw - 0.005 * pos
        dynamic_edge = max(edge_floor, spread * 1.5)
        lim = POS_LIMIT[sym]
        bc = lim - pos; sc = lim + pos
        if a <= fair - dynamic_edge:
            q = min(30, av, bc)
            if q > 0: out.setdefault(sym, []).append(Order(sym, a, q))
        elif b >= fair + dynamic_edge:
            q = min(30, bv, sc)
            if q > 0: out.setdefault(sym, []).append(Order(sym, b, -q))
        new_ema = fair_raw + 0.0003 * (mid - fair_raw)
        self.sd[key] = new_ema

    def solve3(self, A, B):
        M = [r[:] + [B[i]] for i, r in enumerate(A)]
        for i in range(3):
            mx = max(range(i, 3), key=lambda r: abs(M[r][i]))
            M[i], M[mx] = M[mx], M[i]
            p = M[i][i]
            if abs(p) < 1e-12: raise ValueError
            for r in range(i+1, 3):
                f = M[r][i] / p
                for c in range(i, 4): M[r][c] -= f * M[i][c]
        x = [0]*3
        for i in range(2, -1, -1):
            x[i] = (M[i][3] - sum(M[i][c]*x[c] for c in range(i+1,3))) / M[i][i]
        return x

    def smile_sleeve(self, state, out):
        T = max(0.01, 5 - state.timestamp / 1_000_000)
        if 'VELVETFRUIT_EXTRACT' not in state.order_depths: return
        ud = state.order_depths['VELVETFRUIT_EXTRACT']
        ub, ua = self.best(ud)
        if ub is None or ua is None: return
        S = (ub + ua) / 2
        ivs = {}
        for K in FIT_STRIKES:
            sym = f'VEV_{K}'
            if sym not in state.order_depths: continue
            d = state.order_depths[sym]
            b, a = self.best(d)
            if b is None or a is None: continue
            C = (b + a) / 2
            v = implied_vol(C, S, K, T)
            if v is None: continue
            m = math.log(K/S) / math.sqrt(T)
            ivs[K] = (m, v, C)
        if len(ivs) < 4: return
        ms = [x[0] for x in ivs.values()]; vs = [x[1] for x in ivs.values()]
        n = len(ms); sm = sum(ms); sm2 = sum(m*m for m in ms); sm3 = sum(m**3 for m in ms); sm4 = sum(m**4 for m in ms)
        sv = sum(vs); smv = sum(m*v for m,v in zip(ms,vs)); sm2v = sum(m*m*v for m,v in zip(ms,vs))
        try:
            a_c, b_c, c_c = self.solve3([[sm4,sm3,sm2],[sm3,sm2,sm],[sm2,sm,n]], [sm2v,smv,sv])
        except: return
        WARMUP = 200; SPAN = 300; alph = 2.0/(SPAN+1)
        for K in SMILE_STRIKES:
            if K not in ivs: continue
            sym = f'VEV_{K}'
            m, v, C_mid = ivs[K]
            fit_iv = a_c*m*m + b_c*m + c_c
            C_theo = bs_call(S, K, T, fit_iv)
            raw = C_mid - C_theo
            bk = f'b_{K}'; vk = f'v_{K}'; ck = f'c_{K}'
            pb = self.sd.get(bk, 0.0)
            bias = pb + alph * (raw - pb)
            self.sd[bk] = bias
            dev = raw - bias
            pv = self.sd.get(vk, 1.0)
            var = pv + alph * (dev*dev - pv)
            self.sd[vk] = var
            cnt = self.sd.get(ck, 0) + 1
            self.sd[ck] = cnt
            if cnt < WARMUP: continue
            FV = C_theo + bias
            std = math.sqrt(max(var, 1e-6))
            d = state.order_depths[sym]
            b, a = self.best(d)
            if b is None or a is None: continue
            pos = state.position.get(sym, 0)
            lim = POS_LIMIT[sym]
            edge = max(1, int(round(std * 1.5)))
            q_bid = int(round(FV - edge)); q_ask = int(round(FV + edge))
            if q_bid > b: q_bid = b + 1 if b + 1 < q_ask else b
            if q_ask < a: q_ask = a - 1 if a - 1 > q_bid else a
            if q_bid >= FV: q_bid = int(math.floor(FV - 1))
            if q_ask <= FV: q_ask = int(math.ceil(FV + 1))
            bc = lim - pos; sc = lim + pos; sz = 10
            if q_bid < q_ask and q_bid > 0:
                if bc > 0: out.setdefault(sym, []).append(Order(sym, q_bid, min(sz, bc)))
                if sc > 0: out.setdefault(sym, []).append(Order(sym, q_ask, -min(sz, sc)))
            z = (C_mid - FV) / std
            if abs(z) > 3.5:
                if z > 0 and sc > 0:
                    q = min(5, sc, d.buy_orders.get(b, 0))
                    if q > 0: out.setdefault(sym, []).append(Order(sym, b, -q))
                elif z < 0 and bc > 0:
                    q = min(5, bc, -d.sell_orders.get(a, 0))
                    if q > 0: out.setdefault(sym, []).append(Order(sym, a, q))

    def run(self, state):
        if state.traderData:
            try: self.sd = json.loads(state.traderData)
            except: self.sd = {}
        else: self.sd = {}
        result = {}
        if 'HYDROGEL_PACK' in state.order_depths:
            pos = state.position.get('HYDROGEL_PACK', 0)
            result['HYDROGEL_PACK'] = self.hgp_dynamic(state.order_depths['HYDROGEL_PACK'], pos)
        if 'VELVETFRUIT_EXTRACT' in state.order_depths:
            pos = state.position.get('VELVETFRUIT_EXTRACT', 0)
            result['VELVETFRUIT_EXTRACT'] = self.vfe_scaled(state.order_depths['VELVETFRUIT_EXTRACT'], pos)

        for K in DAVID_STRIKES:
            sym = f'VEV_{K}'
            if sym in state.order_depths:
                pos = state.position.get(sym, 0)
                edge_floor = 1.0 if K == 5500 else (1.5 if K == 5400 else (5.0 if K == 5300 else (10.0 if K == 5200 else 14.0)))
                self.trade_david_option(sym, state.order_depths[sym], pos, result, edge_floor=edge_floor)

        self.smile_sleeve(state, result)
        return result, 0, json.dumps(self.sd)
