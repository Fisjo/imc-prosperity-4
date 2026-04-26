"""
v26_no_fv — v25 con CERO valores hardcodeados (sin FVs).

Cambios vs v25 (los 4 puntos donde había FV fijados):
  L32: VFE_ANCHOR_FOR_OPTIONS=5250  → derivado de VFE online
  L70: VFE_ANCHOR=5250              → mediana móvil de wm (W=2000, warmup=200)
  L108: vfe_ema seed=5250.0         → seed = anchor online del momento
  L145: HGP FAIR=9985               → mediana móvil de wm (W=2000, warmup=200)

Conservado de v25 (no son FVs, son meta-parámetros):
  SCALE, TAKE_EDGE, TAKE_CLIP, edge_floors, αs, βs, smile sleeve.

Coste empírico (medido en R3 days 0-2, sólo VFE):
  v25 static (5250 hardcoded) + cap=15:  +98,963
  v26 rolling median W=2000 + cap=15:    +89,846  (-9,117  / -9.2%)

Beneficio:
  - Portable a Round 4 sin recalibrar
  - No depende de conocer el equilibrio del simulador
  - Sigue cualquier μ-shift entre rondas

Detalles del estimador:
  - Mediana sobre wall_mid de los últimos 2000 ticks (≈10× half-life)
  - Warmup: 200 ticks sin operar (evita anchor sesgado por trayectoria inicial)
  - Mediana en lugar de media: robusta a outliers; usa la propiedad de que
    para un OU simétrico, mediana = media = μ pero la mediana converge mejor
    cuando la trayectoria no ha visitado ambos lados igualmente
  - Para opciones ITM (deep-ITM ≈ S - K): se reusa el VFE_ANCHOR online
    en lugar de hardcodear 5250 - K
"""

import json, math
from statistics import NormalDist
from datamodel import Order, OrderDepth, TradingState

N = NormalDist()


def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    try:
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * N.cdf(d1) - K * N.cdf(d2)
    except:
        return max(S - K, 0.0)


def implied_vol(C, S, K, T):
    intrinsic = max(S - K, 0)
    if C <= intrinsic + 1e-4 or C >= S - 1e-4 or T <= 0:
        return None
    lo, hi = 1e-4, 3.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) - C > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


DAVID_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500]
SMILE_STRIKES = []
FIT_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
ITM_STRIKES = (4000, 4500)

POS_LIMIT = {"HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200}
for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
    POS_LIMIT[f"VEV_{k}"] = 300

# Online-anchor hyperparams (NOT prices — these are estimator settings, not FVs)
ANCHOR_WINDOW = 2000  # rolling window size
ANCHOR_WARMUP = 200  # ticks before any trading begins
V33_EDGE = 4.0
V33_Z_ENTRY = 1.5
V33_MIN_OBSERVATIONS = 200
V33_PRODUCTS = tuple(f"VEV_{k}" for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500])


class ExpandingZScore:
    def __init__(self, state=None) -> None:
        state = state or {}
        self.n = int(state.get("n", 0))
        self.mean = float(state.get("mean", 0.0))
        self.m2 = float(state.get("m2", 0.0))

    def to_json(self):
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    def update(self, price: float) -> None:
        self.n += 1
        delta = price - self.mean
        self.mean += delta / self.n
        delta_after_update = price - self.mean
        self.m2 += delta * delta_after_update

    def ready(self) -> bool:
        return self.n >= V33_MIN_OBSERVATIONS

    def z_score(self, price: float) -> float:
        variance = self.m2 / max(1, self.n - 1)
        return (price - self.mean) / math.sqrt(max(variance, 1e-9))


class Trader:
    def __init__(self):
        self.sd = {}
        self.v33_estimators = {product: ExpandingZScore() for product in V33_PRODUCTS}

    def _restore_v33_estimators(self):
        raw_state = self.sd.get("v33_estimators", {})
        self.v33_estimators = {
            product: ExpandingZScore(raw_state.get(product, {}))
            for product in V33_PRODUCTS
        }

    def _store_v33_estimators(self):
        self.sd["v33_estimators"] = {
            product: estimator.to_json()
            for product, estimator in self.v33_estimators.items()
        }

    def best(self, d):
        bid = max(d.buy_orders) if d.buy_orders else None
        ask = min(d.sell_orders) if d.sell_orders else None
        return bid, ask

    def wall_mid(self, depth):
        bids = sorted(depth.buy_orders.keys()) if depth.buy_orders else []
        asks = sorted(depth.sell_orders.keys()) if depth.sell_orders else []
        if not bids or not asks:
            return None
        return (bids[0] + asks[-1]) / 2

    def vamp(self, d):
        b, a = self.best(d)
        if b is None or a is None:
            return None
        bv = d.buy_orders[b]
        av = -d.sell_orders[a]
        if bv + av == 0:
            return (b + a) / 2
        return (b * av + a * bv) / (bv + av)

    def all_level_microprice(self, depth):
        if not depth.buy_orders or not depth.sell_orders:
            return None

        bid_volume = sum(volume for volume in depth.buy_orders.values() if volume > 0)
        ask_volume = sum(-volume for volume in depth.sell_orders.values() if volume < 0)
        if bid_volume <= 0 or ask_volume <= 0:
            return None

        bid_vwap = (
            sum(price * volume for price, volume in depth.buy_orders.items() if volume > 0)
            / bid_volume
        )
        ask_vwap = (
            sum(price * -volume for price, volume in depth.sell_orders.items() if volume < 0)
            / ask_volume
        )
        return (bid_vwap * ask_volume + ask_vwap * bid_volume) / (bid_volume + ask_volume)

    def trade_v33_option(self, product, depth, position):
        microprice = self.all_level_microprice(depth)
        if microprice is None:
            return []

        estimator = self.v33_estimators[product]
        orders = []
        if estimator.ready():
            z_score = estimator.z_score(microprice)
            fair_value = estimator.mean
            limit = POS_LIMIT[product]
            buy_capacity = limit - position
            sell_capacity = limit + position

            if z_score <= -V33_Z_ENTRY:
                for ask_price, ask_volume in sorted(depth.sell_orders.items()):
                    if ask_price > fair_value - V33_EDGE or buy_capacity <= 0:
                        break
                    quantity = min(-ask_volume, buy_capacity)
                    if quantity > 0:
                        orders.append(Order(product, ask_price, quantity))
                        buy_capacity -= quantity

            if z_score >= V33_Z_ENTRY:
                for bid_price, bid_volume in sorted(depth.buy_orders.items(), reverse=True):
                    if bid_price < fair_value + V33_EDGE or sell_capacity <= 0:
                        break
                    quantity = min(bid_volume, sell_capacity)
                    if quantity > 0:
                        orders.append(Order(product, bid_price, -quantity))
                        sell_capacity -= quantity

        estimator.update(microprice)
        return orders

    def online_anchor(self, key, sample):
        """Update rolling history and return current median, or None if warming up.
        State stored in self.sd[f'{key}_hist'] as a list of floats."""
        hist_key = f"{key}_hist"
        hist = self.sd.get(hist_key) or []
        hist.append(sample)
        if len(hist) > ANCHOR_WINDOW:
            hist = hist[-ANCHOR_WINDOW:]
        self.sd[hist_key] = hist
        if len(hist) < ANCHOR_WARMUP:
            return None
        srt = sorted(hist)
        return srt[len(srt) // 2]

    def vfe_scaled(self, depth, pos):
        """v25 chassis with VFE_ANCHOR replaced by online rolling median."""
        sym = "VELVETFRUIT_EXTRACT"
        if not depth.buy_orders or not depth.sell_orders:
            return []
        lim = POS_LIMIT[sym]
        orders = []
        b, a = self.best(depth)
        wm = self.wall_mid(depth)
        if wm is None:
            return []

        # ---- ONLINE ANCHOR (replaces hardcoded 5250) ----
        anchor_f = self.online_anchor("vfe_anchor", wm)
        if anchor_f is None:
            # During warmup we still record but do not trade
            return []
        VFE_ANCHOR = int(round(anchor_f))
        # Cache for ITM options to reuse (avoids double-computing)
        self.sd["vfe_anchor_current"] = VFE_ANCHOR

        bc = lim - pos
        sc = lim + pos
        pos_factor = abs(pos) / lim
        SCALE = 19

        if pos < 0:
            sell_thresh = VFE_ANCHOR + 1
            buy_thresh = VFE_ANCHOR - 1 - int(pos_factor * SCALE)
        elif pos > 0:
            sell_thresh = VFE_ANCHOR + 1 + int(pos_factor * SCALE)
            buy_thresh = VFE_ANCHOR - 1
        else:
            sell_thresh = VFE_ANCHOR + 1
            buy_thresh = VFE_ANCHOR - 1

        if b is not None and b >= sell_thresh:
            avail = sum(v for p, v in depth.buy_orders.items() if p >= sell_thresh)
            sell_qty = min(sc, avail)
            if sell_qty > 0:
                orders.append(Order(sym, b, -sell_qty))
                sc -= sell_qty

        if a is not None and a <= buy_thresh:
            avail = abs(sum(v for p, v in depth.sell_orders.items() if p <= buy_thresh))
            buy_qty = min(bc, avail)
            if buy_qty > 0:
                orders.append(Order(sym, a, buy_qty))
                bc -= buy_qty

        # ---- VFE EMA-take overlay (replaces seed=5250.0) ----
        # Seed the EMA with the current online anchor instead of the hardcoded 5250.0
        VFE_EMA_ALPHA = 0.005
        VFE_EMA_TE = 14
        mid_for_ema = (b + a) / 2
        prev_ema = self.sd.get("vfe_ema_seeded")
        if prev_ema is None:
            prev_ema = float(VFE_ANCHOR)  # seed from online anchor, not 5250.0
        ema = prev_ema + VFE_EMA_ALPHA * (mid_for_ema - prev_ema)
        self.sd["vfe_ema_seeded"] = ema
        if a <= ema - VFE_EMA_TE and bc > 0:
            avail2 = abs(
                sum(v for p, v in depth.sell_orders.items() if p <= ema - VFE_EMA_TE)
            )
            q = min(bc, avail2)
            if q > 0:
                orders.append(Order(sym, a, q))
                bc -= q
        if b >= ema + VFE_EMA_TE and sc > 0:
            avail2 = sum(
                v for p, v in depth.buy_orders.items() if p >= ema + VFE_EMA_TE
            )
            q = min(sc, avail2)
            if q > 0:
                orders.append(Order(sym, b, -q))
                sc -= q

        if abs(pos) < 150:
            prev_fv = self.sd.get("vfe_local_fv")
            local_fv = wm if prev_fv is None else prev_fv + 0.2 * (wm - prev_fv)
            self.sd["vfe_local_fv"] = local_fv
            sp = a - b
            edge = max(1, sp // 2)
            mb = int(round(local_fv - edge))
            ma = int(round(local_fv + edge))
            if mb <= b:
                mb = b + 1
            if ma >= a:
                ma = a - 1
            if mb < ma:
                if bc > 0:
                    orders.append(Order(sym, mb, min(50, bc)))
                if sc > 0:
                    orders.append(Order(sym, ma, -min(50, sc)))
        return orders

    def hgp_scaled(self, depth, pos):
        """v25 chassis with HGP FAIR replaced by online rolling median."""
        sym = "HYDROGEL_PACK"
        if not depth.buy_orders or not depth.sell_orders:
            return []
        wm = self.wall_mid(depth)
        if wm is None:
            return []

        # ---- ONLINE FAIR (replaces hardcoded 9985) ----
        fair_f = self.online_anchor("hgp_anchor", wm)
        if fair_f is None:
            return []
        FAIR = int(round(fair_f))

        TAKE_EDGE = 42
        TAKE_CLIP = 20
        PASSIVE_CLIP = 10
        SKEW = 4.0
        LIMIT = POS_LIMIT[sym]

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        orders = []
        bc = max(0, LIMIT - pos)
        sc = max(0, LIMIT + pos)
        sim_pos = pos

        buy_left = bc
        for ask_price in sorted(depth.sell_orders):
            if ask_price > FAIR - TAKE_EDGE or buy_left <= 0:
                break
            available = -depth.sell_orders[ask_price]
            quantity = min(available, buy_left, TAKE_CLIP)
            if quantity > 0:
                orders.append(Order(sym, ask_price, quantity))
                buy_left -= quantity
                sim_pos += quantity

        sell_left = sc
        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price < FAIR + TAKE_EDGE or sell_left <= 0:
                break
            available = depth.buy_orders[bid_price]
            quantity = min(available, sell_left, TAKE_CLIP)
            if quantity > 0:
                orders.append(Order(sym, bid_price, -quantity))
                sell_left -= quantity
                sim_pos -= quantity

        used_buy = bc - buy_left
        used_sell = sc - sell_left
        bc_left = max(0, bc - used_buy)
        sc_left = max(0, sc - used_sell)

        adjusted_fair = FAIR - SKEW * (sim_pos / LIMIT)
        buy_quote = min(best_bid + 1, int(adjusted_fair))
        sell_base = adjusted_fair
        sell_quote = max(
            best_ask - 1,
            int(sell_base) if sell_base == int(sell_base) else int(sell_base) + 1,
        )

        if bc_left > 0 and buy_quote < best_ask and buy_quote < sell_quote:
            quantity = min(PASSIVE_CLIP, bc_left)
            if quantity > 0:
                orders.append(Order(sym, buy_quote, quantity))

        if sc_left > 0 and sell_quote > best_bid and sell_quote > buy_quote:
            quantity = min(PASSIVE_CLIP, sc_left)
            if quantity > 0:
                orders.append(Order(sym, sell_quote, -quantity))

        return orders

    def mm_rw(
        self,
        sym,
        depth,
        pos,
        key,
        alpha=0.2,
        take_w=1,
        clear_w=2,
        max_sz=50,
        use_wall=False,
    ):
        lim = POS_LIMIT[sym]
        orders = []
        v = self.wall_mid(depth) if use_wall else self.vamp(depth)
        if v is None:
            return orders
        prev = self.sd.get(key)
        fv = v if prev is None else prev + alpha * (v - prev)
        self.sd[key] = fv
        b, a = self.best(depth)
        bc = lim - pos
        sc = lim + pos
        for px in sorted(depth.sell_orders):
            if px <= fv - take_w and bc > 0:
                vol = min(-depth.sell_orders[px], bc)
                if vol > 0:
                    orders.append(Order(sym, px, vol))
                    bc -= vol
        for px in sorted(depth.buy_orders, reverse=True):
            if px >= fv + take_w and sc > 0:
                vol = min(depth.buy_orders[px], sc)
                if vol > 0:
                    orders.append(Order(sym, px, -vol))
                    sc -= vol
        pa = pos + sum(o.quantity for o in orders)
        if pa > 0:
            p = int(round(fv + clear_w))
            if p in depth.buy_orders and sc > 0:
                v2 = min(depth.buy_orders[p], pa, sc)
                if v2 > 0:
                    orders.append(Order(sym, p, -v2))
                    sc -= v2
        elif pa < 0:
            p = int(round(fv - clear_w))
            if p in depth.sell_orders and bc > 0:
                v2 = min(-depth.sell_orders[p], -pa, bc)
                if v2 > 0:
                    orders.append(Order(sym, p, v2))
                    bc -= v2
        if b is not None and a is not None:
            sp = a - b
            edge = max(1, sp // 2)
            mb = int(round(fv - edge))
            ma = int(round(fv + edge))
            if mb <= b:
                mb = b + 1
            if ma >= a:
                ma = a - 1
            if mb < ma:
                if bc > 0:
                    orders.append(Order(sym, mb, min(max_sz, bc)))
                if sc > 0:
                    orders.append(Order(sym, ma, -min(max_sz, sc)))
        return orders

    def trade_itm_option(
        self, sym, depth, pos, out, K, take_edge=1, scale=19, passive_clip=50
    ):
        """ITM (deep) options. FV = E[S] - K, where E[S] is the ONLINE VFE anchor.
        Replaces v25's hardcoded ITM_STRUCTURAL_FV = 5250 - K."""
        if not depth.buy_orders or not depth.sell_orders:
            return
        # Reuse VFE online anchor (already computed in vfe_scaled this tick)
        vfe_anchor = self.sd.get("vfe_anchor_current")
        if vfe_anchor is None:
            return  # still warming up
        fair = vfe_anchor - K

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        lim = POS_LIMIT[sym]
        bc = max(0, lim - pos)
        sc = max(0, lim + pos)
        inv_offset = int(abs(pos) / lim * scale)
        if pos < 0:
            buy_thresh = fair - take_edge - inv_offset
            sell_thresh = fair + take_edge
        elif pos > 0:
            buy_thresh = fair - take_edge
            sell_thresh = fair + take_edge + inv_offset
        else:
            buy_thresh = fair - take_edge
            sell_thresh = fair + take_edge

        if best_ask <= buy_thresh and bc > 0:
            avail = abs(sum(v for p, v in depth.sell_orders.items() if p <= buy_thresh))
            q = min(bc, avail)
            if q > 0:
                out.setdefault(sym, []).append(Order(sym, best_ask, q))
                bc -= q
        if best_bid >= sell_thresh and sc > 0:
            avail = sum(v for p, v in depth.buy_orders.items() if p >= sell_thresh)
            q = min(sc, avail)
            if q > 0:
                out.setdefault(sym, []).append(Order(sym, best_bid, -q))
                sc -= q

        if abs(pos) < 150:
            spread = best_ask - best_bid
            edge = max(1, spread // 2)
            buy_quote = int(round(fair - edge))
            sell_quote = int(round(fair + edge))
            if buy_quote <= best_bid:
                buy_quote = best_bid + 1
            if sell_quote >= best_ask:
                sell_quote = best_ask - 1
            if buy_quote < sell_quote:
                if bc > 0:
                    out.setdefault(sym, []).append(
                        Order(sym, buy_quote, min(passive_clip, bc))
                    )
                if sc > 0:
                    out.setdefault(sym, []).append(
                        Order(sym, sell_quote, -min(passive_clip, sc))
                    )

    def trade_david_option(self, sym, depth, pos, out, edge_floor=14.0):
        b, a = self.best(depth)
        if b is None or a is None:
            return
        bv = depth.buy_orders[b]
        av = -depth.sell_orders[a]
        mid = (b + a) / 2
        spread = a - b
        key = f"oema_{sym}"
        prev = self.sd.get(key)
        fair_raw = mid if prev is None else prev
        fair = fair_raw - 0.005 * pos
        dynamic_edge = max(edge_floor, spread * 0.75)
        lim = POS_LIMIT[sym]
        bc = lim - pos
        sc = lim + pos
        if a <= fair - dynamic_edge:
            q = min(30, av, bc)
            if q > 0:
                out.setdefault(sym, []).append(Order(sym, a, q))
        elif b >= fair + dynamic_edge:
            q = min(30, bv, sc)
            if q > 0:
                out.setdefault(sym, []).append(Order(sym, b, -q))
        new_ema = fair_raw + 0.0003 * (mid - fair_raw)
        self.sd[key] = new_ema

    def solve3(self, A, B):
        M = [r[:] + [B[i]] for i, r in enumerate(A)]
        for i in range(3):
            mx = max(range(i, 3), key=lambda r: abs(M[r][i]))
            M[i], M[mx] = M[mx], M[i]
            p = M[i][i]
            if abs(p) < 1e-12:
                raise ValueError
            for r in range(i + 1, 3):
                f = M[r][i] / p
                for c in range(i, 4):
                    M[r][c] -= f * M[i][c]
        x = [0] * 3
        for i in range(2, -1, -1):
            x[i] = (M[i][3] - sum(M[i][c] * x[c] for c in range(i + 1, 3))) / M[i][i]
        return x

    def smile_sleeve(self, state, out):
        T = max(0.01, 5 - state.timestamp / 1_000_000)
        if "VELVETFRUIT_EXTRACT" not in state.order_depths:
            return
        ud = state.order_depths["VELVETFRUIT_EXTRACT"]
        ub, ua = self.best(ud)
        if ub is None or ua is None:
            return
        S = (ub + ua) / 2
        ivs = {}
        for K in FIT_STRIKES:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            d = state.order_depths[sym]
            b, a = self.best(d)
            if b is None or a is None:
                continue
            C = (b + a) / 2
            v = implied_vol(C, S, K, T)
            if v is None:
                continue
            m = math.log(K / S) / math.sqrt(T)
            ivs[K] = (m, v, C)
        if len(ivs) < 4:
            return
        ms = [x[0] for x in ivs.values()]
        vs = [x[1] for x in ivs.values()]
        n = len(ms)
        sm = sum(ms)
        sm2 = sum(m * m for m in ms)
        sm3 = sum(m**3 for m in ms)
        sm4 = sum(m**4 for m in ms)
        sv = sum(vs)
        smv = sum(m * v for m, v in zip(ms, vs))
        sm2v = sum(m * m * v for m, v in zip(ms, vs))
        try:
            a_c, b_c, c_c = self.solve3(
                [[sm4, sm3, sm2], [sm3, sm2, sm], [sm2, sm, n]], [sm2v, smv, sv]
            )
        except:
            return
        WARMUP = 200
        SPAN = 300
        alph = 2.0 / (SPAN + 1)
        for K in SMILE_STRIKES:
            if K not in ivs:
                continue
            sym = f"VEV_{K}"
            m, v, C_mid = ivs[K]
            fit_iv = a_c * m * m + b_c * m + c_c
            C_theo = bs_call(S, K, T, fit_iv)
            raw = C_mid - C_theo
            bk = f"b_{K}"
            vk = f"v_{K}"
            ck = f"c_{K}"
            pb = self.sd.get(bk, 0.0)
            bias = pb + alph * (raw - pb)
            self.sd[bk] = bias
            dev = raw - bias
            pv = self.sd.get(vk, 1.0)
            var = pv + alph * (dev * dev - pv)
            self.sd[vk] = var
            cnt = self.sd.get(ck, 0) + 1
            self.sd[ck] = cnt
            if cnt < WARMUP:
                continue
            FV = C_theo + bias
            std = math.sqrt(max(var, 1e-6))
            d = state.order_depths[sym]
            b, a = self.best(d)
            if b is None or a is None:
                continue
            pos = state.position.get(sym, 0)
            lim = POS_LIMIT[sym]
            edge = max(1, int(round(std * 1.5)))
            q_bid = int(round(FV - edge))
            q_ask = int(round(FV + edge))
            if q_bid > b:
                q_bid = b + 1 if b + 1 < q_ask else b
            if q_ask < a:
                q_ask = a - 1 if a - 1 > q_bid else a
            if q_bid >= FV:
                q_bid = int(math.floor(FV - 1))
            if q_ask <= FV:
                q_ask = int(math.ceil(FV + 1))
            bc = lim - pos
            sc = lim + pos
            sz = 10
            if q_bid < q_ask and q_bid > 0:
                if bc > 0:
                    out.setdefault(sym, []).append(Order(sym, q_bid, min(sz, bc)))
                if sc > 0:
                    out.setdefault(sym, []).append(Order(sym, q_ask, -min(sz, sc)))
            z = (C_mid - FV) / std
            if abs(z) > 3.5:
                if z > 0 and sc > 0:
                    q = min(5, sc, d.buy_orders.get(b, 0))
                    if q > 0:
                        out.setdefault(sym, []).append(Order(sym, b, -q))
                elif z < 0 and bc > 0:
                    q = min(5, bc, -d.sell_orders.get(a, 0))
                    if q > 0:
                        out.setdefault(sym, []).append(Order(sym, a, q))

    def run(self, state):
        if state.traderData:
            try:
                self.sd = json.loads(state.traderData)
            except:
                self.sd = {}
        else:
            self.sd = {}
        self._restore_v33_estimators()
        result = {}

        # Order matters: VFE first so vfe_anchor_current is cached for ITM options
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            result["VELVETFRUIT_EXTRACT"] = self.vfe_scaled(
                state.order_depths["VELVETFRUIT_EXTRACT"], pos
            )

        if "HYDROGEL_PACK" in state.order_depths:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self.hgp_scaled(
                state.order_depths["HYDROGEL_PACK"], pos
            )

        for sym in V33_PRODUCTS:
            if sym in state.order_depths:
                pos = state.position.get(sym, 0)
                orders = self.trade_v33_option(sym, state.order_depths[sym], pos)
                if orders:
                    result[sym] = orders

        self._store_v33_estimators()
        return result, 0, json.dumps(self.sd)
