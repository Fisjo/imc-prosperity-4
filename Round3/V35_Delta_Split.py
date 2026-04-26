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

ANCHOR_WINDOW = 2000
ANCHOR_WARMUP = 200

DAVID_EDGE_FLOORS = {5500: 1.0, 5400: 2.0, 5300: 5.0, 5200: 8.0}


class Trader:
    def __init__(self):
        self.sd = {}
        self.delta_sd = {}
        self.options_sd = {}

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

    def online_anchor(self, key, sample):
        """Update rolling history and return current median, or None if warming up.
        State stored in self.delta_sd[f'{key}_hist'] as a list of floats."""
        hist_key = f"{key}_hist"
        hist = self.delta_sd.get(hist_key) or []
        hist.append(sample)
        if len(hist) > ANCHOR_WINDOW:
            hist = hist[-ANCHOR_WINDOW:]
        self.delta_sd[hist_key] = hist
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

        anchor_f = self.online_anchor("vfe_anchor", wm)
        if anchor_f is None:
            return []
        VFE_ANCHOR = int(round(anchor_f))
        self.delta_sd["vfe_anchor_current"] = VFE_ANCHOR

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

        VFE_EMA_ALPHA = 0.005
        VFE_EMA_TE = 14
        mid_for_ema = (b + a) / 2
        prev_ema = self.delta_sd.get("vfe_ema_seeded")
        if prev_ema is None:
            prev_ema = float(VFE_ANCHOR)
        ema = prev_ema + VFE_EMA_ALPHA * (mid_for_ema - prev_ema)
        self.delta_sd["vfe_ema_seeded"] = ema
        if a <= ema - VFE_EMA_TE and bc > 0:
            avail2 = abs(sum(v for p, v in depth.sell_orders.items() if p <= ema - VFE_EMA_TE))
            q = min(bc, avail2)
            if q > 0:
                orders.append(Order(sym, a, q))
                bc -= q
        if b >= ema + VFE_EMA_TE and sc > 0:
            avail2 = sum(v for p, v in depth.buy_orders.items() if p >= ema + VFE_EMA_TE)
            q = min(sc, avail2)
            if q > 0:
                orders.append(Order(sym, b, -q))
                sc -= q

        if abs(pos) < 150:
            prev_fv = self.delta_sd.get("vfe_local_fv")
            local_fv = wm if prev_fv is None else prev_fv + 0.2 * (wm - prev_fv)
            self.delta_sd["vfe_local_fv"] = local_fv
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

        fair_f = self.online_anchor("hgp_anchor", wm)
        if fair_f is None:
            return []
        FAIR = int(round(fair_f))

        TAKE_EDGE = 36
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
        sell_quote = max(best_ask - 1, int(sell_base) if sell_base == int(sell_base) else int(sell_base) + 1)

        if bc_left > 0 and buy_quote < best_ask and buy_quote < sell_quote:
            quantity = min(PASSIVE_CLIP, bc_left)
            if quantity > 0:
                orders.append(Order(sym, buy_quote, quantity))

        if sc_left > 0 and sell_quote > best_bid and sell_quote > buy_quote:
            quantity = min(PASSIVE_CLIP, sc_left)
            if quantity > 0:
                orders.append(Order(sym, sell_quote, -quantity))

        return orders

    def trade_itm_option(self, sym, depth, pos, out, K, current_vfe_anchor, take_edge=1, scale=19, passive_clip=50):
        """ITM (deep) options. FV = E[S] - K, where E[S] is the ONLINE VFE anchor.
        Replaces v25's hardcoded ITM_STRUCTURAL_FV = 5250 - K."""
        if not depth.buy_orders or not depth.sell_orders:
            return
        if current_vfe_anchor is None:
            return
        fair = current_vfe_anchor - K

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
                    out.setdefault(sym, []).append(Order(sym, buy_quote, min(passive_clip, bc)))
                if sc > 0:
                    out.setdefault(sym, []).append(Order(sym, sell_quote, -min(passive_clip, sc)))

    def trade_david_option(self, sym, depth, pos, out, edge_floor=14.0):
        b, a = self.best(depth)
        if b is None or a is None:
            return
        bv = depth.buy_orders[b]
        av = -depth.sell_orders[a]
        mid = (b + a) / 2
        spread = a - b
        key = f"oema_{sym}"
        prev = self.options_sd.get(key)
        fair_raw = mid if prev is None else prev
        fair = fair_raw - 0.005 * pos
        dynamic_edge = max(edge_floor, spread * 1.5)
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
        self.options_sd[key] = new_ema

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
        sm3 = sum(m ** 3 for m in ms)
        sm4 = sum(m ** 4 for m in ms)
        sv = sum(vs)
        smv = sum(m * v for m, v in zip(ms, vs))
        sm2v = sum(m * m * v for m, v in zip(ms, vs))
        try:
            a_c, b_c, c_c = self.solve3([[sm4, sm3, sm2], [sm3, sm2, sm], [sm2, sm, n]], [sm2v, smv, sv])
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
            pb = self.options_sd.get(bk, 0.0)
            bias = pb + alph * (raw - pb)
            self.options_sd[bk] = bias
            dev = raw - bias
            pv = self.options_sd.get(vk, 1.0)
            var = pv + alph * (dev * dev - pv)
            self.options_sd[vk] = var
            cnt = self.options_sd.get(ck, 0) + 1
            self.options_sd[ck] = cnt
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
                loaded = json.loads(state.traderData)
                self.sd = loaded if isinstance(loaded, dict) else {}
            except:
                self.sd = {}
        else:
            self.sd = {}

        delta_sd = self.sd.get("delta1")
        self.delta_sd = delta_sd if isinstance(delta_sd, dict) else {}
        self.sd["delta1"] = self.delta_sd

        options_sd = self.sd.get("options")
        self.options_sd = options_sd if isinstance(options_sd, dict) else {}
        self.sd["options"] = self.options_sd

        result = {}

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            result["VELVETFRUIT_EXTRACT"] = self.vfe_scaled(state.order_depths["VELVETFRUIT_EXTRACT"], pos)

        if "HYDROGEL_PACK" in state.order_depths:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self.hgp_scaled(state.order_depths["HYDROGEL_PACK"], pos)

        current_vfe_anchor = self.delta_sd.get("vfe_anchor_current")
        for K in DAVID_STRIKES:
            sym = f"VEV_{K}"
            if sym in state.order_depths:
                pos = state.position.get(sym, 0)
                if K in ITM_STRIKES:
                    self.trade_itm_option(
                        sym,
                        state.order_depths[sym],
                        pos,
                        result,
                        K=K,
                        current_vfe_anchor=current_vfe_anchor,
                    )
                else:
                    edge_floor = DAVID_EDGE_FLOORS.get(K, 14.0)
                    self.trade_david_option(sym, state.order_depths[sym], pos, result, edge_floor=edge_floor)

        self.smile_sleeve(state, result)
        return result, 0, json.dumps(self.sd)
