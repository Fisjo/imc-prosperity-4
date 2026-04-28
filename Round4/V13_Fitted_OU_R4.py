"""V13_Fitted_OU_R4 — online OU strategy with liquid-IV EWMA option pricing.

The underlying and Hydrogel are modeled as OU processes with fixed means:

    Y_t = X_t - mu
    Y_{t+dt} = phi * Y_t + eps_t

phi is fit by exponentially weighted OLS through the origin. Kappa and
stationary sigma are then:

    kappa = -log(phi) / dt
    sigma_inf = std(eps) / sqrt(1 - phi^2)

The all-days Round 4 fitted values are only the warmup defaults. During live
trading the option edge is the online EWMA OU/Bachelier call fair minus market
price; no implied-vol lock is used.
"""

import math
import json

try:
    from datamodel import Order
except ModuleNotFoundError:
    from prosperity4bt.datamodel import Order

# ============================================================================
# Products and limits
# ============================================================================
UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"
AVAILABLE_OPTION_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
OPTION_STRIKES = {
    product: strike
    for product, strike in AVAILABLE_OPTION_STRIKES.items()
    if strike <= 5500
}
POSITION_LIMITS = {UNDERLYING: 200, HYDROGEL: 200}
for _p in OPTION_STRIKES:
    POSITION_LIMITS[_p] = 300

# ============================================================================
# Model parameters
# ============================================================================
FINAL_TIMESTAMP = 1_000_000.0
UNDERLYING_MU = 5250.0
HYDROGEL_MU = 10000.0

# Fitted on Round 4 days 1-3 with dt = 100 / 1_000_000.
UNDERLYING_PHI = 0.998064
UNDERLYING_KAPPA = 19.3741
UNDERLYING_STATIONARY_SIGMA = 18.2827
HYDROGEL_PHI = 0.998062
HYDROGEL_KAPPA = 19.4017
HYDROGEL_STATIONARY_SIGMA = 34.8349

# Online fixed-mean OU EWMA. Lambda is the decay in:
#   S_t = lambda * S_{t-1} + observation_t
# and q_t = lambda * q_{t-1} + (1-lambda) * residual_t^2.
OU_EWMA_LAMBDA = 0.999
OU_MIN_OBSERVATIONS = 50
OU_MIN_PHI = 1e-6
OU_MAX_PHI = 0.999999
OU_MIN_SXX = 1e-9
OU_MIN_DT = 1e-12

# Edges
UNDERLYING_EDGE = 14.0
HYDROGEL_EDGE = 12.0
OPTION_EDGE = 10.5
HIGH_GAMMA_EDGE = 2.5
PASSIVE_OPTION_EDGE = 0.0
PASSIVE_VFE_BID_EDGE = 4
PASSIVE_VFE_ASK_EDGE = 4
PASSIVE_HYDROGEL_EDGE = 5.5
PASSIVE_SIZE = 10
OPTION_POSITION_PENALTY = 0.0
HIGH_GAMMA_OPTIONS = {"VEV_5300", "VEV_5400", "VEV_5500"}

# ============================================================================
# Deep ITM module for VEV_4000 / VEV_4500
# ============================================================================
ITM_STRIKES = (4000, 4500)
ITM_TAKE_EDGE = 1
ITM_INV_SCALE = 19
ITM_PASSIVE_CAP = 150
ITM_PASSIVE_CLIP = 50

# ============================================================================
# Vulnerable-strike asymmetric buy edge
# ============================================================================
MARK22_VULNERABLE_STRIKES = {
    "VEV_5200", "VEV_5400", "VEV_5500",
}
OPTION_BUY_EDGE_REDUCTION = 0.0
OPTION_BUY_EDGE_FLOOR = 0.5

# Cross-sectional option-implied stationary sigma estimate.
LIQUID_IV_OPTIONS = ("VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500")
LIQUID_IV_EWMA_LAMBDA = 0.9996666666666667  # span ~= 3,000 observations
LIQUID_IV_TIME_RAMP_POWER = 0.5
LIQUID_IV_TIME_RAMP_MULT = 5.0
LIQUID_IV_MIN_SAMPLES = 3
LIQUID_IV_FLOOR = 50.0
LIQUID_IV_CAP = 350.0


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


def call_fair_with_params(spot, strike, ts, kappa, vol):
    tau = time_left(ts)
    rho = math.exp(-kappa * tau)
    tm = UNDERLYING_MU + (spot - UNDERLYING_MU) * rho
    tsd = vol * math.sqrt(max(0.0, 1.0 - rho * rho))
    if tsd < 1e-9:
        return max(tm - strike, 0.0)
    d = (tm - strike) / tsd
    return (tm - strike) * normal_cdf(d) + tsd * normal_pdf(d)


def implied_sigma_inf_bisect(market_price, spot, strike, ts, kappa, lo=1e-9, hi=50.0):
    tau = time_left(ts)
    rho = math.exp(-kappa * tau)
    terminal_mean = UNDERLYING_MU + (spot - UNDERLYING_MU) * rho
    terminal_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
    intrinsic = max(terminal_mean - strike, 0.0)
    if terminal_scale < 1e-12 or market_price <= intrinsic + 1e-6:
        return None

    while hi < 5_000.0 and call_fair_with_params(spot, strike, ts, kappa, hi) < market_price:
        hi *= 2.0
    if call_fair_with_params(spot, strike, ts, kappa, hi) < market_price:
        return None

    for _ in range(55):
        mid = (lo + hi) / 2.0
        if call_fair_with_params(spot, strike, ts, kappa, mid) < market_price:
            lo = mid
        else:
            hi = mid
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

    # ---- Passive quoting ----
    def add_passive_quotes_split(self, p, d, fair, pos, lim, bid_edge, ask_edge, orders):
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

    def trade_ou_split(self, p, d, pos, mean, edge, bid_pe, ask_pe, ts, kappa=None):
        spot = self.best_mid(d)
        if spot is None:
            return []
        lim = POSITION_LIMITS[p]
        if kappa is None:
            kappa = HYDROGEL_KAPPA if p == HYDROGEL else UNDERLYING_KAPPA
        fair = ou_terminal_mean(spot, mean, kappa, ts)
        orders = []
        sc = lim + pos
        if sc > 0:
            o = self.sell_available(p, d, fair + edge, sc)
            if o is not None:
                orders.append(o)
                pos += o.quantity
        bc = lim - pos
        if bc > 0:
            o = self.buy_available(p, d, fair - edge, bc)
            if o is not None:
                orders.append(o)
                pos += o.quantity
        self.add_passive_quotes_split(p, d, fair, pos, lim, bid_pe, ask_pe, orders)
        return orders

    # ---- Generic option trader ----
    def trade_option(self, p, d, pos, spot, ts, kappa, vol):
        lim = POSITION_LIMITS[p]
        fair = call_fair_with_params(spot, OPTION_STRIKES[p], ts, kappa, vol)
        fair -= OPTION_POSITION_PENALTY * pos / lim

        sell_edge = self.option_edge(p)
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

    # ---- Specialized deep ITM module for VEV_4000 / VEV_4500 ----
    def trade_itm_option(self, sym, depth, pos, strike, spot, ts, kappa,
                         take_edge=ITM_TAKE_EDGE, scale=ITM_INV_SCALE,
                         passive_cap=ITM_PASSIVE_CAP, passive_clip=ITM_PASSIVE_CLIP):
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return []
        if spot is None:
            fair = UNDERLYING_MU - strike
        else:
            terminal_mean = ou_terminal_mean(spot, UNDERLYING_MU, kappa, ts)
            fair = max(terminal_mean - strike, 0.0)
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

    def default_ou_state(self, default_phi, default_kappa, default_sigma):
        return {
            "prev_spot": None,
            "prev_ts": None,
            "n": 0,
            "sxx": 0.0,
            "sxy": 0.0,
            "q": default_sigma * default_sigma * max(0.0, 1.0 - default_phi * default_phi),
            "phi": default_phi,
            "kappa": default_kappa,
            "sigma_inf": default_sigma,
        }

    def update_ou_state(self, ss, key, spot, mean, default_phi, default_kappa, default_sigma, ts):
        payload = ss.get(key)
        if not isinstance(payload, dict):
            payload = self.default_ou_state(default_phi, default_kappa, default_sigma)

        phi = float(payload.get("phi", default_phi))
        kappa = float(payload.get("kappa", default_kappa))
        sigma_inf = float(payload.get("sigma_inf", default_sigma))
        prev_spot = payload.get("prev_spot")
        prev_ts = payload.get("prev_ts")

        if spot is not None and prev_spot is not None and prev_ts is not None:
            dt = (float(ts) - float(prev_ts)) / FINAL_TIMESTAMP
            if dt > OU_MIN_DT:
                y0 = float(prev_spot) - mean
                y1 = float(spot) - mean
                sxx = OU_EWMA_LAMBDA * float(payload.get("sxx", 0.0)) + y0 * y0
                sxy = OU_EWMA_LAMBDA * float(payload.get("sxy", 0.0)) + y0 * y1
                n = int(payload.get("n", 0)) + 1

                if sxx > OU_MIN_SXX:
                    phi = min(max(sxy / sxx, OU_MIN_PHI), OU_MAX_PHI)
                    residual = y1 - phi * y0
                    q = (
                        OU_EWMA_LAMBDA * float(payload.get("q", 0.0))
                        + (1.0 - OU_EWMA_LAMBDA) * residual * residual
                    )
                    if n >= OU_MIN_OBSERVATIONS:
                        kappa = -math.log(phi) / dt
                        sigma_inf = math.sqrt(q / max(1e-12, 1.0 - phi * phi))
                    payload.update({
                        "n": n,
                        "sxx": sxx,
                        "sxy": sxy,
                        "q": q,
                        "phi": phi,
                        "kappa": kappa,
                        "sigma_inf": sigma_inf,
                    })

        if spot is not None:
            payload["prev_spot"] = float(spot)
            payload["prev_ts"] = float(ts)

        ss[key] = payload
        return ss, kappa, sigma_inf

    def liquid_iv_mean(self, state, spot, kappa):
        if spot is None:
            return None

        samples = []
        for product in LIQUID_IV_OPTIONS:
            depth = state.order_depths.get(product)
            option_mid = self.best_mid(depth) if depth is not None else None
            if option_mid is None:
                continue
            sigma = implied_sigma_inf_bisect(
                float(option_mid),
                float(spot),
                AVAILABLE_OPTION_STRIKES[product],
                state.timestamp,
                kappa,
            )
            if sigma is not None and math.isfinite(sigma):
                samples.append(max(LIQUID_IV_FLOOR, min(LIQUID_IV_CAP, sigma)))

        if len(samples) < LIQUID_IV_MIN_SAMPLES:
            return None
        return sum(samples) / len(samples)

    def update_liquid_iv_state(self, ss, state, spot, kappa, underlying_vol_signal):
        prev = float(ss.get("liquid_iv_ewma", underlying_vol_signal))
        instant = self.liquid_iv_mean(state, spot, kappa)
        if instant is None:
            ss["liquid_iv_ewma"] = prev
            return ss, prev

        elapsed = 1.0 - time_left(state.timestamp)
        ramp = min(
            1.0,
            LIQUID_IV_TIME_RAMP_MULT * math.pow(max(0.0, elapsed), LIQUID_IV_TIME_RAMP_POWER),
        )
        alpha = (1.0 - LIQUID_IV_EWMA_LAMBDA) * ramp
        estimate = prev + alpha * (instant - prev)
        estimate = max(LIQUID_IV_FLOOR, min(LIQUID_IV_CAP, estimate))
        ss["liquid_iv_ewma"] = estimate
        ss["liquid_iv_instant"] = instant
        ss["liquid_iv_alpha"] = alpha
        ss["liquid_iv_ramp"] = ramp
        ss["underlying_vol_signal"] = underlying_vol_signal
        return ss, estimate

    # ---- Main loop ----
    def run(self, state):
        try:
            ss = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ss = {}

        result = {}
        vvf_depth = state.order_depths.get(UNDERLYING)
        vvf_spot = self.best_mid(vvf_depth)
        hydrogel_depth = state.order_depths.get(HYDROGEL)
        hydrogel_spot = self.best_mid(hydrogel_depth)

        ss, cur_kappa, cur_vol = self.update_ou_state(
            ss, "underlying_ou", vvf_spot, UNDERLYING_MU,
            UNDERLYING_PHI, UNDERLYING_KAPPA, UNDERLYING_STATIONARY_SIGMA,
            state.timestamp,
        )
        ss, hydrogel_kappa, _ = self.update_ou_state(
            ss, "hydrogel_ou", hydrogel_spot, HYDROGEL_MU,
            HYDROGEL_PHI, HYDROGEL_KAPPA, HYDROGEL_STATIONARY_SIGMA,
            state.timestamp,
        )
        ss, cur_vol = self.update_liquid_iv_state(ss, state, vvf_spot, cur_kappa, cur_vol)

        # --- VVF ---
        if UNDERLYING in state.order_depths:
            position = state.position.get(UNDERLYING, 0)
            orders = self.trade_ou_split(
                UNDERLYING, state.order_depths[UNDERLYING], position,
                UNDERLYING_MU, UNDERLYING_EDGE,
                PASSIVE_VFE_BID_EDGE, PASSIVE_VFE_ASK_EDGE,
                state.timestamp, UNDERLYING_KAPPA,
            )
            if orders:
                result[UNDERLYING] = orders

        # --- Hydrogel ---
        if HYDROGEL in state.order_depths:
            position = state.position.get(HYDROGEL, 0)
            orders = self.trade_ou_split(
                HYDROGEL, state.order_depths[HYDROGEL], position,
                HYDROGEL_MU, HYDROGEL_EDGE,
                PASSIVE_HYDROGEL_EDGE, PASSIVE_HYDROGEL_EDGE,
                state.timestamp,
                kappa=hydrogel_kappa,
            )
            if orders:
                result[HYDROGEL] = orders

        # --- All options (generic logic; ITM strikes overridden below) ---
        if vvf_spot is not None:
            for product in OPTION_STRIKES:
                depth = state.order_depths.get(product)
                if depth is None:
                    continue
                position = state.position.get(product, 0)
                orders = self.trade_option(product, depth, position, vvf_spot,
                                            state.timestamp, cur_kappa, cur_vol)
                if orders:
                    result[product] = orders

        # --- Override ITM strikes with specialized logic ---
        for strike in ITM_STRIKES:
            sym = f"VEV_{strike}"
            result.pop(sym, None)
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            position = state.position.get(sym, 0)
            orders = self.trade_itm_option(sym, depth, position, strike, vvf_spot,
                                           state.timestamp, cur_kappa)
            if orders:
                result[sym] = orders

        return result, 0, json.dumps(ss)
