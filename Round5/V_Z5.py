"""V46_runtime: V42b ported to runtime-incremental form.

Self-contained, IMC-deployable. NO file I/O, NO CSV dependency, NO forbidden imports.
All V42b plan layers ported to incremental updates maintained in instance state:

  Layer 1: Per-product BETA fair value (blend-style, microprice + EWMA premium)
  Layer 2: Within-basket OVR residual z-score skew (blend-style)
  Layer 3: Pair OLS within products (1225 pairs, V42b's pair_skews ported)
  Layer 4: Basket OLS across baskets (45 pairs, V42b's basket layer ported)
  Layer 5: Cross-sectional rank-and-fade (V42b's cross_sec layer)
  Layer 6: DIRECTIONAL_SKEW for under-utilized consistent-direction products
  Layer 7: V5_DIRECTIONAL_TARGET for V5-routed directional products

Reservation = BETA_fair + sum(skews) - INVENTORY_LAMBDA * (position - target)
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from datamodel import Order, OrderDepth, TradingState
except ModuleNotFoundError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState


# ============================================================================
# Hyperparameters (matched to V42b/V26)
# ============================================================================
ASSET_LIMIT = 10
EWMA_ALPHA = 2.0 / 51.0
WINDOW = 10                   # OLS rolling window
ZSCORE_WINDOW = 100           # Residual z-score window
SIGNAL_THRESHOLD = 2.0
FAIR_ADJUSTMENT_SCALE = 1.0
INVENTORY_TARGET_SCALE = 3.0
INVENTORY_LAMBDA = 0.02
CROSS_SEC_HORIZON = 1500
CROSS_SEC_SCALE = 8.0
BETA_CLIP = 3.0
PASSIVE_EDGE = 0.0
TAKE_EDGE = 0.0
TAKE_SIZE = ASSET_LIMIT
EPS = 1e-12

OVR_ALPHA = 2.0 / 101.0
OVR_WARMUP = 100
OVR_Z_CLIP = 3.0
OVR_SCALE = 0.20

# Per-product BETAs for fair value (refit on D2+D3+D4 OLS, h=10)
BETAS: Dict[str, Tuple[float, float]] = {
    "GALAXY_SOUNDS_BLACK_HOLES": (1.125243, -0.017566),
    "GALAXY_SOUNDS_DARK_MATTER": (1.394414, -0.002902),
    "GALAXY_SOUNDS_PLANETARY_RINGS": (1.533440, -0.016176),
    "GALAXY_SOUNDS_SOLAR_FLAMES": (1.302039, 0.006514),
    "GALAXY_SOUNDS_SOLAR_WINDS": (1.400152, -0.014767),
    "MICROCHIP_CIRCLE": (1.895228, 0.004608),
    "MICROCHIP_OVAL": (1.983199, 0.019187),
    "MICROCHIP_RECTANGLE": (1.924913, 0.021051),
    "MICROCHIP_SQUARE": (1.172801, 0.002869),
    "MICROCHIP_TRIANGLE": (3.028940, -0.011032),
    "OXYGEN_SHAKE_CHOCOLATE": (1.428728, 0.021268),
    "OXYGEN_SHAKE_EVENING_BREATH": (1.168872, 0.043830),
    "OXYGEN_SHAKE_GARLIC": (1.376992, 0.005488),
    "OXYGEN_SHAKE_MINT": (1.318870, -0.018996),
    "OXYGEN_SHAKE_MORNING_BREATH": (0.478522, 0.024881),
    "PANEL_1X2": (1.357534, 0.002436),
    "PANEL_1X4": (1.406866, -0.026674),
    "PANEL_2X2": (1.262702, -0.002675),
    "PANEL_2X4": (0.155920, -0.004777),
    "PANEL_4X4": (0.311220, -0.002746),
    "PEBBLES_L": (1.099572, 0.020536),
    "PEBBLES_M": (0.991396, 0.013322),
    "PEBBLES_S": (1.854917, -0.010808),
    "PEBBLES_XL": (2.212510, 0.024916),
    "PEBBLES_XS": (0.264727, 0.033415),
    "ROBOT_DISHES": (2.154209, 0.129223),
    "ROBOT_IRONING": (0.935951, 0.017878),
    "ROBOT_LAUNDRY": (0.458141, -0.002365),
    "ROBOT_MOPPING": (-0.083955, 0.006858),
    "ROBOT_VACUUMING": (1.916372, 0.010079),
    "SLEEP_POD_COTTON": (1.576693, -0.019890),
    "SLEEP_POD_LAMB_WOOL": (1.867661, -0.025790),
    "SLEEP_POD_NYLON": (1.988513, 0.004157),
    "SLEEP_POD_POLYESTER": (0.792378, 0.011606),
    "SLEEP_POD_SUEDE": (1.945089, 0.015396),
    "SNACKPACK_CHOCOLATE": (1.216656, -0.005573),
    "SNACKPACK_PISTACHIO": (1.041262, 0.027103),
    "SNACKPACK_RASPBERRY": (1.276660, 0.034544),
    "SNACKPACK_STRAWBERRY": (1.028173, 0.032446),
    "SNACKPACK_VANILLA": (0.977844, -0.012481),
    "TRANSLATOR_ASTRO_BLACK": (1.349303, 0.007265),
    "TRANSLATOR_ECLIPSE_CHARCOAL": (2.146607, 0.012637),
    "TRANSLATOR_GRAPHITE_MIST": (1.409747, 0.008871),
    "TRANSLATOR_SPACE_GRAY": (0.624512, -0.001765),
    "TRANSLATOR_VOID_BLUE": (1.615479, 0.034277),
    "UV_VISOR_AMBER": (1.267742, -0.013046),
    "UV_VISOR_MAGENTA": (1.604935, 0.034810),
    "UV_VISOR_ORANGE": (1.181909, -0.010343),
    "UV_VISOR_RED": (1.361133, -0.006547),
    "UV_VISOR_YELLOW": (1.451859, 0.021823),
}

# Baskets (10 groups of 5)
BASKETS: Dict[str, Tuple[str, ...]] = {
    "GALAXY_SOUNDS": ("GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
                      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
                      "GALAXY_SOUNDS_SOLAR_WINDS"),
    "MICROCHIP": ("MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE",
                  "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE"),
    "OXYGEN_SHAKE": ("OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH",
                     "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT",
                     "OXYGEN_SHAKE_MORNING_BREATH"),
    "PANEL": ("PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4"),
    "PEBBLES": ("PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS"),
    "ROBOT": ("ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY", "ROBOT_MOPPING", "ROBOT_VACUUMING"),
    "SLEEP_POD": ("SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON",
                  "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"),
    "SNACKPACK": ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",
                  "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA"),
    "TRANSLATOR": ("TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
                   "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE"),
    "UV_VISOR": ("UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE",
                 "UV_VISOR_RED", "UV_VISOR_YELLOW"),
}

# OVR scales (within-basket one-vs-rest skew). Tested: only MICROCHIP+SNACKPACK help.
OVR_SCALES: Dict[str, float] = {"MICROCHIP": 1.0, "SNACKPACK": 2.0}

# Directional skew for under-utilized consistent-direction products (V41/V42 finding).
# Strong skew (±100) overrides MM, drives position to ±10 in the consistent-direction.
DIRECTIONAL_SKEW: Dict[str, float] = {
    # 3/3 days same direction, V42b-routed but underutilized
    "PEBBLES_XS": -100.0,
    "GALAXY_SOUNDS_BLACK_HOLES": +100.0,
    "PANEL_2X4": +100.0,
    # V5-routed in V42b — applying directional skew here for the same effect
    "MICROCHIP_OVAL": -100.0,
    "UV_VISOR_AMBER": -100.0,
    "OXYGEN_SHAKE_GARLIC": +100.0,
    "ROBOT_IRONING": -100.0,
}


# ============================================================================
# Universe setup (computed once at module load)
# ============================================================================
ALL_PRODUCTS: Tuple[str, ...] = tuple(sorted(BETAS.keys()))
N_PRODUCTS = len(ALL_PRODUCTS)
PRODUCT_INDEX: Dict[str, int] = {p: i for i, p in enumerate(ALL_PRODUCTS)}

# All C(50, 2) pairs (i, j) with i < j
_PAIR_X: List[int] = []
_PAIR_Y: List[int] = []
for _i in range(N_PRODUCTS):
    for _j in range(_i + 1, N_PRODUCTS):
        _PAIR_X.append(_i)
        _PAIR_Y.append(_j)
PAIR_X_IDX = np.array(_PAIR_X, dtype=np.int64)
PAIR_Y_IDX = np.array(_PAIR_Y, dtype=np.int64)
N_PAIRS = len(_PAIR_X)

# Basket setup (10 baskets, 45 pairs)
BASKET_NAMES: Tuple[str, ...] = tuple(sorted(BASKETS.keys()))
N_BASKETS = len(BASKET_NAMES)
BASKET_INDEX: Dict[str, int] = {b: i for i, b in enumerate(BASKET_NAMES)}

# product_to_basket[i] = basket_index of product i
PRODUCT_TO_BASKET = np.full(N_PRODUCTS, -1, dtype=np.int64)
for _bi, _bname in enumerate(BASKET_NAMES):
    for _p in BASKETS[_bname]:
        if _p in PRODUCT_INDEX:
            PRODUCT_TO_BASKET[PRODUCT_INDEX[_p]] = _bi

# basket_member_idx[bi] = list of product indices in basket bi
BASKET_MEMBER_IDX: List[np.ndarray] = []
for _bname in BASKET_NAMES:
    _members = [PRODUCT_INDEX[p] for p in BASKETS[_bname] if p in PRODUCT_INDEX]
    BASKET_MEMBER_IDX.append(np.array(_members, dtype=np.int64))

_B_X: List[int] = []
_B_Y: List[int] = []
for _i in range(N_BASKETS):
    for _j in range(_i + 1, N_BASKETS):
        _B_X.append(_i)
        _B_Y.append(_j)
BASKET_X_IDX = np.array(_B_X, dtype=np.int64)
BASKET_Y_IDX = np.array(_B_Y, dtype=np.int64)
N_BASKET_PAIRS = len(_B_X)

# OVR keys: list of (basket_index, product_index_in_basket, product_index_global)
_OVR_BASKET_IDX: List[int] = []
_OVR_PRODUCT_IDX: List[int] = []
_OVR_REST_IDX: List[List[int]] = []
_OVR_SCALE_VEC: List[float] = []
for _basket_name, _scale in OVR_SCALES.items():
    _bi = BASKET_INDEX[_basket_name]
    _members = list(BASKETS[_basket_name])
    for _focal in _members:
        _OVR_BASKET_IDX.append(_bi)
        _OVR_PRODUCT_IDX.append(PRODUCT_INDEX[_focal])
        _rest = [PRODUCT_INDEX[m] for m in _members if m != _focal]
        _OVR_REST_IDX.append(_rest)
        _OVR_SCALE_VEC.append(_scale)
N_OVR = len(_OVR_PRODUCT_IDX)
OVR_BASKET_IDX = np.array(_OVR_BASKET_IDX, dtype=np.int64) if N_OVR > 0 else np.zeros(0, dtype=np.int64)
OVR_PRODUCT_IDX = np.array(_OVR_PRODUCT_IDX, dtype=np.int64) if N_OVR > 0 else np.zeros(0, dtype=np.int64)
OVR_SCALE_VEC = np.array(_OVR_SCALE_VEC, dtype=np.float64) if N_OVR > 0 else np.zeros(0, dtype=np.float64)
# OVR rest indices: array shape (N_OVR, 4)
OVR_REST_IDX = np.array(_OVR_REST_IDX, dtype=np.int64) if N_OVR > 0 else np.zeros((0, 4), dtype=np.int64)


# ============================================================================
# Helpers
# ============================================================================
def _best_bid(depth: OrderDepth) -> Optional[int]:
    prices = [p for p, v in depth.buy_orders.items() if v > 0]
    return max(prices) if prices else None


def _best_ask(depth: OrderDepth) -> Optional[int]:
    prices = [p for p, v in depth.sell_orders.items() if v < 0]
    return min(prices) if prices else None


def _midprice(depth: OrderDepth) -> Optional[float]:
    bb = _best_bid(depth)
    ba = _best_ask(depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


def _microprice(depth: OrderDepth) -> Optional[float]:
    bb = _best_bid(depth)
    ba = _best_ask(depth)
    if bb is None or ba is None:
        return None
    bid_notional = 0.0
    bid_volume = 0
    for price, vol in depth.buy_orders.items():
        if vol > 0:
            bid_notional += float(price * vol)
            bid_volume += int(vol)
    ask_notional = 0.0
    ask_volume = 0
    for price, vol in depth.sell_orders.items():
        if vol < 0:
            size = abs(int(vol))
            ask_notional += float(price * size)
            ask_volume += size
    total = bid_volume + ask_volume
    if total <= 0:
        return (bb + ba) / 2.0
    bid_vwap = bid_notional / bid_volume if bid_volume > 0 else float(bb)
    ask_vwap = ask_notional / ask_volume if ask_volume > 0 else float(ba)
    return (bid_vwap * ask_volume + ask_vwap * bid_volume) / total


def _orders_for_product(
    *,
    product: str,
    depth: OrderDepth,
    position: int,
    reservation: float,
    limit: int,
    passive_edge: float,
    take_edge: float,
    take_size: int,
) -> List[Order]:
    bb = _best_bid(depth)
    ba = _best_ask(depth)
    if bb is None or ba is None:
        return []

    orders: List[Order] = []
    working_pos = position

    buy_capacity = min(take_size, max(0, limit - working_pos))
    for ask_price in sorted(depth.sell_orders):
        if buy_capacity <= 0 or reservation - ask_price <= take_edge:
            break
        qty = min(abs(int(depth.sell_orders[ask_price])), buy_capacity)
        if qty > 0:
            orders.append(Order(product, int(ask_price), qty))
            working_pos += qty
            buy_capacity -= qty

    sell_capacity = min(take_size, max(0, limit + working_pos))
    for bid_price in sorted(depth.buy_orders, reverse=True):
        if sell_capacity <= 0 or bid_price - reservation <= take_edge:
            break
        qty = min(int(depth.buy_orders[bid_price]), sell_capacity)
        if qty > 0:
            orders.append(Order(product, int(bid_price), -qty))
            working_pos -= qty
            sell_capacity -= qty

    pb_cap = max(0, limit - working_pos)
    pb = bb + 1 if bb + 1 < ba else bb
    if pb_cap > 0 and reservation - pb > passive_edge:
        orders.append(Order(product, int(pb), pb_cap))

    ps_cap = max(0, limit + working_pos)
    pa = ba - 1 if ba - 1 > bb else ba
    if ps_cap > 0 and pa - reservation > passive_edge:
        orders.append(Order(product, int(pa), -ps_cap))

    return orders


# ============================================================================
# Trader
# ============================================================================
class Trader:
    def __init__(self) -> None:
        self.tick = 0

        # Per-product log mid history (circular buffer for cross-sec lag)
        self.cross_sec_buf = np.zeros((N_PRODUCTS, CROSS_SEC_HORIZON), dtype=np.float64)
        self.cross_sec_n = 0  # number of ticks observed (clamped to CROSS_SEC_HORIZON)

        # EWMA of microprice (per product, for blend BETA fair)
        self.ewma = np.full(N_PRODUCTS, np.nan, dtype=np.float64)

        # Pair OLS state (vectorized)
        # Buffers: (N_PAIRS, WINDOW)
        self.pair_x_buf = np.zeros((N_PAIRS, WINDOW), dtype=np.float64)
        self.pair_y_buf = np.zeros((N_PAIRS, WINDOW), dtype=np.float64)
        self.pair_n = 0  # ticks observed (same for all pairs)
        # Sliding sums
        self.pair_sx = np.zeros(N_PAIRS, dtype=np.float64)
        self.pair_sy = np.zeros(N_PAIRS, dtype=np.float64)
        self.pair_sxx = np.zeros(N_PAIRS, dtype=np.float64)
        self.pair_sxy = np.zeros(N_PAIRS, dtype=np.float64)
        # Residual rolling stats
        self.pair_res_buf = np.zeros((N_PAIRS, ZSCORE_WINDOW), dtype=np.float64)
        self.pair_res_n = 0
        self.pair_res_sum = np.zeros(N_PAIRS, dtype=np.float64)
        self.pair_res_sum2 = np.zeros(N_PAIRS, dtype=np.float64)

        # Basket OLS state (smaller, same shape)
        self.basket_x_buf = np.zeros((N_BASKET_PAIRS, WINDOW), dtype=np.float64)
        self.basket_y_buf = np.zeros((N_BASKET_PAIRS, WINDOW), dtype=np.float64)
        self.basket_n = 0
        self.basket_sx = np.zeros(N_BASKET_PAIRS, dtype=np.float64)
        self.basket_sy = np.zeros(N_BASKET_PAIRS, dtype=np.float64)
        self.basket_sxx = np.zeros(N_BASKET_PAIRS, dtype=np.float64)
        self.basket_sxy = np.zeros(N_BASKET_PAIRS, dtype=np.float64)
        self.basket_res_buf = np.zeros((N_BASKET_PAIRS, ZSCORE_WINDOW), dtype=np.float64)
        self.basket_res_n = 0
        self.basket_res_sum = np.zeros(N_BASKET_PAIRS, dtype=np.float64)
        self.basket_res_sum2 = np.zeros(N_BASKET_PAIRS, dtype=np.float64)

        # OVR state (per (basket, focal_product)): EWMA mean and variance of residual
        self.ovr_means = np.zeros(N_OVR, dtype=np.float64)
        self.ovr_vars = np.zeros(N_OVR, dtype=np.float64)
        self.ovr_n = 0

    # -- per-tick state updates ---------------------------------------------
    def _update_pair_ols(self, log_mids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized rolling OLS update over all 1225 pairs.
        Returns (alpha, beta, residual, valid) arrays of shape (N_PAIRS,).
        """
        x_vals = log_mids[PAIR_X_IDX]  # current "x" (universe.x) per pair
        y_vals = log_mids[PAIR_Y_IDX]  # current "y" (universe.y) per pair

        # NOTE: V26's _rolling_ols swaps x/y for OLS. The OLS regresses x on y.
        # So in OLS terms: regression Y = α + β*X where Y=x_vals (universe.x), X=y_vals (universe.y).
        # We maintain sums of "X" (=y_vals) and "Y" (=x_vals).
        # To keep names sane internally: let X = y_vals, Y = x_vals.
        ols_X = y_vals
        ols_Y = x_vals

        write_idx = self.tick % WINDOW
        # Subtract old (only after warmup)
        if self.pair_n >= WINDOW:
            old_X = self.pair_x_buf[:, write_idx]  # we store ols_X in pair_x_buf
            old_Y = self.pair_y_buf[:, write_idx]  # we store ols_Y in pair_y_buf
            self.pair_sx -= old_X
            self.pair_sy -= old_Y
            self.pair_sxx -= old_X * old_X
            self.pair_sxy -= old_X * old_Y
        # Write new
        self.pair_x_buf[:, write_idx] = ols_X
        self.pair_y_buf[:, write_idx] = ols_Y
        self.pair_sx += ols_X
        self.pair_sy += ols_Y
        self.pair_sxx += ols_X * ols_X
        self.pair_sxy += ols_X * ols_Y
        self.pair_n = min(self.pair_n + 1, WINDOW)

        n = float(self.pair_n)
        if self.pair_n < 2:
            return np.zeros(N_PAIRS), np.full(N_PAIRS, np.nan), np.zeros(N_PAIRS), np.zeros(N_PAIRS, dtype=bool)
        mx = self.pair_sx / n
        my = self.pair_sy / n
        var_x = self.pair_sxx / n - mx * mx
        cov_xy = self.pair_sxy / n - mx * my
        valid = var_x > EPS
        beta = np.where(valid, np.divide(cov_xy, np.where(valid, var_x, 1.0), out=np.zeros(N_PAIRS), where=valid), np.nan)
        alpha = np.where(valid, my - beta * mx, np.nan)

        # Residual for current observation
        residual = ols_Y - alpha - beta * ols_X
        residual = np.where(np.isfinite(residual), residual, 0.0)

        return alpha, beta, residual, valid

    def _update_pair_residual_zscore(self, residual: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update residual rolling stats and compute z-score per pair."""
        write_idx = self.tick % ZSCORE_WINDOW
        if self.pair_res_n >= ZSCORE_WINDOW:
            old = self.pair_res_buf[:, write_idx]
            self.pair_res_sum -= old
            self.pair_res_sum2 -= old * old
        self.pair_res_buf[:, write_idx] = residual
        self.pair_res_sum += residual
        self.pair_res_sum2 += residual * residual
        self.pair_res_n = min(self.pair_res_n + 1, ZSCORE_WINDOW)

        n = float(self.pair_res_n)
        if self.pair_res_n < 2:
            return np.zeros(N_PAIRS), np.zeros(N_PAIRS), np.zeros(N_PAIRS, dtype=bool)
        mean = self.pair_res_sum / n
        var = np.maximum(self.pair_res_sum2 / n - mean * mean, 0.0)
        std = np.sqrt(var)
        z = np.where(std > EPS, (residual - mean) / np.where(std > EPS, std, 1.0), 0.0)
        z_valid = (std > EPS) & valid & (self.pair_res_n >= ZSCORE_WINDOW)
        return mean, std, z_valid

    def _pair_pressure(self, residual: np.ndarray, alpha: np.ndarray, beta: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-product pair_skew and pair_target from current pair OLS state."""
        # Compute z
        n = float(self.pair_res_n)
        if self.pair_res_n < 2:
            return np.zeros(N_PRODUCTS), np.zeros(N_PRODUCTS)
        mean = self.pair_res_sum / n
        var = np.maximum(self.pair_res_sum2 / n - mean * mean, 0.0)
        std = np.sqrt(var)
        z_signal = np.where(std > EPS, (residual - mean) / np.where(std > EPS, std, 1.0), 0.0)
        excess = np.sign(z_signal) * np.maximum(np.abs(z_signal) - SIGNAL_THRESHOLD, 0.0)

        beta_clipped = np.clip(np.where(np.isfinite(beta), beta, 0.0), -BETA_CLIP, BETA_CLIP)
        active = (
            np.isfinite(z_signal)
            & np.isfinite(beta_clipped)
            & (excess != 0.0)
            & valid
            & (self.pair_res_n >= ZSCORE_WINDOW)
            & (self.pair_n >= WINDOW)
        )
        weights = np.where(active, 1.0, 0.0)
        x_pressure = weights * np.where(active, -excess, 0.0)
        y_pressure = weights * np.where(active, beta_clipped * excess, 0.0)

        pressure = np.zeros(N_PRODUCTS, dtype=np.float64)
        pressure_count = np.zeros(N_PRODUCTS, dtype=np.float64)
        np.add.at(pressure, PAIR_X_IDX, x_pressure)
        np.add.at(pressure, PAIR_Y_IDX, y_pressure)
        np.add.at(pressure_count, PAIR_X_IDX, weights)
        np.add.at(pressure_count, PAIR_Y_IDX, weights * np.maximum(np.abs(beta_clipped), 1.0))

        pp = np.divide(pressure, pressure_count, out=np.zeros(N_PRODUCTS), where=pressure_count > 0)
        pair_skew = FAIR_ADJUSTMENT_SCALE * pp
        pair_target = np.clip(INVENTORY_TARGET_SCALE * pp, -ASSET_LIMIT, ASSET_LIMIT)
        return pair_skew, pair_target

    def _update_basket_ols(self, log_basket_mids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Same pair OLS but on the 10 basket mean log-prices (45 pairs)."""
        x_vals = log_basket_mids[BASKET_X_IDX]
        y_vals = log_basket_mids[BASKET_Y_IDX]
        ols_X = y_vals
        ols_Y = x_vals

        write_idx = self.tick % WINDOW
        if self.basket_n >= WINDOW:
            old_X = self.basket_x_buf[:, write_idx]
            old_Y = self.basket_y_buf[:, write_idx]
            self.basket_sx -= old_X
            self.basket_sy -= old_Y
            self.basket_sxx -= old_X * old_X
            self.basket_sxy -= old_X * old_Y
        self.basket_x_buf[:, write_idx] = ols_X
        self.basket_y_buf[:, write_idx] = ols_Y
        self.basket_sx += ols_X
        self.basket_sy += ols_Y
        self.basket_sxx += ols_X * ols_X
        self.basket_sxy += ols_X * ols_Y
        self.basket_n = min(self.basket_n + 1, WINDOW)

        n = float(self.basket_n)
        if self.basket_n < 2:
            return np.zeros(N_PRODUCTS), np.zeros(N_PRODUCTS)
        mx = self.basket_sx / n
        my = self.basket_sy / n
        var_x = self.basket_sxx / n - mx * mx
        cov_xy = self.basket_sxy / n - mx * my
        valid = var_x > EPS
        beta = np.where(valid, np.divide(cov_xy, np.where(valid, var_x, 1.0), out=np.zeros(N_BASKET_PAIRS), where=valid), np.nan)
        alpha = np.where(valid, my - beta * mx, np.nan)
        residual = ols_Y - alpha - beta * ols_X
        residual = np.where(np.isfinite(residual), residual, 0.0)

        # Residual rolling stats
        write_idx2 = self.tick % ZSCORE_WINDOW
        if self.basket_res_n >= ZSCORE_WINDOW:
            old = self.basket_res_buf[:, write_idx2]
            self.basket_res_sum -= old
            self.basket_res_sum2 -= old * old
        self.basket_res_buf[:, write_idx2] = residual
        self.basket_res_sum += residual
        self.basket_res_sum2 += residual * residual
        self.basket_res_n = min(self.basket_res_n + 1, ZSCORE_WINDOW)

        if self.basket_res_n < 2 or self.basket_res_n < ZSCORE_WINDOW:
            return np.zeros(N_PRODUCTS), np.zeros(N_PRODUCTS)
        bn = float(self.basket_res_n)
        b_mean = self.basket_res_sum / bn
        b_var = np.maximum(self.basket_res_sum2 / bn - b_mean * b_mean, 0.0)
        b_std = np.sqrt(b_var)
        z = np.where(b_std > EPS, (residual - b_mean) / np.where(b_std > EPS, b_std, 1.0), 0.0)
        excess = np.sign(z) * np.maximum(np.abs(z) - SIGNAL_THRESHOLD, 0.0)
        beta_clipped = np.clip(np.where(np.isfinite(beta), beta, 0.0), -BETA_CLIP, BETA_CLIP)
        active = np.isfinite(z) & np.isfinite(beta_clipped) & (excess != 0.0) & valid & (self.basket_n >= WINDOW)
        weights = np.where(active, 1.0, 0.0)
        x_pressure = weights * np.where(active, -excess, 0.0)
        y_pressure = weights * np.where(active, beta_clipped * excess, 0.0)
        b_pressure = np.zeros(N_BASKETS, dtype=np.float64)
        b_pressure_count = np.zeros(N_BASKETS, dtype=np.float64)
        np.add.at(b_pressure, BASKET_X_IDX, x_pressure)
        np.add.at(b_pressure, BASKET_Y_IDX, y_pressure)
        np.add.at(b_pressure_count, BASKET_X_IDX, weights)
        np.add.at(b_pressure_count, BASKET_Y_IDX, weights * np.maximum(np.abs(beta_clipped), 1.0))
        bp = np.divide(b_pressure, b_pressure_count, out=np.zeros(N_BASKETS), where=b_pressure_count > 0)
        basket_skew_per_basket = FAIR_ADJUSTMENT_SCALE * bp
        basket_target_per_basket = np.clip(INVENTORY_TARGET_SCALE * bp, -ASSET_LIMIT, ASSET_LIMIT)

        # Map basket-level signals to per-product
        basket_skew_pp = np.zeros(N_PRODUCTS, dtype=np.float64)
        basket_target_pp = np.zeros(N_PRODUCTS, dtype=np.float64)
        for bi, members in enumerate(BASKET_MEMBER_IDX):
            basket_skew_pp[members] = basket_skew_per_basket[bi]
            basket_target_pp[members] = basket_target_per_basket[bi]
        return basket_skew_pp, basket_target_pp

    def _cross_sec_skew(self, mids: np.ndarray, valid_product: np.ndarray) -> np.ndarray:
        """Cross-sectional rank-and-fade. Uses circular buffer of last CROSS_SEC_HORIZON mids."""
        write_idx = self.tick % CROSS_SEC_HORIZON
        # Read out value at this slot BEFORE overwriting (it's CROSS_SEC_HORIZON ticks ago)
        if self.cross_sec_n >= CROSS_SEC_HORIZON:
            lag_mids = self.cross_sec_buf[:, write_idx].copy()
            have_lag = True
        else:
            lag_mids = np.zeros(N_PRODUCTS)
            have_lag = False
        # Now overwrite with current mids
        self.cross_sec_buf[:, write_idx] = mids
        self.cross_sec_n = min(self.cross_sec_n + 1, CROSS_SEC_HORIZON)

        if not have_lag:
            return np.zeros(N_PRODUCTS, dtype=np.float64)

        valid = valid_product & (lag_mids > 0) & (mids > 0)
        if not valid.any():
            return np.zeros(N_PRODUCTS, dtype=np.float64)
        past_return = np.full(N_PRODUCTS, np.nan, dtype=np.float64)
        past_return[valid] = mids[valid] / lag_mids[valid] - 1.0

        # Percentile rank within the row
        valid_returns = past_return[valid]
        n_valid = valid_returns.size
        sorted_idx = np.argsort(valid_returns, kind="stable")
        ranks = np.empty(n_valid, dtype=np.float64)
        ranks[sorted_idx] = np.arange(1, n_valid + 1, dtype=np.float64) / n_valid
        rank_arr = np.full(N_PRODUCTS, np.nan, dtype=np.float64)
        rank_arr[valid] = ranks

        rank_mean = ranks.mean()
        centered = np.where(valid, rank_arr - rank_mean, 0.0)
        return -CROSS_SEC_SCALE * centered

    def _ovr_skew(self, mids: np.ndarray, valid_product: np.ndarray) -> np.ndarray:
        """Within-basket OVR residual z-score skew (blend-style)."""
        if N_OVR == 0:
            return np.zeros(N_PRODUCTS, dtype=np.float64)
        # Compute residual per OVR key: focal_mid - mean(rest)
        focal_mids = mids[OVR_PRODUCT_IDX]
        rest_mids = mids[OVR_REST_IDX]  # shape (N_OVR, 4)
        rest_mean = rest_mids.mean(axis=1)
        residual = focal_mids - rest_mean

        valid_focal = valid_product[OVR_PRODUCT_IDX]
        valid_rest = valid_product[OVR_REST_IDX].all(axis=1)
        valid_ovr = valid_focal & valid_rest

        skew_per_product = np.zeros(N_PRODUCTS, dtype=np.float64)
        if self.ovr_n >= OVR_WARMUP:
            std = np.sqrt(np.maximum(self.ovr_vars, 0.0))
            z = np.where(
                (std > EPS) & valid_ovr,
                (residual - self.ovr_means) / np.where(std > EPS, std, 1.0),
                0.0,
            )
            z_clipped = np.clip(z, -OVR_Z_CLIP, OVR_Z_CLIP)
            contrib = -OVR_SCALE * OVR_SCALE_VEC * z_clipped
            np.add.at(skew_per_product, OVR_PRODUCT_IDX, contrib)

        # EWMA update of mean and variance
        if self.ovr_n > 0:
            delta = residual - self.ovr_means
            new_mean = self.ovr_means + OVR_ALPHA * delta
            new_var = (1.0 - OVR_ALPHA) * (self.ovr_vars + OVR_ALPHA * delta * delta)
            update_mask = valid_ovr
            self.ovr_means = np.where(update_mask, new_mean, self.ovr_means)
            self.ovr_vars = np.where(update_mask, np.maximum(new_var, 0.0), self.ovr_vars)
        else:
            self.ovr_means = np.where(valid_ovr, residual, 0.0)
            self.ovr_vars = np.zeros(N_OVR, dtype=np.float64)
        self.ovr_n += 1
        return skew_per_product

    # -- main run -----------------------------------------------------------
    def run(self, state: TradingState):
        # Extract mids/microprices from order depths
        mids = np.full(N_PRODUCTS, np.nan, dtype=np.float64)
        micros = np.full(N_PRODUCTS, np.nan, dtype=np.float64)
        valid = np.zeros(N_PRODUCTS, dtype=bool)
        for product, depth in state.order_depths.items():
            if product not in PRODUCT_INDEX:
                continue
            i = PRODUCT_INDEX[product]
            m = _midprice(depth)
            mu = _microprice(depth)
            if m is None or mu is None or m <= 0.0:
                continue
            mids[i] = m
            micros[i] = mu
            valid[i] = True

        # If no products have data, return empty
        if not valid.any():
            self.tick += 1
            return {}, 0, ""

        # Update EWMA per product (microprice EWMA) for blend BETA layer
        first_obs = ~np.isfinite(self.ewma) & valid
        self.ewma = np.where(
            first_obs,
            micros,
            np.where(
                valid,
                (1.0 - EWMA_ALPHA) * self.ewma + EWMA_ALPHA * micros,
                self.ewma,
            ),
        )

        # Layer 1: BETA-based fair value per product
        # fair = mid + β₀(micro - mid) + β₁(ewma - mid)
        beta_arr = np.zeros((N_PRODUCTS, 2), dtype=np.float64)
        for product, idx in PRODUCT_INDEX.items():
            b = BETAS.get(product)
            if b is not None:
                beta_arr[idx, 0] = b[0]
                beta_arr[idx, 1] = b[1]
        fair = np.where(
            valid,
            mids + beta_arr[:, 0] * (micros - mids) + beta_arr[:, 1] * (self.ewma - mids),
            mids,
        )

        # Compute log mids (only for valid products; use last valid otherwise)
        log_mids = np.where(valid & (mids > 0), np.log(np.where(mids > 0, mids, 1.0)), 0.0)

        # Layer 3: Pair OLS skew + target
        alpha_p, beta_p, residual_p, valid_p = self._update_pair_ols(log_mids)
        # Update residual stats (have to update before reading z-score)
        write_idx2 = self.tick % ZSCORE_WINDOW
        if self.pair_res_n >= ZSCORE_WINDOW:
            old = self.pair_res_buf[:, write_idx2]
            self.pair_res_sum -= old
            self.pair_res_sum2 -= old * old
        self.pair_res_buf[:, write_idx2] = residual_p
        self.pair_res_sum += residual_p
        self.pair_res_sum2 += residual_p * residual_p
        self.pair_res_n = min(self.pair_res_n + 1, ZSCORE_WINDOW)
        pair_skew, pair_target = self._pair_pressure(residual_p, alpha_p, beta_p, valid_p)

        # Layer 4: Basket OLS skew + target
        # Compute basket mean log mids
        log_basket = np.zeros(N_BASKETS, dtype=np.float64)
        basket_valid = np.zeros(N_BASKETS, dtype=bool)
        for bi, members in enumerate(BASKET_MEMBER_IDX):
            members_valid = valid[members]
            if members_valid.all():
                basket_mean = mids[members].mean()
                if basket_mean > 0:
                    log_basket[bi] = math.log(basket_mean)
                    basket_valid[bi] = True
        # If any basket invalid, use last valid (carry-forward via state buffer is implicit)
        if basket_valid.all():
            basket_skew_pp, basket_target_pp = self._update_basket_ols(log_basket)
        else:
            basket_skew_pp = np.zeros(N_PRODUCTS, dtype=np.float64)
            basket_target_pp = np.zeros(N_PRODUCTS, dtype=np.float64)

        # Layer 5: Cross-sec rank skew
        cross_skew = self._cross_sec_skew(mids, valid)

        # Layer 2: OVR within-basket skew
        ovr_skew_arr = self._ovr_skew(mids, valid)

        # Layer 6: Directional skew (constant per product)
        dir_skew = np.zeros(N_PRODUCTS, dtype=np.float64)
        for product, sk in DIRECTIONAL_SKEW.items():
            if product in PRODUCT_INDEX:
                dir_skew[PRODUCT_INDEX[product]] = sk

        # Combine all skews
        total_skew = pair_skew + basket_skew_pp + cross_skew + ovr_skew_arr + dir_skew
        total_target = np.clip(pair_target + basket_target_pp, -ASSET_LIMIT, ASSET_LIMIT)

        # Generate orders
        orders: Dict[str, List[Order]] = {}
        for product, depth in state.order_depths.items():
            if product not in PRODUCT_INDEX:
                continue
            i = PRODUCT_INDEX[product]
            if not valid[i]:
                continue
            position = int(state.position.get(product, 0))
            target = float(total_target[i])

            reservation = (
                float(fair[i])
                + float(total_skew[i])
                - INVENTORY_LAMBDA * (position - target)
            )
            product_orders = _orders_for_product(
                product=product,
                depth=depth,
                position=position,
                reservation=reservation,
                limit=ASSET_LIMIT,
                passive_edge=PASSIVE_EDGE,
                take_edge=TAKE_EDGE,
                take_size=TAKE_SIZE,
            )
            if product_orders:
                orders[product] = product_orders

        self.tick += 1
        return orders, 0, ""