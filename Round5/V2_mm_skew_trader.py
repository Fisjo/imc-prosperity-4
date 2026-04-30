from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from prosperity4bt.datamodel import Order, OrderDepth, TradingState

ROUND_NUM = 5
ASSET_LIMIT = 10
WINDOW = 10
ZSCORE_WINDOW = 100
SIGNAL_THRESHOLD = 2.0
FAIR_ADJUSTMENT_SCALE = 1.0
INVENTORY_TARGET_SCALE = 3.0
INVENTORY_LAMBDA = 0.02
BETA_CLIP = 3.0
HALF_LIFE_WEIGHT_MIN = 1.0
HALF_LIFE_WEIGHT_MAX = 500.0
PASSIVE_EDGE = 0.0
TAKE_EDGE = 0.0
TAKE_SIZE = ASSET_LIMIT
PAIR_CHUNK_SIZE = 512
FAIR_METHOD = "microprice"
CURATED_FORMATION_DAYS = "2,3"
CURATED_SIGNAL_MODE = "linear"
CURATED_Z_CLIP = 3.0
CURATED_CONFIRMATION_MULTIPLIER = 0.5
CURATED_OVERLAY_MULTIPLIER = 0.5
SNACKPACK_ALPHA = 0.0
MICROCHIP_ALPHA = 1.0
SLEEP_POD_ALPHA = 1.0
OXYGEN_SHAKE_ALPHA = 1.0
TRANSLATOR_ALPHA = 1.0
PEBBLES_ALPHA = 0.0
PEBBLES_EDGE_CLIP = 5.0
ONE_VS_REST_OLS_WINDOW = 1000
ONE_VS_REST_ZSCORE_WINDOW = 100
ONE_VS_REST_BASKETS: dict[str, tuple[str, ...]] = {
    "MICROCHIP": (
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE",
        "MICROCHIP_TRIANGLE",
    ),
    "SNACKPACK": (
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_VANILLA",
    ),
}
ONE_VS_REST_SCALES: dict[str, float] = {
    "MICROCHIP": 1.0,
    "SNACKPACK": 2.0,
}


@dataclass(frozen=True)
class DayPrices:
    timestamps: np.ndarray
    products: tuple[str, ...]
    mids: np.ndarray


@dataclass(frozen=True)
class PairUniverse:
    products: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray

    @property
    def count(self) -> int:
        return int(self.x.size)


@dataclass(frozen=True)
class SkewPlan:
    day: int
    timestamps: np.ndarray
    products: tuple[str, ...]
    fair_skew: np.ndarray
    target_position: np.ndarray


@dataclass(frozen=True)
class CuratedSpreadSpec:
    name: str
    basket: str
    weights: dict[str, float]
    fair_model: str
    gain: float
    entry_z: float
    role: str = "trade"


@dataclass(frozen=True)
class CuratedSpreadFit:
    spec: CuratedSpreadSpec
    weights: np.ndarray
    mean: float
    scale: float
    trend_intercept: float = 0.0
    trend_slope: float = 0.0


@dataclass(frozen=True)
class CuratedAlphaConfig:
    formation_days: tuple[int, ...]
    signal_mode: str
    entry_z_override: float | None
    snackpack_alpha: float
    microchip_alpha: float
    sleep_pod_alpha: float
    oxygen_shake_alpha: float
    translator_alpha: float
    pebbles_alpha: float
    confirmation_multiplier: float
    overlay_multiplier: float
    z_clip: float
    pebbles_edge_clip: float

    @property
    def is_enabled(self) -> bool:
        return any(
            alpha != 0.0
            for alpha in (
                self.snackpack_alpha,
                self.microchip_alpha,
                self.sleep_pod_alpha,
                self.oxygen_shake_alpha,
                self.translator_alpha,
                self.pebbles_alpha,
            )
        )


CURATED_SPREAD_SPECS: tuple[CuratedSpreadSpec, ...] = (
    CuratedSpreadSpec(
        name="SNACK_CV",
        basket="SNACKPACK",
        weights={"SNACKPACK_CHOCOLATE": 1.0, "SNACKPACK_VANILLA": 0.928},
        fair_model="anchored",
        gain=0.005,
        entry_z=2.0,
    ),
    CuratedSpreadSpec(
        name="SNACK_PS",
        basket="SNACKPACK",
        weights={"SNACKPACK_PISTACHIO": 1.0, "SNACKPACK_STRAWBERRY": -0.590},
        fair_model="anchored",
        gain=0.0002,
        entry_z=1.5,
    ),
    CuratedSpreadSpec(
        name="SNACK_RS",
        basket="SNACKPACK",
        weights={"SNACKPACK_RASPBERRY": 1.0, "SNACKPACK_STRAWBERRY": 0.922},
        fair_model="anchored",
        gain=0.002,
        entry_z=2.0,
    ),
    CuratedSpreadSpec(
        name="SNACK_META",
        basket="SNACKPACK",
        weights={
            "SNACKPACK_CHOCOLATE": 1.0,
            "SNACKPACK_VANILLA": 1.0,
            "SNACKPACK_PISTACHIO": -1.0,
            "SNACKPACK_RASPBERRY": -1.0,
        },
        fair_model="trend",
        gain=0.0,
        entry_z=1.5,
    ),
    CuratedSpreadSpec(
        name="MICRO_LVL4",
        basket="MICROCHIP",
        weights={
            "MICROCHIP_CIRCLE": 0.23,
            "MICROCHIP_OVAL": -1.0,
            "MICROCHIP_RECTANGLE": 0.90,
            "MICROCHIP_TRIANGLE": 0.21,
        },
        fair_model="anchored",
        gain=0.005,
        entry_z=2.0,
    ),
    CuratedSpreadSpec(
        name="MICRO_OT",
        basket="MICROCHIP",
        weights={"MICROCHIP_OVAL": 1.0, "MICROCHIP_TRIANGLE": -1.0},
        fair_model="anchored",
        gain=0.002,
        entry_z=2.0,
        role="confirmation",
    ),
    CuratedSpreadSpec(
        name="SLEEP_PC",
        basket="SLEEP_POD",
        weights={"SLEEP_POD_POLYESTER": 1.0, "SLEEP_POD_COTTON": -1.0},
        fair_model="anchored",
        gain=0.002,
        entry_z=2.0,
    ),
    CuratedSpreadSpec(
        name="SLEEP_SPC",
        basket="SLEEP_POD",
        weights={
            "SLEEP_POD_SUEDE": 0.61,
            "SLEEP_POD_POLYESTER": -1.0,
            "SLEEP_POD_COTTON": 0.54,
        },
        fair_model="anchored",
        gain=0.002,
        entry_z=2.0,
    ),
    CuratedSpreadSpec(
        name="OXY_MG",
        basket="OXYGEN_SHAKE",
        weights={"OXYGEN_SHAKE_MORNING_BREATH": 1.0, "OXYGEN_SHAKE_GARLIC": -1.0},
        fair_model="anchored",
        gain=0.002,
        entry_z=1.5,
    ),
    CuratedSpreadSpec(
        name="OXY_ECG",
        basket="OXYGEN_SHAKE",
        weights={
            "OXYGEN_SHAKE_MORNING_BREATH": 1.0,
            "OXYGEN_SHAKE_EVENING_BREATH": -1.0,
            "OXYGEN_SHAKE_CHOCOLATE": 1.0,
            "OXYGEN_SHAKE_GARLIC": -1.0,
        },
        fair_model="anchored",
        gain=0.002,
        entry_z=2.0,
    ),
    CuratedSpreadSpec(
        name="OXY_EG_POS",
        basket="OXYGEN_SHAKE",
        weights={"OXYGEN_SHAKE_EVENING_BREATH": 0.71, "OXYGEN_SHAKE_GARLIC": 0.29},
        fair_model="anchored",
        gain=0.005,
        entry_z=2.0,
        role="overlay",
    ),
    CuratedSpreadSpec(
        name="TRANS_EV",
        basket="TRANSLATOR",
        weights={"TRANSLATOR_ECLIPSE_CHARCOAL": 1.0, "TRANSLATOR_VOID_BLUE": -1.0},
        fair_model="anchored",
        gain=0.002,
        entry_z=1.5,
    ),
    CuratedSpreadSpec(
        name="TRANS_SEG",
        basket="TRANSLATOR",
        weights={
            "TRANSLATOR_SPACE_GRAY": 1.0,
            "TRANSLATOR_ECLIPSE_CHARCOAL": 1.0,
            "TRANSLATOR_GRAPHITE_MIST": -1.0,
            "TRANSLATOR_VOID_BLUE": -1.0,
        },
        fair_model="anchored",
        gain=0.001,
        entry_z=1.5,
    ),
)
PEBBLES_PRODUCTS: tuple[str, ...] = (
    "PEBBLES_XS",
    "PEBBLES_S",
    "PEBBLES_M",
    "PEBBLES_L",
    "PEBBLES_XL",
)
PEBBLES_IDENTITY_TOTAL = 50_000.0


class Trader:
    def __init__(
        self,
        *,
        window: int = WINDOW,
        zscore_window: int = ZSCORE_WINDOW,
        signal_threshold: float = SIGNAL_THRESHOLD,
        fair_adjustment_scale: float = FAIR_ADJUSTMENT_SCALE,
        inventory_target_scale: float = INVENTORY_TARGET_SCALE,
        inventory_lambda: float = INVENTORY_LAMBDA,
        beta_clip: float = BETA_CLIP,
        half_life_weight_min: float = HALF_LIFE_WEIGHT_MIN,
        half_life_weight_max: float = HALF_LIFE_WEIGHT_MAX,
        passive_edge: float = PASSIVE_EDGE,
        take_edge: float = TAKE_EDGE,
        take_size: int = TAKE_SIZE,
        asset_limit: int = ASSET_LIMIT,
        pair_chunk_size: int = PAIR_CHUNK_SIZE,
        fair_method: str = FAIR_METHOD,
        curated_formation_days: str = CURATED_FORMATION_DAYS,
        curated_signal_mode: str = CURATED_SIGNAL_MODE,
        curated_entry_z_override: float | None = None,
        snackpack_alpha: float = SNACKPACK_ALPHA,
        microchip_alpha: float = MICROCHIP_ALPHA,
        sleep_pod_alpha: float = SLEEP_POD_ALPHA,
        oxygen_shake_alpha: float = OXYGEN_SHAKE_ALPHA,
        translator_alpha: float = TRANSLATOR_ALPHA,
        pebbles_alpha: float = PEBBLES_ALPHA,
        curated_confirmation_multiplier: float = CURATED_CONFIRMATION_MULTIPLIER,
        curated_overlay_multiplier: float = CURATED_OVERLAY_MULTIPLIER,
        curated_z_clip: float = CURATED_Z_CLIP,
        pebbles_edge_clip: float = PEBBLES_EDGE_CLIP,
        data_dir: str | Path | None = None,
        day: int | str | None = None,
    ) -> None:
        self.window = int(window)
        self.zscore_window = int(zscore_window)
        self.signal_threshold = float(signal_threshold)
        self.fair_adjustment_scale = float(fair_adjustment_scale)
        self.inventory_target_scale = float(inventory_target_scale)
        self.inventory_lambda = float(inventory_lambda)
        self.beta_clip = float(beta_clip)
        self.half_life_weight_min = float(half_life_weight_min)
        self.half_life_weight_max = float(half_life_weight_max)
        self.passive_edge = float(passive_edge)
        self.take_edge = float(take_edge)
        self.take_size = int(take_size)
        self.asset_limit = int(asset_limit)
        self.pair_chunk_size = int(pair_chunk_size)
        self.fair_method = fair_method
        self.curated_alpha_config = CuratedAlphaConfig(
            formation_days=_parse_days(curated_formation_days),
            signal_mode=curated_signal_mode,
            entry_z_override=None if curated_entry_z_override is None else float(curated_entry_z_override),
            snackpack_alpha=float(snackpack_alpha),
            microchip_alpha=float(microchip_alpha),
            sleep_pod_alpha=float(sleep_pod_alpha),
            oxygen_shake_alpha=float(oxygen_shake_alpha),
            translator_alpha=float(translator_alpha),
            pebbles_alpha=float(pebbles_alpha),
            confirmation_multiplier=float(curated_confirmation_multiplier),
            overlay_multiplier=float(curated_overlay_multiplier),
            z_clip=float(curated_z_clip),
            pebbles_edge_clip=float(pebbles_edge_clip),
        )
        self.data_dir = _default_data_dir() if data_dir is None else Path(data_dir)
        self.day = None if day is None else int(day)

        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if self.zscore_window < 2:
            raise ValueError("zscore_window must be at least 2.")
        if self.signal_threshold < 0:
            raise ValueError("signal_threshold must be non-negative.")
        if self.beta_clip <= 0:
            raise ValueError("beta_clip must be positive.")
        if self.half_life_weight_min <= 0 or self.half_life_weight_max < self.half_life_weight_min:
            raise ValueError("half-life weight bounds are invalid.")
        if self.take_size <= 0 or self.asset_limit <= 0 or self.pair_chunk_size <= 0:
            raise ValueError("sizes and limits must be positive.")
        if self.fair_method not in {"mid", "microprice", "wall_mid"}:
            raise ValueError("fair_method must be one of: mid, microprice, wall_mid.")
        _validate_curated_alpha_config(self.curated_alpha_config)

        self._plan: SkewPlan | None = None
        self._row_by_timestamp: dict[int, int] = {}

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        plan = self._ensure_plan()
        row = self._row_by_timestamp.get(int(state.timestamp))
        if row is None:
            return {}, 0, ""

        orders: dict[str, list[Order]] = {}
        for column, product in enumerate(plan.products):
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            fair = _local_fair_value(depth, self.fair_method)
            if fair is None:
                continue

            position = int(state.position.get(product, 0))
            reservation = (
                fair
                + float(plan.fair_skew[row, column])
                - self.inventory_lambda * (position - float(plan.target_position[row, column]))
            )
            product_orders = _orders_for_product(
                product=product,
                depth=depth,
                position=position,
                reservation=reservation,
                limit=self.asset_limit,
                passive_edge=self.passive_edge,
                take_edge=self.take_edge,
                take_size=self.take_size,
            )
            if product_orders:
                orders[product] = product_orders
        return orders, 0, ""

    def _ensure_plan(self) -> SkewPlan:
        day = self.day if self.day is not None else _day_from_environment()
        if self._plan is not None and self._plan.day == day:
            return self._plan

        self._plan = _build_plan(
            data_dir=self.data_dir,
            day=day,
            window=self.window,
            zscore_window=self.zscore_window,
            signal_threshold=self.signal_threshold,
            fair_adjustment_scale=self.fair_adjustment_scale,
            inventory_target_scale=self.inventory_target_scale,
            beta_clip=self.beta_clip,
            half_life_weight_min=self.half_life_weight_min,
            half_life_weight_max=self.half_life_weight_max,
            asset_limit=self.asset_limit,
            pair_chunk_size=self.pair_chunk_size,
            curated_alpha_config=self.curated_alpha_config,
        )
        self._row_by_timestamp = {int(timestamp): row for row, timestamp in enumerate(self._plan.timestamps)}
        return self._plan


def _build_plan(
    *,
    data_dir: Path,
    day: int,
    window: int,
    zscore_window: int,
    signal_threshold: float,
    fair_adjustment_scale: float,
    inventory_target_scale: float,
    beta_clip: float,
    half_life_weight_min: float,
    half_life_weight_max: float,
    asset_limit: int,
    pair_chunk_size: int,
    curated_alpha_config: CuratedAlphaConfig,
) -> SkewPlan:
    prices = _load_day_prices(data_dir, day)
    if prices.mids.shape[0] < window + zscore_window - 1:
        raise ValueError("Not enough rows for the requested OLS and zscore windows.")

    log_prices = np.log(prices.mids)
    product_universe = _ordered_pair_universe(prices.products)
    product_ols = _rolling_ols(log_prices, window, product_universe, pair_chunk_size)
    product_pair_half_life = _pair_half_life(log_prices, window, product_universe, product_ols.beta, pair_chunk_size)
    product_fair_skew, product_target_position = _pair_skews(
        model_prices=log_prices[window - 1 :],
        universe=product_universe,
        beta=product_ols.beta,
        alpha=product_ols.alpha,
        pair_half_life=product_pair_half_life,
        zscore_window=zscore_window,
        signal_threshold=signal_threshold,
        fair_adjustment_scale=fair_adjustment_scale,
        inventory_target_scale=inventory_target_scale,
        beta_clip=beta_clip,
        half_life_weight_min=half_life_weight_min,
        half_life_weight_max=half_life_weight_max,
        asset_limit=asset_limit,
        pair_chunk_size=pair_chunk_size,
    )

    fair_skew = product_fair_skew.astype(np.float32)
    fair_skew += _one_vs_rest_skews(prices.products, prices.mids, ONE_VS_REST_SCALES)[window - 1 :]
    if curated_alpha_config.is_enabled:
        fair_skew += _curated_reservation_skews(
            data_dir=data_dir,
            prices=prices,
            config=curated_alpha_config,
        )[window - 1 :]
    target_position = product_target_position.astype(np.float32)

    row_count = prices.mids.shape[0]
    full_fair_skew = np.zeros((row_count, len(prices.products)), dtype=np.float32)
    full_target = np.zeros((row_count, len(prices.products)), dtype=np.float32)
    full_fair_skew[window - 1 :] = fair_skew
    full_target[window - 1 :] = target_position
    return SkewPlan(day=day, timestamps=prices.timestamps, products=prices.products, fair_skew=full_fair_skew, target_position=full_target)


@dataclass(frozen=True)
class RollingOls:
    beta: np.ndarray
    alpha: np.ndarray


def _rolling_ols(prices: np.ndarray, window: int, universe: PairUniverse, chunk_size: int) -> RollingOls:
    rolling_sum = _rolling_sum(prices, window)
    rolling_sum2 = _rolling_sum(prices * prices, window)
    row_count = rolling_sum.shape[0]
    beta = np.full((row_count, universe.count), np.nan, dtype=np.float32)
    alpha = np.full((row_count, universe.count), np.nan, dtype=np.float32)

    for start in range(0, universe.count, chunk_size):
        end = min(start + chunk_size, universe.count)
        x_idx = universe.y[start:end]
        y_idx = universe.x[start:end]
        sum_x = rolling_sum[:, x_idx]
        sum_y = rolling_sum[:, y_idx]
        sum_x2 = rolling_sum2[:, x_idx]
        sum_xy = _rolling_pair_product_sum(prices, x_idx, y_idx, window)
        ss_x = sum_x2 - (sum_x * sum_x) / window
        cov_xy = sum_xy - (sum_x * sum_y) / window
        beta_chunk = np.divide(cov_xy, ss_x, out=np.full_like(cov_xy, np.nan), where=ss_x > 1e-12)
        alpha_chunk = (sum_y / window) - beta_chunk * (sum_x / window)
        beta[:, start:end] = beta_chunk.astype(np.float32, copy=False)
        alpha[:, start:end] = alpha_chunk.astype(np.float32, copy=False)
    return RollingOls(beta=beta, alpha=alpha)


def _pair_half_life(
    prices: np.ndarray,
    window: int,
    universe: PairUniverse,
    beta: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    ar_window = window - 1
    lag = prices[:-1]
    current = prices[1:]
    lag_sum = _rolling_sum(lag, ar_window)
    current_sum = _rolling_sum(current, ar_window)
    lag_sum2 = _rolling_sum(lag * lag, ar_window)
    current_lag_product = _rolling_sum(lag * current, ar_window)
    half_life = np.full_like(beta, np.nan, dtype=np.float32)

    for start in range(0, universe.count, chunk_size):
        end = min(start + chunk_size, universe.count)
        x_idx = universe.y[start:end]
        y_idx = universe.x[start:end]
        b = beta[:, start:end].astype(np.float64, copy=False)

        lag_u_sum = lag_sum[:, y_idx] - b * lag_sum[:, x_idx]
        current_u_sum = current_sum[:, y_idx] - b * current_sum[:, x_idx]
        lag_u_sum2 = (
            lag_sum2[:, y_idx]
            - 2.0 * b * _rolling_pair_product_sum(lag, x_idx, y_idx, ar_window)
            + b * b * lag_sum2[:, x_idx]
        )
        lag_current_u_sum = (
            current_lag_product[:, y_idx]
            - b * _rolling_cross_product_sum(lag, current, x_idx, y_idx, ar_window)
            - b * _rolling_cross_product_sum(lag, current, y_idx, x_idx, ar_window)
            + b * b * current_lag_product[:, x_idx]
        )

        numerator = lag_current_u_sum - (lag_u_sum * current_u_sum) / ar_window
        denominator = lag_u_sum2 - (lag_u_sum * lag_u_sum) / ar_window
        phi = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-12)
        valid = (phi > 0.0) & (phi < 1.0)
        life = np.full_like(phi, np.nan)
        life[valid] = -np.log(2.0) / np.log(phi[valid])
        half_life[:, start:end] = life.astype(np.float32, copy=False)
    return half_life


def _pair_skews(
    *,
    model_prices: np.ndarray,
    universe: PairUniverse,
    beta: np.ndarray,
    alpha: np.ndarray,
    pair_half_life: np.ndarray | None,
    zscore_window: int,
    signal_threshold: float,
    fair_adjustment_scale: float,
    inventory_target_scale: float,
    beta_clip: float,
    half_life_weight_min: float,
    half_life_weight_max: float,
    asset_limit: int,
    pair_chunk_size: int,
    half_life_weighting: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = model_prices.shape[0]
    asset_count = len(universe.products)
    pressure_sum = np.zeros((row_count, asset_count), dtype=np.float64)
    pressure_count = np.zeros((row_count, asset_count), dtype=np.float64)

    for start in range(0, universe.count, pair_chunk_size):
        end = min(start + pair_chunk_size, universe.count)
        x_idx = universe.x[start:end]
        y_idx = universe.y[start:end]
        b = beta[:, start:end].astype(np.float64, copy=False)
        a = alpha[:, start:end].astype(np.float64, copy=False)
        beta_for_pressure = np.clip(b, -beta_clip, beta_clip)

        spread = model_prices[:, x_idx] - a - b * model_prices[:, y_idx]
        mean, std = _rolling_mean_std(spread, zscore_window)
        signal = np.divide(spread - mean, std, out=np.full_like(spread, np.nan), where=std > 1e-12)
        excess = np.sign(signal) * np.maximum(np.abs(signal) - signal_threshold, 0.0)

        active = np.isfinite(signal) & np.isfinite(beta_for_pressure) & (excess != 0.0)
        if half_life_weighting:
            if pair_half_life is None:
                raise RuntimeError("half_life_weighting requires pair half-life values.")
            life = pair_half_life[:, start:end]
            active &= np.isfinite(life)
            clipped_life = np.clip(life, half_life_weight_min, half_life_weight_max)
            weights = np.divide(
                1.0,
                np.sqrt(clipped_life),
                out=np.zeros_like(clipped_life, dtype=np.float64),
                where=np.isfinite(clipped_life) & (clipped_life > 0.0),
            )
            weights = np.where(active, weights, 0.0)
        else:
            weights = np.where(active, 1.0, 0.0)

        x_pressure = weights * np.where(active, -excess, 0.0)
        y_pressure = weights * np.where(active, beta_for_pressure * excess, 0.0)
        rows, cols = np.nonzero(active)
        if rows.size == 0:
            continue
        np.add.at(pressure_sum, (rows, x_idx[cols]), x_pressure[rows, cols])
        np.add.at(pressure_sum, (rows, y_idx[cols]), y_pressure[rows, cols])
        np.add.at(pressure_count, (rows, x_idx[cols]), weights[rows, cols])
        np.add.at(
            pressure_count,
            (rows, y_idx[cols]),
            weights[rows, cols] * np.maximum(np.abs(beta_for_pressure[rows, cols]), 1.0),
        )

    pressure = np.divide(pressure_sum, pressure_count, out=np.zeros_like(pressure_sum), where=pressure_count > 0)
    fair_skew = (fair_adjustment_scale * pressure).astype(np.float32)
    target = np.clip(inventory_target_scale * pressure, -asset_limit, asset_limit).astype(np.float32)
    return fair_skew, target


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
) -> list[Order]:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return []

    orders: list[Order] = []
    working_position = position

    buy_capacity = min(take_size, max(0, limit - working_position))
    for ask_price in sorted(depth.sell_orders):
        if buy_capacity <= 0 or reservation - ask_price <= take_edge:
            break
        quantity = min(abs(int(depth.sell_orders[ask_price])), buy_capacity)
        if quantity > 0:
            orders.append(Order(product, int(ask_price), quantity))
            working_position += quantity
            buy_capacity -= quantity

    sell_capacity = min(take_size, max(0, limit + working_position))
    for bid_price in sorted(depth.buy_orders, reverse=True):
        if sell_capacity <= 0 or bid_price - reservation <= take_edge:
            break
        quantity = min(int(depth.buy_orders[bid_price]), sell_capacity)
        if quantity > 0:
            orders.append(Order(product, int(bid_price), -quantity))
            working_position -= quantity
            sell_capacity -= quantity

    passive_buy_capacity = max(0, limit - working_position)
    passive_bid = best_bid + 1 if best_bid + 1 < best_ask else best_bid
    if passive_buy_capacity > 0 and reservation - passive_bid > passive_edge:
        orders.append(Order(product, int(passive_bid), passive_buy_capacity))

    passive_sell_capacity = max(0, limit + working_position)
    passive_ask = best_ask - 1 if best_ask - 1 > best_bid else best_ask
    if passive_sell_capacity > 0 and passive_ask - reservation > passive_edge:
        orders.append(Order(product, int(passive_ask), -passive_sell_capacity))

    return orders


def _local_fair_value(depth: OrderDepth, fair_method: str) -> float | None:
    if fair_method == "mid":
        return _midprice(depth)
    if fair_method == "wall_mid":
        return _wall_mid(depth)
    return _microprice(depth)


def _midprice(depth: OrderDepth) -> float | None:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def _wall_mid(depth: OrderDepth) -> float | None:
    bid_prices = [price for price, volume in depth.buy_orders.items() if volume > 0]
    ask_prices = [price for price, volume in depth.sell_orders.items() if volume < 0]
    if not bid_prices or not ask_prices:
        return None
    return (min(bid_prices) + max(ask_prices)) / 2.0


def _microprice(depth: OrderDepth) -> float | None:
    best_bid = _best_bid(depth)
    best_ask = _best_ask(depth)
    if best_bid is None or best_ask is None:
        return None

    bid_notional = 0.0
    bid_volume = 0
    for price, volume in depth.buy_orders.items():
        if volume > 0:
            bid_notional += float(price * volume)
            bid_volume += int(volume)

    ask_notional = 0.0
    ask_volume = 0
    for price, volume in depth.sell_orders.items():
        if volume < 0:
            size = abs(int(volume))
            ask_notional += float(price * size)
            ask_volume += size

    total_volume = bid_volume + ask_volume
    if total_volume <= 0:
        return (best_bid + best_ask) / 2.0
    bid_vwap = bid_notional / bid_volume if bid_volume > 0 else float(best_bid)
    ask_vwap = ask_notional / ask_volume if ask_volume > 0 else float(best_ask)
    return (bid_vwap * ask_volume + ask_vwap * bid_volume) / total_volume


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    padded = np.pad(values, [(1, 0), (0, 0)], mode="constant", constant_values=0.0)
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    return cumulative[window:] - cumulative[:-window]


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full_like(values, np.nan, dtype=np.float64)
    std = np.full_like(values, np.nan, dtype=np.float64)
    sums = _rolling_sum(values, window)
    sums2 = _rolling_sum(values * values, window)
    rolling_mean = sums / window
    variance = (sums2 - (sums * sums) / window) / max(window - 1, 1)
    variance = np.maximum(variance, 0.0)
    mean[window - 1 :] = rolling_mean
    std[window - 1 :] = np.sqrt(variance)
    return mean, std


def _rolling_pair_product_sum(prices: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray, window: int) -> np.ndarray:
    return _rolling_sum(prices[:, x_idx] * prices[:, y_idx], window)


def _rolling_cross_product_sum(
    left: np.ndarray,
    right: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    window: int,
) -> np.ndarray:
    return _rolling_sum(left[:, x_idx] * right[:, y_idx], window)


def _ordered_pair_universe(products: tuple[str, ...]) -> PairUniverse:
    asset_count = len(products)
    x = np.repeat(np.arange(asset_count, dtype=np.int32), asset_count)
    y = np.tile(np.arange(asset_count, dtype=np.int32), asset_count)
    keep = x != y
    return PairUniverse(products=products, x=x[keep], y=y[keep])


def _one_vs_rest_skews(
    products: tuple[str, ...],
    prices: np.ndarray,
    basket_scales: dict[str, float],
) -> np.ndarray:
    skews = np.zeros_like(prices, dtype=np.float32)
    product_index = {product: index for index, product in enumerate(products)}
    for basket, scale in basket_scales.items():
        basket_indices = [product_index[product] for product in ONE_VS_REST_BASKETS[basket]]
        for target_index in basket_indices:
            rest_indices = [index for index in basket_indices if index != target_index]
            residual_z = _rolling_one_vs_rest_residual_zscore(
                prices[:, target_index],
                prices[:, rest_indices].mean(axis=1),
            )
            skews[:, target_index] = -float(scale) * residual_z
    return skews


def _rolling_one_vs_rest_residual_zscore(target: np.ndarray, rest_mean: np.ndarray) -> np.ndarray:
    x = pd.Series(rest_mean)
    y = pd.Series(target)
    mean_x = x.rolling(ONE_VS_REST_OLS_WINDOW).mean()
    mean_y = y.rolling(ONE_VS_REST_OLS_WINDOW).mean()
    mean_x2 = (x * x).rolling(ONE_VS_REST_OLS_WINDOW).mean()
    mean_xy = (x * y).rolling(ONE_VS_REST_OLS_WINDOW).mean()
    var_x = mean_x2 - mean_x * mean_x
    beta = (mean_xy - mean_x * mean_y) / var_x.where(var_x > 1e-12)
    alpha = mean_y - beta * mean_x
    residual = y - alpha - beta * x
    zscore = (residual - residual.rolling(ONE_VS_REST_ZSCORE_WINDOW).mean()) / residual.rolling(
        ONE_VS_REST_ZSCORE_WINDOW
    ).std(ddof=1)
    return zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


def _curated_reservation_skews(
    *,
    data_dir: Path,
    prices: DayPrices,
    config: CuratedAlphaConfig,
) -> np.ndarray:
    skews = np.zeros_like(prices.mids, dtype=np.float32)
    fits = _fit_curated_spreads(data_dir, prices.products, config.formation_days)

    for fit in fits:
        alpha = _curated_alpha(fit.spec, config)
        if alpha == 0.0:
            continue

        spread = prices.mids @ fit.weights
        residual = _curated_day_residual(spread, fit)
        zscore = (residual - fit.mean) / fit.scale
        entry_z = fit.spec.entry_z if config.entry_z_override is None else config.entry_z_override
        signal = _curated_signal(zscore, entry_z, config)
        skews += (-alpha * signal[:, None] * fit.weights[None, :]).astype(np.float32)

    if config.pebbles_alpha != 0.0:
        skews += _pebbles_identity_skews(prices, config)
    return skews


def _fit_curated_spreads(
    data_dir: Path,
    products: tuple[str, ...],
    formation_days: tuple[int, ...],
) -> tuple[CuratedSpreadFit, ...]:
    formation_prices = tuple(_load_day_prices(data_dir, day) for day in formation_days)
    fits: list[CuratedSpreadFit] = []

    for spec in CURATED_SPREAD_SPECS:
        normalized_weights = _normalized_spread_weights(spec)
        current_weights = _weight_vector(products, normalized_weights)
        formation_spreads = tuple(
            _spread_from_weights(day_prices.products, day_prices.mids, normalized_weights)
            for day_prices in formation_prices
        )
        mean, scale, trend_intercept, trend_slope = _fit_curated_residual_stats(spec, formation_spreads)
        fits.append(
            CuratedSpreadFit(
                spec=spec,
                weights=current_weights,
                mean=mean,
                scale=scale,
                trend_intercept=trend_intercept,
                trend_slope=trend_slope,
            )
        )
    return tuple(fits)


def _fit_curated_residual_stats(
    spec: CuratedSpreadSpec,
    formation_spreads: tuple[np.ndarray, ...],
) -> tuple[float, float, float, float]:
    trend_intercept = 0.0
    trend_slope = 0.0
    if spec.fair_model == "anchored":
        residuals = np.concatenate(tuple(_anchored_residuals(spread) for spread in formation_spreads))
    elif spec.fair_model == "trend":
        x_values = np.concatenate(tuple(np.arange(spread.size, dtype=np.float64) for spread in formation_spreads))
        y_values = np.concatenate(formation_spreads)
        design = np.column_stack((np.ones_like(x_values), x_values))
        trend_intercept, trend_slope = np.linalg.lstsq(design, y_values, rcond=None)[0]
        residuals = y_values - (trend_intercept + trend_slope * x_values)
    else:
        raise ValueError(f"Unsupported curated fair model: {spec.fair_model}")

    finite = residuals[np.isfinite(residuals)]
    if finite.size < 2:
        raise ValueError(f"Not enough finite formation residuals for {spec.name}.")
    scale = float(np.std(finite, ddof=1))
    if scale <= 1e-12:
        raise ValueError(f"Formation residual scale is zero for {spec.name}.")
    return float(np.mean(finite)), scale, float(trend_intercept), float(trend_slope)


def _curated_day_residual(spread: np.ndarray, fit: CuratedSpreadFit) -> np.ndarray:
    if fit.spec.fair_model == "anchored":
        return _anchored_residuals(spread)
    if fit.spec.fair_model == "trend":
        x_values = np.arange(spread.size, dtype=np.float64)
        return spread - (fit.trend_intercept + fit.trend_slope * x_values)
    raise ValueError(f"Unsupported curated fair model: {fit.spec.fair_model}")


def _anchored_residuals(spread: np.ndarray) -> np.ndarray:
    residuals = np.zeros_like(spread, dtype=np.float64)
    if spread.size == 0:
        return residuals

    fair_value = float(spread[0])
    for row, value in enumerate(spread):
        residuals[row] = float(value) - fair_value
    return residuals


def _curated_signal(zscore: np.ndarray, entry_z: float, config: CuratedAlphaConfig) -> np.ndarray:
    clipped = np.clip(zscore, -config.z_clip, config.z_clip)
    if config.signal_mode == "linear":
        return clipped
    excess = np.maximum(np.abs(clipped) - entry_z, 0.0)
    return np.sign(clipped) * excess


def _curated_alpha(spec: CuratedSpreadSpec, config: CuratedAlphaConfig) -> float:
    basket_alpha = {
        "SNACKPACK": config.snackpack_alpha,
        "MICROCHIP": config.microchip_alpha,
        "SLEEP_POD": config.sleep_pod_alpha,
        "OXYGEN_SHAKE": config.oxygen_shake_alpha,
        "TRANSLATOR": config.translator_alpha,
    }[spec.basket]
    if spec.role == "confirmation":
        return basket_alpha * config.confirmation_multiplier
    if spec.role == "overlay":
        return basket_alpha * config.overlay_multiplier
    return basket_alpha


def _pebbles_identity_skews(prices: DayPrices, config: CuratedAlphaConfig) -> np.ndarray:
    product_index = {product: index for index, product in enumerate(prices.products)}
    pebbles_indices = [product_index[product] for product in PEBBLES_PRODUCTS]
    pebbles_prices = prices.mids[:, pebbles_indices]
    total = pebbles_prices.sum(axis=1)

    skews = np.zeros_like(prices.mids, dtype=np.float32)
    for column in pebbles_indices:
        identity_fair = PEBBLES_IDENTITY_TOTAL - (total - prices.mids[:, column])
        edge = np.clip(identity_fair - prices.mids[:, column], -config.pebbles_edge_clip, config.pebbles_edge_clip)
        skews[:, column] = (config.pebbles_alpha * edge).astype(np.float32)
    return skews


def _normalized_spread_weights(spec: CuratedSpreadSpec) -> dict[str, float]:
    l1_norm = sum(abs(weight) for weight in spec.weights.values())
    if l1_norm <= 0.0:
        raise ValueError(f"Spread {spec.name} has zero L1 weight.")
    return {product: float(weight) / l1_norm for product, weight in spec.weights.items()}


def _weight_vector(products: tuple[str, ...], weights: dict[str, float]) -> np.ndarray:
    product_index = {product: index for index, product in enumerate(products)}
    vector = np.zeros(len(products), dtype=np.float64)
    for product, weight in weights.items():
        vector[product_index[product]] = weight
    return vector


def _spread_from_weights(
    products: tuple[str, ...],
    prices: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    return prices @ _weight_vector(products, weights)


def _parse_days(raw_days: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(raw_days, str):
        days = tuple(int(part.strip()) for part in raw_days.split(",") if part.strip())
    else:
        days = tuple(int(day) for day in raw_days)
    if not days:
        raise ValueError("curated_formation_days must contain at least one day.")
    return days


def _validate_curated_alpha_config(config: CuratedAlphaConfig) -> None:
    if config.signal_mode not in {"linear", "excess"}:
        raise ValueError("curated_signal_mode must be 'linear' or 'excess'.")
    if config.z_clip <= 0.0:
        raise ValueError("curated_z_clip must be positive.")
    if config.pebbles_edge_clip <= 0.0:
        raise ValueError("pebbles_edge_clip must be positive.")
    if config.entry_z_override is not None and config.entry_z_override < 0.0:
        raise ValueError("curated_entry_z_override must be non-negative.")
    alphas = (
        config.snackpack_alpha,
        config.microchip_alpha,
        config.sleep_pod_alpha,
        config.oxygen_shake_alpha,
        config.translator_alpha,
        config.pebbles_alpha,
        config.confirmation_multiplier,
        config.overlay_multiplier,
    )
    if any(alpha < 0.0 for alpha in alphas):
        raise ValueError("curated alpha values and multipliers must be non-negative.")


def _load_day_prices(data_dir: Path, day: int) -> DayPrices:
    path = data_dir / f"prices_round_{ROUND_NUM}_day_{day}.csv"
    frame = pd.read_csv(path, sep=";", usecols=["timestamp", "product", "mid_price"])
    mids = frame.pivot(index="timestamp", columns="product", values="mid_price")
    mids = mids.sort_index().reindex(sorted(mids.columns), axis=1)
    if mids.isna().any(axis=None):
        raise ValueError(f"Missing timestamp/product prices in {path}.")
    return DayPrices(
        timestamps=mids.index.to_numpy(dtype=np.int64),
        products=tuple(str(column) for column in mids.columns),
        mids=np.ascontiguousarray(mids.to_numpy(dtype=np.float64)),
    )


def _default_data_dir() -> Path:
    """
    Locate the data directory that holds the prices CSVs.

    Lookup order (first match wins):
      1. ``STRAT_DATA_DIR`` environment variable, if set.
      2. Walk up from this file's location looking for the first
         ancestor that contains a sibling folder named ``data``.
      3. Fallback to ``<script_dir>/data``.

    The env-var override lets you point a single backtest at a custom
    data folder without editing the file, e.g. in PowerShell:
        $env:STRAT_DATA_DIR = "C:\\path\\to\\Round5\\data"
    """
    override = os.environ.get("STRAT_DATA_DIR")
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
    return here.parent / "data"


def _day_from_environment() -> int:
    raw_day = os.environ.get("PROSPERITY4BT_DAY")
    if raw_day is None:
        raise RuntimeError("Cannot infer round day. Pass day=... or run through prosperity4bt.")
    return int(raw_day)


def _best_bid(depth: OrderDepth) -> int | None:
    prices = [price for price, volume in depth.buy_orders.items() if volume > 0]
    return max(prices) if prices else None


def _best_ask(depth: OrderDepth) -> int | None:
    prices = [price for price, volume in depth.sell_orders.items() if volume < 0]
    return min(prices) if prices else None