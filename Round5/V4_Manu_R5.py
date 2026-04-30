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
CROSS_SECTIONAL_RETURN_HORIZON = 1500
CROSS_SECTIONAL_SKEW_SCALE = 8.0
BETA_CLIP = 3.0
HALF_LIFE_WEIGHT_MIN = 1.0
HALF_LIFE_WEIGHT_MAX = 500.0
PASSIVE_EDGE = 0.0
TAKE_EDGE = 0.0
TAKE_SIZE = ASSET_LIMIT
PAIR_CHUNK_SIZE = 512
BASKET_PREFIXES: dict[str, str] = {
    "GALAXY_SOUNDS": "GALAXY_SOUNDS_",
    "MICROCHIP": "MICROCHIP_",
    "OXYGEN_SHAKE": "OXYGEN_SHAKE_",
    "PANEL": "PANEL_",
    "PEBBLES": "PEBBLES_",
    "ROBOT": "ROBOT_",
    "SLEEP_POD": "SLEEP_POD_",
    "SNACKPACK": "SNACKPACK_",
    "TRANSLATOR": "TRANSLATOR_",
    "UV_VISOR": "UV_VISOR_",
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
class BasketUniverse:
    baskets: tuple[str, ...]
    basket_products: tuple[tuple[str, ...], ...]
    product_to_basket: np.ndarray


@dataclass(frozen=True)
class SkewPlan:
    day: int
    timestamps: np.ndarray
    products: tuple[str, ...]
    fair_skew: np.ndarray
    target_position: np.ndarray


class MmSkewBaseTrader:
    def __init__(
        self,
        *,
        window: int = WINDOW,
        zscore_window: int = ZSCORE_WINDOW,
        signal_threshold: float = SIGNAL_THRESHOLD,
        fair_adjustment_scale: float = FAIR_ADJUSTMENT_SCALE,
        inventory_target_scale: float = INVENTORY_TARGET_SCALE,
        inventory_lambda: float = INVENTORY_LAMBDA,
        cross_sectional_return_horizon: int = CROSS_SECTIONAL_RETURN_HORIZON,
        cross_sectional_skew_scale: float = CROSS_SECTIONAL_SKEW_SCALE,
        beta_clip: float = BETA_CLIP,
        half_life_weight_min: float = HALF_LIFE_WEIGHT_MIN,
        half_life_weight_max: float = HALF_LIFE_WEIGHT_MAX,
        passive_edge: float = PASSIVE_EDGE,
        take_edge: float = TAKE_EDGE,
        take_size: int = TAKE_SIZE,
        asset_limit: int = ASSET_LIMIT,
        pair_chunk_size: int = PAIR_CHUNK_SIZE,
        data_dir: str | Path | None = None,
        day: int | str | None = None,
    ) -> None:
        self.window = int(window)
        self.zscore_window = int(zscore_window)
        self.signal_threshold = float(signal_threshold)
        self.fair_adjustment_scale = float(fair_adjustment_scale)
        self.inventory_target_scale = float(inventory_target_scale)
        self.inventory_lambda = float(inventory_lambda)
        self.cross_sectional_return_horizon = int(cross_sectional_return_horizon)
        self.cross_sectional_skew_scale = float(cross_sectional_skew_scale)
        self.beta_clip = float(beta_clip)
        self.half_life_weight_min = float(half_life_weight_min)
        self.half_life_weight_max = float(half_life_weight_max)
        self.passive_edge = float(passive_edge)
        self.take_edge = float(take_edge)
        self.take_size = int(take_size)
        self.asset_limit = int(asset_limit)
        self.pair_chunk_size = int(pair_chunk_size)
        self.data_dir = _default_data_dir() if data_dir is None else Path(data_dir)
        self.day = None if day is None else int(day)

        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if self.zscore_window < 2:
            raise ValueError("zscore_window must be at least 2.")
        if self.signal_threshold < 0:
            raise ValueError("signal_threshold must be non-negative.")
        if self.cross_sectional_return_horizon <= 0:
            raise ValueError("cross_sectional_return_horizon must be positive.")
        if self.beta_clip <= 0:
            raise ValueError("beta_clip must be positive.")
        if self.half_life_weight_min <= 0 or self.half_life_weight_max < self.half_life_weight_min:
            raise ValueError("half-life weight bounds are invalid.")
        if self.take_size <= 0 or self.asset_limit <= 0 or self.pair_chunk_size <= 0:
            raise ValueError("sizes and limits must be positive.")

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
            fair = _microprice(depth)
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
            cross_sectional_return_horizon=self.cross_sectional_return_horizon,
            cross_sectional_skew_scale=self.cross_sectional_skew_scale,
            beta_clip=self.beta_clip,
            half_life_weight_min=self.half_life_weight_min,
            half_life_weight_max=self.half_life_weight_max,
            asset_limit=self.asset_limit,
            pair_chunk_size=self.pair_chunk_size,
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
    cross_sectional_return_horizon: int,
    cross_sectional_skew_scale: float,
    beta_clip: float,
    half_life_weight_min: float,
    half_life_weight_max: float,
    asset_limit: int,
    pair_chunk_size: int,
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

    basket_universe = _basket_universe(prices.products)
    basket_mids = _equal_weighted_basket_prices(prices.mids, basket_universe)
    log_basket_prices = np.log(basket_mids)
    basket_pair_universe = _ordered_pair_universe(basket_universe.baskets)
    basket_ols = _rolling_ols(log_basket_prices, window, basket_pair_universe, pair_chunk_size)
    basket_fair_skew, basket_target_position = _pair_skews(
        model_prices=log_basket_prices[window - 1 :],
        universe=basket_pair_universe,
        beta=basket_ols.beta,
        alpha=basket_ols.alpha,
        pair_half_life=None,
        zscore_window=zscore_window,
        signal_threshold=signal_threshold,
        fair_adjustment_scale=fair_adjustment_scale,
        inventory_target_scale=inventory_target_scale,
        beta_clip=beta_clip,
        half_life_weight_min=half_life_weight_min,
        half_life_weight_max=half_life_weight_max,
        asset_limit=asset_limit,
        pair_chunk_size=pair_chunk_size,
        half_life_weighting=False,
    )
    mapped_basket_fair_skew = _map_basket_values_to_products(basket_fair_skew, basket_universe.product_to_basket)
    mapped_basket_target_position = _map_basket_values_to_products(
        basket_target_position,
        basket_universe.product_to_basket,
    )

    fair_skew = (product_fair_skew + mapped_basket_fair_skew).astype(np.float32)
    cross_sectional_skew = _cross_sectional_return_skew(
        prices.mids,
        horizon=cross_sectional_return_horizon,
        scale=cross_sectional_skew_scale,
    )[window - 1 :]
    fair_skew = (fair_skew + cross_sectional_skew).astype(np.float32)
    target_position = np.clip(
        product_target_position + mapped_basket_target_position,
        -asset_limit,
        asset_limit,
    ).astype(np.float32)

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
        np.add.at(pressure_count, (rows, y_idx[cols]), weights[rows, cols] * np.maximum(np.abs(beta_for_pressure[rows, cols]), 1.0))

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


def _cross_sectional_return_skew(prices: np.ndarray, *, horizon: int, scale: float) -> np.ndarray:
    if scale == 0.0:
        return np.zeros_like(prices, dtype=np.float32)
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if horizon >= prices.shape[0]:
        return np.zeros_like(prices, dtype=np.float32)

    past_return = np.full_like(prices, np.nan, dtype=np.float64)
    past_return[horizon:] = prices[horizon:] / prices[:-horizon] - 1.0
    rank = pd.DataFrame(past_return).rank(axis=1, pct=True).to_numpy(dtype=np.float64)
    valid = np.isfinite(rank)
    count = valid.sum(axis=1, keepdims=True)
    rank_sum = np.where(valid, rank, 0.0).sum(axis=1, keepdims=True)
    rank_mean = np.divide(rank_sum, count, out=np.zeros_like(rank_sum), where=count > 0)
    centered_rank = rank - rank_mean
    return np.nan_to_num(-scale * centered_rank, nan=0.0).astype(np.float32)


def _basket_universe(products: tuple[str, ...]) -> BasketUniverse:
    basket_products: list[tuple[str, ...]] = []
    product_to_basket = np.full(len(products), -1, dtype=np.int32)
    product_index = {product: index for index, product in enumerate(products)}

    for basket_index, (basket, prefix) in enumerate(BASKET_PREFIXES.items()):
        members = tuple(product for product in products if product.startswith(prefix))
        if len(members) != 5:
            raise ValueError(f"{basket}: expected 5 products, found {len(members)}.")
        basket_products.append(members)
        for product in members:
            product_to_basket[product_index[product]] = basket_index

    if np.any(product_to_basket < 0):
        missing = [product for product, basket_index in zip(products, product_to_basket) if basket_index < 0]
        raise ValueError(f"Products are not assigned to a basket: {missing!r}.")

    return BasketUniverse(
        baskets=tuple(BASKET_PREFIXES),
        basket_products=tuple(basket_products),
        product_to_basket=product_to_basket,
    )


def _equal_weighted_basket_prices(product_prices: np.ndarray, basket_universe: BasketUniverse) -> np.ndarray:
    basket_prices = np.empty((product_prices.shape[0], len(basket_universe.baskets)), dtype=np.float64)
    for basket_index, members in enumerate(basket_universe.basket_products):
        product_indices = np.nonzero(basket_universe.product_to_basket == basket_index)[0]
        if product_indices.size != len(members):
            raise RuntimeError("Basket/product mapping is inconsistent.")
        basket_prices[:, basket_index] = product_prices[:, product_indices].mean(axis=1)
    return basket_prices


def _map_basket_values_to_products(values: np.ndarray, product_to_basket: np.ndarray) -> np.ndarray:
    return values[:, product_to_basket].astype(np.float32, copy=False)


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


PEBBLES_PRODUCTS = ("PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL")
PEBBLES_ACTIVE = ("PEBBLES_S", "PEBBLES_XL")
PEBBLES_TARGET_SUM = 50000.0
PEBBLES_TRIGGER = 8.5
PEBBLES_TAKE_SIZE = 2
PEBBLES_PLAN_SKEW_BLEND = 0.25


class BaseTrader(MmSkewBaseTrader):
    def __init__(self, *args, tradeable_products: set[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tradeable_products = None if tradeable_products is None else set(tradeable_products)

    def run(self, state: TradingState):
        orders, conversions, trader_data = super().run(state)
        if self.tradeable_products is None:
            filtered_orders = dict(orders)
        else:
            filtered_orders = {
                product: product_orders
                for product, product_orders in orders.items()
                if product in self.tradeable_products
            }

        residual, synth_fair = _pebbles_group_residual_and_fair(state)
        if residual is None or synth_fair is None or abs(residual) < PEBBLES_TRIGGER:
            return filtered_orders, conversions, trader_data

        plan = self._ensure_plan()
        row = self._row_by_timestamp.get(int(state.timestamp))
        if row is None:
            return filtered_orders, conversions, trader_data

        product_to_column = {product: column for column, product in enumerate(plan.products)}

        for product in PEBBLES_ACTIVE:
            if self.tradeable_products is not None and product not in self.tradeable_products:
                continue
            depth = state.order_depths.get(product)
            column = product_to_column.get(product)
            if depth is None or column is None:
                continue

            position = int(state.position.get(product, 0))
            reservation = (
                synth_fair[product]
                + PEBBLES_PLAN_SKEW_BLEND * float(plan.fair_skew[row, column])
                - self.inventory_lambda * (position - float(plan.target_position[row, column]))
            )
            pebbles_orders = _orders_for_product(
                product=product,
                depth=depth,
                position=position,
                reservation=reservation,
                limit=self.asset_limit,
                passive_edge=self.passive_edge,
                take_edge=self.take_edge,
                take_size=min(self.take_size, PEBBLES_TAKE_SIZE),
            )
            if pebbles_orders:
                filtered_orders[product] = pebbles_orders
            else:
                filtered_orders.pop(product, None)

        return filtered_orders, conversions, trader_data


def _pebbles_group_residual_and_fair(
    state: TradingState,
) -> tuple[float | None, dict[str, float] | None]:
    current_fair: dict[str, float] = {}
    for product in PEBBLES_PRODUCTS:
        depth = state.order_depths.get(product)
        if depth is None:
            return None, None
        fair = _microprice(depth)
        if fair is None:
            return None, None
        current_fair[product] = fair

    total = sum(current_fair.values())
    residual = total - PEBBLES_TARGET_SUM
    synth_fair = {
        product: current_fair[product] - residual
        for product in PEBBLES_PRODUCTS
    }
    return residual, synth_fair

import json
import math
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState


# Names where the direct-product test showed V5 beats the mm_skew baseline.
USE_V5_PRODUCTS: set[str] = {
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "MICROCHIP_OVAL",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "PANEL_1X4",
    "PANEL_2X2",
    "PEBBLES_L",
    "PEBBLES_S",
    "ROBOT_IRONING",
    "UV_VISOR_AMBER",
}

# The blacklist has been explicitly emptied to prevent curve-fitting.
# We now allow the Base Trader to freely market-make across all eligible assets.
DISABLE_PRODUCTS: set[str] = set()


LIMIT = 10
PENNY_SIZE = 5
FAIR_ALPHA = 0.08
FAIR_EDGE = 1.0
INVENTORY_SKEW = 0.55
TREND_GATE = 18.0
FAIR_PRODUCTS = {"PEBBLES_L"}

SELECTED = [
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "MICROCHIP_OVAL",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "PANEL_1X4",
    "PANEL_2X2",
    "PANEL_2X4",
    "PEBBLES_L",
    "PEBBLES_S",
    "PEBBLES_XL",
    "ROBOT_DISHES",
    "ROBOT_IRONING",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_SUEDE",
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_VANILLA",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_VOID_BLUE",
    "UV_VISOR_AMBER",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
]

ML = {
    "OXYGEN_SHAKE_GARLIC": {
        "q": 0.60,
        "mean": [
            0.1818181818,
            0.3631363136,
            0.904490449,
            1.8104310431,
            14.0453045305,
            -0.0034726759,
            18.2591259126,
            -18.2854285429,
        ],
        "scale": [
            11.1317782543,
            15.7824207169,
            25.2089384524,
            35.8075244898,
            1.5074730353,
            0.2978849111,
            4.4281510056,
            4.381330654,
        ],
        "coef": [
            0.0065544877,
            0.0133072666,
            -0.0051592792,
            -0.0042823853,
            -0.0264587087,
            0.0199232035,
            0.1458620188,
            0.1409653857,
        ],
        "intercept": 0.0001534159,
    },
}

BASE_PRODUCTS = [
    product
    for product in SELECTED
    if product not in ML and product != "ROBOT_IRONING" and product not in FAIR_PRODUCTS
]


def best_bid_ask(depth: OrderDepth):
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid = max(depth.buy_orders)
    ask = min(depth.sell_orders)
    if bid >= ask:
        return None
    return bid, ask


def mid(depth: OrderDepth) -> Optional[float]:
    bba = best_bid_ask(depth)
    if bba is None:
        return None
    bid, ask = bba
    return (bid + ask) / 2.0


def load_data(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def dump_data(data: Dict) -> str:
    return json.dumps(data, separators=(",", ":"))


class V5PebblesAndStructureTrader:
    def position(self, state: TradingState, product: str) -> int:
        return int(state.position.get(product, 0))

    def remaining_buy(self, state: TradingState, product: str) -> int:
        return max(0, LIMIT - self.position(state, product))

    def remaining_sell(self, state: TradingState, product: str) -> int:
        return max(0, LIMIT + self.position(state, product))

    def used_order_position(self, orders: List[Order]) -> int:
        return sum(order.quantity for order in orders)

    def add_buy(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        product: str,
        price: int,
        size: int,
    ) -> None:
        orders = orders_by_product[product]
        available = self.remaining_buy(state, product) - max(0, self.used_order_position(orders))
        qty = min(size, available)
        if qty > 0:
            orders.append(Order(product, int(price), int(qty)))

    def add_sell(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        product: str,
        price: int,
        size: int,
    ) -> None:
        orders = orders_by_product[product]
        available = self.remaining_sell(state, product) + min(0, self.used_order_position(orders))
        qty = min(size, available)
        if qty > 0:
            orders.append(Order(product, int(price), -int(qty)))

    def trade_penny_products(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        products: List[str],
    ) -> None:
        for product in products:
            if product not in USE_V5_PRODUCTS:
                continue
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            bba = best_bid_ask(depth)
            if bba is None:
                continue
            bid, ask = bba
            if ask - bid > 1:
                buy_px = bid + 1
                sell_px = ask - 1
            else:
                buy_px = bid
                sell_px = ask
            self.add_buy(state, orders_by_product, product, buy_px, PENNY_SIZE)
            self.add_sell(state, orders_by_product, product, sell_px, PENNY_SIZE)

    def stable_fair(self, data: Dict, product: str, current_mid: float) -> float:
        key = "fair_" + product
        previous = data.get(key)
        fair = current_mid if previous is None else (1.0 - FAIR_ALPHA) * float(previous) + FAIR_ALPHA * current_mid
        data[key] = fair
        history_key = "fair_hist_" + product
        history = data.get(history_key, [])
        history.append(current_mid)
        history = history[-31:]
        data[history_key] = history
        return fair

    def trade_stable_fair_products(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        data: Dict,
        products,
    ) -> None:
        for product in products:
            if product not in USE_V5_PRODUCTS:
                continue
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            bba = best_bid_ask(depth)
            m = mid(depth)
            if bba is None or m is None:
                continue
            bid, ask = bba
            fair = self.stable_fair(data, product, m)
            history = data.get("fair_hist_" + product, [])
            trend = (history[-1] - history[0]) if len(history) >= 31 else 0.0
            pos = self.position(state, product)
            reservation = fair - INVENTORY_SKEW * pos
            buy_px = bid + 1 if ask - bid > 1 else bid
            sell_px = ask - 1 if ask - bid > 1 else ask
            if buy_px <= reservation - FAIR_EDGE and not (trend < -TREND_GATE and pos >= 0):
                self.add_buy(state, orders_by_product, product, buy_px, PENNY_SIZE)
            if sell_px >= reservation + FAIR_EDGE and not (trend > TREND_GATE and pos <= 0):
                self.add_sell(state, orders_by_product, product, sell_px, PENNY_SIZE)

    def features(self, mids: List[float], spread: int, obi: float, bidv: int, askv: int) -> List[float]:
        vals = []
        for lag in (1, 2, 5, 10):
            vals.append(mids[-1] - mids[-1 - lag] if len(mids) > lag else 0.0)
        vals.extend([spread, obi, bidv, askv])
        return vals

    def ml_probability(self, product: str, vals: List[float]) -> float:
        cfg = ML[product]
        z = cfg["intercept"]
        for x, mu, scale, coef in zip(vals, cfg["mean"], cfg["scale"], cfg["coef"]):
            z += ((x - mu) / scale) * coef
        return 1.0 / (1.0 + math.exp(-z))

    def trade_ml_product(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        data: Dict,
        product: str,
    ) -> None:
        if product not in USE_V5_PRODUCTS:
            return
        depth = state.order_depths.get(product)
        if depth is None:
            return
        bba = best_bid_ask(depth)
        m = mid(depth)
        if bba is None or m is None:
            return

        bid, ask = bba
        bidv = max(0, int(depth.buy_orders.get(bid, 0)))
        askv = max(0, -int(depth.sell_orders.get(ask, 0)))
        total = bidv + askv
        obi = (bidv - askv) / total if total > 0 else 0.0

        key = "ml_mid_" + product
        mids = data.get(key, [])
        mids.append(m)
        mids = mids[-11:]
        data[key] = mids
        if len(mids) < 11:
            return

        prob_up = self.ml_probability(product, self.features(mids, ask - bid, obi, bidv, askv))
        threshold = ML[product]["q"]
        buy_px = bid + 1 if ask - bid > 1 else bid
        sell_px = ask - 1 if ask - bid > 1 else ask

        if prob_up > threshold:
            self.add_buy(state, orders_by_product, product, buy_px, LIMIT)
        elif prob_up < 1.0 - threshold:
            self.add_sell(state, orders_by_product, product, sell_px, LIMIT)

    def trade_robot_mean_reversion(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        data: Dict,
    ) -> None:
        product = "ROBOT_IRONING"
        if product not in USE_V5_PRODUCTS:
            return
        depth = state.order_depths.get(product)
        if depth is None:
            return
        bba = best_bid_ask(depth)
        m = mid(depth)
        if bba is None or m is None:
            return

        key = "last_mid_" + product
        prev = data.get(key)
        data[key] = m
        if prev is None:
            return

        last_ret = m - float(prev)
        if abs(last_ret) < 0.5:
            return

        bid, ask = bba
        buy_px = bid + 1 if ask - bid > 1 else bid
        sell_px = ask - 1 if ask - bid > 1 else ask
        if last_ret > 0:
            self.add_sell(state, orders_by_product, product, sell_px, LIMIT)
        elif last_ret < 0:
            self.add_buy(state, orders_by_product, product, buy_px, LIMIT)

    def run(self, state: TradingState):
        data = load_data(state.traderData)
        orders_by_product: Dict[str, List[Order]] = {product: [] for product in state.order_depths}

        self.trade_penny_products(state, orders_by_product, BASE_PRODUCTS)
        self.trade_stable_fair_products(state, orders_by_product, data, FAIR_PRODUCTS)
        for product in ML:
            self.trade_ml_product(state, orders_by_product, data, product)
        self.trade_robot_mean_reversion(state, orders_by_product, data)

        filtered = {product: orders for product, orders in orders_by_product.items() if orders}
        return filtered, 0, dump_data(data)


class Trader:
    def __init__(self, *args, **kwargs) -> None:
        self.base = BaseTrader(*args, **kwargs)
        self.v5 = V5PebblesAndStructureTrader()

    def run(self, state: TradingState):
        raw = state.traderData or ""
        try:
            wrapper_data = json.loads(raw) if raw else {}
            if not isinstance(wrapper_data, dict):
                wrapper_data = {}
        except Exception:
            wrapper_data = {}

        state.traderData = wrapper_data.get("v5", "")
        v5_orders, _, v5_data = self.v5.run(state)
        state.traderData = raw

        base_orders, _, _ = self.base.run(state)

        result: Dict[str, List[Order]] = {}
        for product, orders in base_orders.items():
            if product not in USE_V5_PRODUCTS and product not in DISABLE_PRODUCTS:
                result[product] = orders

        for product in USE_V5_PRODUCTS:
            orders = v5_orders.get(product)
            if orders:
                result[product] = orders
            else:
                result.pop(product, None)

        for product in DISABLE_PRODUCTS:
            result.pop(product, None)

        return result, 0, json.dumps({"v5": v5_data}, separators=(",", ":"))

if __name__ == "__main__":
    pass