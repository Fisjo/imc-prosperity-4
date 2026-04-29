import json
import math
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState


POS_LIMIT = 10
MAX_BETA_JUMP_PCT = 2.0


# Selected strictly from Day 2 and Day 3 only.
# Day 4 was treated as out-of-sample and was not used for pair selection.
PAIR_CONFIGS = (
    {
        "name": "PANELS_2X2_4X4",
        "y": "PANEL_2X2",
        "x": "PANEL_4X4",
        "beta_init": -1.090508,
        "entry_z": 1.75,
        "exit_z": 0.25,
        "beta_win": 500,
        "z_win": 320,
        "beta_refresh": 50,
        "min_hist": 180,
        "step_qty": 2,
        "max_y": 6,
        "add_z": 0.60,
    },
    {
        "name": "ROBOTS_VACUUM_LAUNDRY",
        "y": "ROBOT_VACUUMING",
        "x": "ROBOT_LAUNDRY",
        "beta_init": 2.292020,
        "entry_z": 1.75,
        "exit_z": 0.25,
        "beta_win": 500,
        "z_win": 320,
        "beta_refresh": 50,
        "min_hist": 180,
        "step_qty": 2,
        "max_y": 4,
        "add_z": 0.60,
    },
)


MANAGED_PRODUCTS = tuple(
    sorted({cfg["y"] for cfg in PAIR_CONFIGS} | {cfg["x"] for cfg in PAIR_CONFIGS})
)
MAX_PRICE_WIN = max(cfg["beta_win"] for cfg in PAIR_CONFIGS)


def _get_depth(state: TradingState, product: str) -> Optional[OrderDepth]:
    return state.order_depths.get(product)


def _best_bid(depth: Optional[OrderDepth]) -> Optional[int]:
    return int(max(depth.buy_orders)) if depth and depth.buy_orders else None


def _best_ask(depth: Optional[OrderDepth]) -> Optional[int]:
    return int(min(depth.sell_orders)) if depth and depth.sell_orders else None


def _mid_price(depth: Optional[OrderDepth]) -> Optional[float]:
    bid = _best_bid(depth)
    ask = _best_ask(depth)
    if bid is None or ask is None:
        return None
    return 0.5 * (bid + ask)


def _push_bounded(values: List[float], value: float, limit: int) -> float:
    values.append(value)
    dropped = 0.0
    if len(values) > limit:
        dropped = float(values.pop(0))
    return dropped


def _tls_beta_2x2(y_vals: List[float], x_vals: List[float]) -> Optional[float]:
    """Closed-form TLS beta from the 2x2 demeaned covariance matrix."""
    n = min(len(y_vals), len(x_vals))
    if n < 30:
        return None

    y_slice = y_vals[-n:]
    x_slice = x_vals[-n:]
    mean_y = sum(y_slice) / n
    mean_x = sum(x_slice) / n

    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for y_val, x_val in zip(y_slice, x_slice):
        dx = x_val - mean_x
        dy = y_val - mean_y
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy

    disc = math.sqrt((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy)
    lam_min = 0.5 * ((sxx + syy) - disc)
    denom = lam_min - sxx
    if abs(denom) <= 1e-12:
        return None
    return -sxy / denom


def _update_target_y(current_target: int, zscore: float, cfg: dict, beta: float) -> int:
    """
    Stateful spread target:
    - Hold until the exit band is reached.
    - Scale in only when the same-side signal gets stronger.
    - Cap the Y leg so the implied X hedge also respects the hard limit.
    """
    if abs(zscore) < cfg["exit_z"]:
        return 0

    target = current_target

    if zscore > cfg["entry_z"]:
        levels = 1 + int((zscore - cfg["entry_z"]) / cfg["add_z"])
        desired = -min(cfg["max_y"], cfg["step_qty"] * levels)
        target = desired if current_target >= 0 else min(current_target, desired)
    elif zscore < -cfg["entry_z"]:
        levels = 1 + int(((-zscore) - cfg["entry_z"]) / cfg["add_z"])
        desired = min(cfg["max_y"], cfg["step_qty"] * levels)
        target = desired if current_target <= 0 else max(current_target, desired)

    # Clamp the Y target so the implied hedge leg also stays within +/-10.
    while target != 0 and abs(int(round(beta * target))) > POS_LIMIT:
        target += -1 if target > 0 else 1

    return target


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        td = json.loads(state.traderData) if state.traderData else {}
        price_state: Dict[str, List[float]] = td.get("prices", {})
        pair_state: Dict[str, dict] = td.get("pairs", {})

        depths = {product: _get_depth(state, product) for product in MANAGED_PRODUCTS}
        mids: Dict[str, float] = {}
        bids: Dict[str, Optional[int]] = {}
        asks: Dict[str, Optional[int]] = {}

        # Maintain one bounded mid-price history per product.
        for product in MANAGED_PRODUCTS:
            depth = depths[product]
            bids[product] = _best_bid(depth)
            asks[product] = _best_ask(depth)
            mid = _mid_price(depth)
            if mid is None:
                continue
            mids[product] = mid
            hist = price_state.get(product, [])
            hist.append(mid)
            if len(hist) > MAX_PRICE_WIN:
                hist.pop(0)
            price_state[product] = hist

        raw_targets: Dict[str, int] = {product: 0 for product in MANAGED_PRODUCTS}
        pair_targets: Dict[str, tuple[str, str, int, int]] = {}

        for cfg in PAIR_CONFIGS:
            y_name = cfg["y"]
            x_name = cfg["x"]
            if y_name not in mids or x_name not in mids:
                continue

            y_hist = price_state.get(y_name, [])
            x_hist = price_state.get(x_name, [])
            if not y_hist or not x_hist:
                continue

            ps = pair_state.get(
                cfg["name"],
                {
                    "beta": float(cfg["beta_init"]),
                    "ticks": 0,
                    "target_y": 0,
                    "spreads": [],
                    "spread_sum": 0.0,
                    "spread_sumsq": 0.0,
                },
            )

            ps["ticks"] += 1

            usable = min(cfg["beta_win"], len(y_hist), len(x_hist))
            if usable >= 100 and ps["ticks"] % cfg["beta_refresh"] == 0:
                new_beta = _tls_beta_2x2(y_hist[-usable:], x_hist[-usable:])
                if new_beta is not None:
                    old_beta = float(ps["beta"])
                    rel_jump = abs(new_beta - old_beta) / max(abs(old_beta), 1e-9)
                    if rel_jump <= MAX_BETA_JUMP_PCT:
                        ps["beta"] = float(new_beta)

            beta = float(ps["beta"])
            spread_now = mids[y_name] - beta * mids[x_name]

            spreads = ps["spreads"]
            dropped = _push_bounded(spreads, spread_now, cfg["z_win"])
            ps["spread_sum"] = float(ps["spread_sum"]) + spread_now - dropped
            ps["spread_sumsq"] = (
                float(ps["spread_sumsq"]) + spread_now * spread_now - dropped * dropped
            )

            spread_len = len(spreads)
            if spread_len >= cfg["min_hist"]:
                mean_spread = ps["spread_sum"] / spread_len
                variance = ps["spread_sumsq"] / spread_len - mean_spread * mean_spread
                sigma = math.sqrt(variance) if variance > 1e-9 else 1e-9
                zscore = (spread_now - mean_spread) / sigma
                ps["target_y"] = _update_target_y(int(ps["target_y"]), zscore, cfg, beta)

            target_y = int(ps["target_y"])
            target_x = -int(round(beta * target_y))

            raw_targets[y_name] += target_y
            raw_targets[x_name] += target_x
            pair_targets[cfg["name"]] = (y_name, x_name, target_y, target_x)
            pair_state[cfg["name"]] = ps

        # Portfolio-level netting and hard clamp.
        final_targets = {
            product: max(-POS_LIMIT, min(POS_LIMIT, raw_targets.get(product, 0)))
            for product in MANAGED_PRODUCTS
        }

        orders: dict[str, list[Order]] = {}

        for cfg in PAIR_CONFIGS:
            pair_key = cfg["name"]
            if pair_key not in pair_targets:
                continue

            y_name, x_name, _, _ = pair_targets[pair_key]
            current_y = state.position.get(y_name, 0)
            current_x = state.position.get(x_name, 0)
            target_y = final_targets[y_name]
            target_x = final_targets[x_name]
            delta_y = target_y - current_y
            delta_x = target_x - current_x

            if delta_y == 0 and delta_x == 0:
                continue

            # Never send the spread unless every required book side exists.
            y_price = asks[y_name] if delta_y > 0 else bids[y_name] if delta_y < 0 else None
            x_price = asks[x_name] if delta_x > 0 else bids[x_name] if delta_x < 0 else None

            if delta_y != 0 and y_price is None:
                continue
            if delta_x != 0 and x_price is None:
                continue

            # Final worst-case limit guard.
            if abs(current_y + delta_y) > POS_LIMIT:
                continue
            if abs(current_x + delta_x) > POS_LIMIT:
                continue

            if delta_y != 0:
                orders.setdefault(y_name, []).append(Order(y_name, int(y_price), int(delta_y)))
            if delta_x != 0:
                orders.setdefault(x_name, []).append(Order(x_name, int(x_price), int(delta_x)))

        td["prices"] = price_state
        td["pairs"] = pair_state
        trader_data = json.dumps(td, separators=(",", ":"))
        return orders, 0, trader_data
