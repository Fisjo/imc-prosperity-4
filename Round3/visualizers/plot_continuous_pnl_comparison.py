#!/usr/bin/env python3
"""
Generate continuous 30,000-tick PnL comparison plots across Round 3 days.

This script:
1) Finds the latest backtest run per strategy/day from R3_DATA/runs/*/metrics.json
2) Parses tick-by-tick `activitiesLog` from each run's submission.log
3) Builds continuous cumulative PnL (day1 starts at day0 end, day2 starts at day1 end)
4) Produces 3 comparison charts for:
   - VELVETFRUIT_EXTRACT
   - Total options (sum of all products starting with VEV_)
   - HYDROGEL_PACK

It supports composite strategy specs so you can create comparison lines without
new runs. Example:
V13_R3.py=velvet:V9_R3.py,options:V12_R3.py,hydrogel:V12_R3.py

Simple mode (recommended):
- Point to local `.py` strategy paths via `--strategy-paths ...`
- Or edit `DEFAULT_STRATEGY_PATHS` below and just run the script
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_DAYS = (0, 1, 2)
DAY_BOUNDARIES = (10_000, 20_000)
METRICS = ("velvet", "options", "hydrogel")

# Easiest workflow: edit this list with local .py paths, then run the script.
DEFAULT_STRATEGY_PATHS = [
    Path("C:/Users/Usuario/Desktop/prosperity-4/Round3/ntrader-v20.py")
]

# Optional fallback for path-based strategies that don't have their own runs yet.
# Key = plotted label (basename of strategy path). Values = metric source strategy files.
PATH_COMPOSITE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "V13_R3.py": {
        "velvet": "V9_R3.py",
        "options": "V12_R3.py",
        "hydrogel": "V12_R3.py",
    }
}


@dataclass(frozen=True)
class RunInfo:
    strategy_file: str
    day: int
    run_dir: Path
    generated_at: str
    run_id: str


@dataclass(frozen=True)
class StrategyRequest:
    label: str
    sources: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot continuous 30k-tick PnL comparison for Round 3 strategies."
    )
    parser.add_argument(
        "--strategies-path",
        type=Path,
        default=Path("/Users/davidmarco/Desktop/IMC Prosperity/Round 3/STRATEGIES"),
        help="Path containing strategy files (used for strategy names).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/Users/davidmarco/Desktop/IMC Prosperity/Round 3/R3_DATA"),
        help="Path containing `runs/` folder with metrics.json and submission.log.",
    )
    parser.add_argument(
        "--strategy-paths",
        nargs="+",
        default=DEFAULT_STRATEGY_PATHS,
        help=(
            "Local paths to strategy .py files. This is the simple mode. "
            "Each path is matched to runs by its basename (e.g., V10_R3.py)."
        ),
    )
    parser.add_argument(
        "--strategy-specs",
        nargs="+",
        default=None,
        help=(
            "Advanced mode (overrides --strategy-paths). Plain format: V10_R3.py "
            "or composite format: "
            "V13_R3.py=velvet:V9_R3.py,options:V12_R3.py,hydrogel:V12_R3.py"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Users/davidmarco/Desktop/IMC Prosperity/Round 3/reports/pnl_comparison_continuous_30k.png"
        ),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window in addition to saving.",
    )
    parser.add_argument(
        "--no-auto-backtest",
        action="store_true",
        help=(
            "Disable automatic local backtest fallback for strategies that do not have "
            "day 0/1/2 runs in R3_DATA/runs."
        ),
    )
    return parser.parse_args()


def parse_strategy_spec(spec: str) -> StrategyRequest:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty strategy spec provided.")

    if "=" not in spec:
        base = Path(spec).name
        return StrategyRequest(label=base, sources={m: base for m in METRICS})

    label_raw, mapping_raw = spec.split("=", 1)
    label = Path(label_raw.strip()).name
    if not label:
        raise ValueError(f"Invalid strategy label in spec: {spec}")

    sources: Dict[str, str] = {}
    for part in mapping_raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(
                f"Invalid composite entry '{piece}' in spec '{spec}'. Expected metric:strategy."
            )
        metric_raw, src_raw = piece.split(":", 1)
        metric = metric_raw.strip().lower()
        src = Path(src_raw.strip()).name
        if metric not in METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}' in spec '{spec}'. Allowed: {METRICS}"
            )
        if not src:
            raise ValueError(f"Missing source strategy for '{metric}' in spec '{spec}'")
        sources[metric] = src

    missing = [m for m in METRICS if m not in sources]
    if missing:
        raise ValueError(f"Spec '{spec}' missing metric source(s): {missing}")

    return StrategyRequest(label=label, sources=sources)


def request_from_strategy_path(path_str: str) -> StrategyRequest:
    path = Path(path_str).expanduser()
    label = path.name
    override = PATH_COMPOSITE_OVERRIDES.get(label)
    if override:
        return StrategyRequest(label=label, sources={m: override[m] for m in METRICS})
    return StrategyRequest(label=label, sources={m: label for m in METRICS})


def discover_runs(data_path: Path, strategy_files: List[str]) -> Dict[str, Dict[int, RunInfo]]:
    runs_root = data_path / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"`runs` directory not found: {runs_root}")

    targets = set(strategy_files)
    discovered: Dict[str, Dict[int, RunInfo]] = {s: {} for s in strategy_files}

    for metrics_file in runs_root.glob("*/metrics.json"):
        with metrics_file.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        trader_path = Path(metrics.get("trader_path", ""))
        strategy_file = trader_path.name
        day = int(metrics.get("day", -1))
        if strategy_file not in targets or day not in EXPECTED_DAYS:
            continue

        info = RunInfo(
            strategy_file=strategy_file,
            day=day,
            run_dir=metrics_file.parent,
            generated_at=str(metrics.get("generated_at", "")),
            run_id=str(metrics.get("run_id", metrics_file.parent.name)),
        )

        prev = discovered[strategy_file].get(day)
        if prev is None or info.generated_at > prev.generated_at:
            discovered[strategy_file][day] = info

    return discovered


def load_activities_csv(submission_log_path: Path) -> pd.DataFrame:
    with submission_log_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    activities = payload.get("activitiesLog", "")
    if not activities:
        raise RuntimeError(f"No `activitiesLog` found in {submission_log_path}")

    df = pd.read_csv(
        StringIO(activities),
        sep=";",
        usecols=["day", "timestamp", "product", "profit_and_loss"],
    )
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["profit_and_loss"] = pd.to_numeric(df["profit_and_loss"], errors="coerce")
    df = df.dropna(subset=["day", "timestamp", "product", "profit_and_loss"]).copy()
    df["day"] = df["day"].astype(int)
    return df


def _dashboard_simulator_module() -> Any:
    base_dir = Path(__file__).resolve().parents[1]
    sim_dir = base_dir / "IMC_Prosperity" / "dashboard"
    if not sim_dir.exists():
        raise FileNotFoundError(f"Dashboard simulator path not found: {sim_dir}")
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    import simulator  # type: ignore
    return simulator


def _load_trader_class_from_path(strategy_path: Path):
    module_name = (
        f"strategy_{strategy_path.stem}_{int(time.time_ns())}_{abs(hash(str(strategy_path)))}"
    )
    spec = importlib.util.spec_from_file_location(module_name, str(strategy_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import strategy from: {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trader_class = getattr(module, "Trader", None)
    if trader_class is None:
        raise RuntimeError(f"Strategy file does not define Trader class: {strategy_path}")
    return trader_class


def _load_round3_day_data(data_path: Path, day: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices_path = data_path / f"prices_round_3_day_{day}.csv"
    trades_path = data_path / f"trades_round_3_day_{day}.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing prices file for day {day}: {prices_path}")

    prices = pd.read_csv(prices_path, sep=";")
    prices["day"] = day

    if trades_path.exists():
        trades = pd.read_csv(trades_path, sep=";")
        trades["day"] = day
        if "symbol" not in trades.columns and "product" in trades.columns:
            trades["symbol"] = trades["product"]
    else:
        trades = pd.DataFrame(
            columns=["day", "timestamp", "symbol", "price", "quantity", "buyer", "seller"]
        )
    return prices, trades


def _apply_round3_position_limits(simulator_module: Any, products: List[str]) -> None:
    for product in products:
        if product in {"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"}:
            simulator_module.POSITION_LIMITS[product] = 200
        elif product.startswith("VEV_"):
            simulator_module.POSITION_LIMITS[product] = 300


def _simresult_to_day_pnl_df(
    prices_df: pd.DataFrame,
    sim_result: Any,
    day: int,
) -> pd.DataFrame:
    ts_sorted = sorted(prices_df["timestamp"].unique())
    mid_table = (
        prices_df.pivot_table(
            index="timestamp", columns="product", values="mid_price", aggfunc="last"
        )
        .reindex(ts_sorted)
        .ffill()
    )
    products = [p for p in mid_table.columns if isinstance(p, str)]

    rows: List[Dict[str, float]] = []
    for product in products:
        fills = sim_result.fills_by_product.get(product, [])
        cash_changes: Dict[int, float] = {}
        for fill in fills:
            ts = int(fill.timestamp)
            cash_changes[ts] = cash_changes.get(ts, 0.0) - float(fill.quantity) * float(fill.price)

        cash_series = (
            pd.Series(cash_changes, dtype=float).reindex(ts_sorted, fill_value=0.0).cumsum()
        )
        pos_map = {int(ts): int(pos) for ts, pos in sim_result.position_curve.get(product, [])}
        pos_series = (
            pd.Series(pos_map, dtype=float).reindex(ts_sorted).ffill().fillna(0.0)
        )
        mid_series = mid_table[product].astype(float).ffill().fillna(0.0)
        pnl_series = cash_series + pos_series * mid_series

        for ts, pnl in pnl_series.items():
            rows.append(
                {
                    "day": int(day),
                    "timestamp": int(ts),
                    "product": str(product),
                    "profit_and_loss": float(pnl),
                }
            )

    return pd.DataFrame(rows, columns=["day", "timestamp", "product", "profit_and_loss"])


def _run_local_day_backtest_to_pnl(
    strategy_path: Path,
    data_path: Path,
    day: int,
) -> pd.DataFrame:
    simulator = _dashboard_simulator_module()
    prices_df, trades_df = _load_round3_day_data(data_path, day)
    products = sorted(str(p) for p in prices_df["product"].unique())
    _apply_round3_position_limits(simulator, products)

    trader_class = _load_trader_class_from_path(strategy_path)
    trader = trader_class()
    sim_result = simulator.run_simulator(trader, prices_df, trades_df, products=products)
    return _simresult_to_day_pnl_df(prices_df, sim_result, day)


def load_strategy_day_pnls(
    source_strategy_file: str,
    discovered_runs: Dict[str, Dict[int, RunInfo]],
    strategy_path_lookup: Dict[str, Path],
    data_path: Path,
    auto_backtest_missing: bool,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, str]]:
    per_day: Dict[int, pd.DataFrame] = {}
    day_origin: Dict[int, str] = {}
    run_map = discovered_runs.get(source_strategy_file, {})

    for day in EXPECTED_DAYS:
        runinfo = run_map.get(day)
        if runinfo is not None:
            submission_path = runinfo.run_dir / "submission.log"
            day_df = load_activities_csv(submission_path)
            day_df = day_df[day_df["day"] == day][
                ["day", "timestamp", "product", "profit_and_loss"]
            ].copy()
            if day_df.empty:
                raise RuntimeError(f"No day {day} rows found in {submission_path}")
            per_day[day] = day_df
            day_origin[day] = f"run:{runinfo.run_id}"
            continue

        if not auto_backtest_missing:
            raise RuntimeError(
                f"Missing run for {source_strategy_file} day {day} and auto-backtest is disabled."
            )

        strategy_path = strategy_path_lookup.get(source_strategy_file)
        if strategy_path is None:
            raise RuntimeError(
                f"Missing run for {source_strategy_file} day {day}. "
                f"Provide this strategy in --strategy-paths to enable auto-backtest."
            )

        day_df = _run_local_day_backtest_to_pnl(strategy_path, data_path, day)
        if day_df.empty:
            raise RuntimeError(
                f"Local backtest produced no PnL rows for {source_strategy_file} day {day}."
            )
        per_day[day] = day_df
        day_origin[day] = f"local-backtest:{strategy_path.name}:day{day}"

    return per_day, day_origin


def series_from_day_df(
    day_df: pd.DataFrame,
    product_filter: Callable[[pd.Series], pd.Series],
) -> pd.Series:
    filtered = day_df.loc[product_filter(day_df["product"])].copy()
    if filtered.empty:
        raise RuntimeError("No rows matched the requested product filter for this day.")

    timestamps = sorted(filtered["timestamp"].unique())
    tick_map = {ts: i for i, ts in enumerate(timestamps)}
    n_ticks = len(timestamps)

    filtered["tick_in_day"] = filtered["timestamp"].map(tick_map)
    tick_pnl = (
        filtered.groupby("tick_in_day", sort=True)["profit_and_loss"]
        .sum()
        .reindex(range(n_ticks))
        .ffill()
        .fillna(0.0)
    )
    tick_pnl.index.name = "tick_in_day"
    return tick_pnl


def stitch_days_continuously(day_series: Dict[int, pd.Series]) -> pd.Series:
    continuous_parts: List[pd.Series] = []
    offset = 0.0
    global_tick_start = 0

    expected_ticks = len(day_series[0])
    for day in EXPECTED_DAYS:
        s = day_series[day].copy()
        if len(s) != expected_ticks:
            raise RuntimeError(
                f"Day {day} has {len(s)} ticks while day 0 has {expected_ticks}. "
                "Continuous stitching expects equal tick counts per day."
            )

        s = s + offset
        s.index = range(global_tick_start, global_tick_start + len(s))
        continuous_parts.append(s)

        offset = float(s.iloc[-1])
        global_tick_start += len(s)

    out = pd.concat(continuous_parts)
    out.index.name = "global_tick"
    return out


def build_continuous_metric(
    strategy_day_pnls: Dict[int, pd.DataFrame],
    metric_name: str,
) -> pd.Series:
    if metric_name == "velvet":
        flt = lambda products: products.eq("VELVETFRUIT_EXTRACT")
    elif metric_name == "options":
        flt = lambda products: products.str.startswith("VEV_")
    elif metric_name == "hydrogel":
        flt = lambda products: products.eq("HYDROGEL_PACK")
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    per_day: Dict[int, pd.Series] = {}
    for day in EXPECTED_DAYS:
        day_df = strategy_day_pnls[day]
        if day_df.empty:
            raise RuntimeError(f"No rows for day {day} in provided strategy day data.")
        per_day[day] = series_from_day_df(day_df, flt)

    return stitch_days_continuously(per_day)


def plot_comparison(
    data: Dict[str, Dict[str, pd.Series]],
    strategies_order: List[str],
    output_path: Path,
    show: bool = False,
) -> None:
    metric_titles = {
        "velvet": "Velvetfruit Extract Cumulative PnL",
        "options": "Total Options (All VEV_ Strikes) Cumulative PnL",
        "hydrogel": "Hydrogel Pack Cumulative PnL",
    }
    metric_ylabels = {
        "velvet": "PnL (VELVETFRUIT_EXTRACT)",
        "options": "PnL (sum of VEV_*)",
        "hydrogel": "PnL (HYDROGEL_PACK)",
    }
    strategy_labels = {s: s.replace(".py", "") for s in strategies_order}
    palette = plt.get_cmap("tab10")
    colors = {s: palette(i % 10) for i, s in enumerate(strategies_order)}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 14), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, ["velvet", "options", "hydrogel"]):
        for strategy in strategies_order:
            s = data[strategy][metric]
            ax.plot(
                s.index,
                s.values,
                label=strategy_labels[strategy],
                color=colors[strategy],
                linewidth=2.2,
                alpha=0.95,
            )

        for x in DAY_BOUNDARIES:
            ax.axvline(x=x, color="black", linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_title(metric_titles[metric], fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(metric_ylabels[metric], fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", ncol=min(6, len(strategies_order)), fontsize=9, frameon=True)

    total_ticks = len(next(iter(data.values()))["velvet"])
    axes[-1].set_xlabel("Continuous Tick (0 to 29,999)", fontsize=12)
    axes[-1].set_xlim(0, total_ticks - 1)
    axes[-1].set_xticks([0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000 - 1])
    axes[-1].set_xticklabels(["0", "5k", "10k", "15k", "20k", "25k", "30k"])

    fig.suptitle(
        "Round 3 Strategy Comparison: Continuous Cumulative PnL Across Days 0-2",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        y_annot = ymax - 0.08 * (ymax - ymin)
        ax.text(10_000 + 80, y_annot, "Day 1 Start", fontsize=9, color="black")
        ax.text(20_000 + 80, y_annot, "Day 2 Start", fontsize=9, color="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    auto_backtest_missing = not args.no_auto_backtest

    strategy_path_lookup: Dict[str, Path] = {}
    for path_str in args.strategy_paths:
        p = Path(path_str).expanduser()
        if p.exists():
            strategy_path_lookup[p.name] = p
        elif not args.strategy_specs:
            raise FileNotFoundError(f"Strategy path does not exist: {p}")

    if args.strategy_specs:
        requests = [parse_strategy_spec(spec) for spec in args.strategy_specs]
    else:
        requested_paths = [Path(p).expanduser() for p in args.strategy_paths]
        missing_paths = [p for p in requested_paths if not p.exists()]
        if missing_paths:
            missing_txt = "\n".join(f"  - {p}" for p in missing_paths)
            raise FileNotFoundError(f"These strategy paths do not exist:\n{missing_txt}")
        requests = [request_from_strategy_path(str(p)) for p in requested_paths]

    source_strategy_files = sorted(
        {src for req in requests for src in req.sources.values()}
    )
    discovered = discover_runs(args.data_path, source_strategy_files)

    source_day_origins: Dict[str, Dict[int, str]] = {}
    source_day_pnls: Dict[str, Dict[int, pd.DataFrame]] = {}
    for source in source_strategy_files:
        day_pnls, day_origin = load_strategy_day_pnls(
            source_strategy_file=source,
            discovered_runs=discovered,
            strategy_path_lookup=strategy_path_lookup,
            data_path=args.data_path,
            auto_backtest_missing=auto_backtest_missing,
        )
        source_day_pnls[source] = day_pnls
        source_day_origins[source] = day_origin

    metric_cache: Dict[str, Dict[str, pd.Series]] = {}
    for strategy_file in source_strategy_files:
        metric_cache[strategy_file] = {
            "velvet": build_continuous_metric(source_day_pnls[strategy_file], "velvet"),
            "options": build_continuous_metric(source_day_pnls[strategy_file], "options"),
            "hydrogel": build_continuous_metric(source_day_pnls[strategy_file], "hydrogel"),
        }

    all_data: Dict[str, Dict[str, pd.Series]] = {}
    plot_order: List[str] = []
    for req in requests:
        plot_order.append(req.label)
        all_data[req.label] = {
            metric: metric_cache[req.sources[metric]][metric] for metric in METRICS
        }

    plot_comparison(
        data=all_data,
        strategies_order=plot_order,
        output_path=args.output,
        show=args.show,
    )

    print(f"Saved plot to: {args.output}")
    print("Strategies included:")
    for req in requests:
        src_text = ", ".join(f"{m}:{req.sources[m]}" for m in METRICS)
        print(f"  - {req.label} <- {src_text}")
        unique_sources = sorted(set(req.sources.values()))
        for src in unique_sources:
            origin = source_day_origins[src]
            day_sources = [origin[d] for d in EXPECTED_DAYS]
            print(f"      source {src}: day0/day1/day2 -> {day_sources}")


if __name__ == "__main__":
    main()
