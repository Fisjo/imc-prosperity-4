from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev

from prosperity4bt.models import ActivityLogRow, BacktestResult

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class RiskMetrics:
    final_pnl: float
    sharpe_ratio: float
    annualized_sharpe: float
    sortino_ratio: float
    max_drawdown_abs: float
    max_drawdown_pct: float
    calmar_ratio: float


def portfolio_pnl_by_timestamp(activity_logs: list[ActivityLogRow]) -> list[tuple[int, float]]:
    by_ts: dict[int, float] = defaultdict(float)
    for row in activity_logs:
        by_ts[row.timestamp] += float(row.columns[-1])
    return sorted(by_ts.items())


def equity_levels_from_activity(activity_logs: list[ActivityLogRow]) -> list[float]:
    return [v for _, v in portfolio_pnl_by_timestamp(activity_logs)]


def max_drawdown_from_levels(levels: list[float]) -> tuple[float, float]:
    if not levels:
        return 0.0, float("nan")
    hwm = levels[0]
    max_dd_abs = 0.0
    max_dd_pct = float("nan")
    for e in levels:
        hwm = max(hwm, e)
        dd_abs = hwm - e
        max_dd_abs = max(max_dd_abs, dd_abs)
        if hwm > 0:
            p = dd_abs / hwm
            max_dd_pct = p if math.isnan(max_dd_pct) else max(max_dd_pct, p)
    return max_dd_abs, max_dd_pct


def sharpe_from_returns(returns: list[float]) -> float:
    if len(returns) < 2:
        return float("nan")
    m = mean(returns)
    s = stdev(returns)
    if s == 0:
        return float("nan")
    return m / s


def sortino_from_returns(returns: list[float], target: float = 0.0) -> float:
    if len(returns) < 1:
        return float("nan")
    m = mean(returns)
    downside_sq_sum = sum(min(0.0, r - target) ** 2 for r in returns)
    d = math.sqrt(downside_sq_sum / len(returns))
    if d == 0:
        return float("nan") if m <= target else float("inf")
    return (m - target) / d


def calmar_from_pnl_and_drawdown(final_pnl: float, max_drawdown_abs: float) -> float:
    if max_drawdown_abs <= 0:
        return float("nan")
    return final_pnl / max_drawdown_abs


def annualized_sharpe_from_sample_sharpe(sample_sharpe: float) -> float:
    if math.isnan(sample_sharpe):
        return float("nan")
    return sample_sharpe * math.sqrt(TRADING_DAYS_PER_YEAR)


def stitched_equity_levels(results: list[BacktestResult]) -> list[float]:
    offset = 0.0
    out: list[float] = []
    for r in results:
        day_levels = equity_levels_from_activity(r.activity_logs)
        if not day_levels:
            continue
        shifted = [offset + x for x in day_levels]
        out.extend(shifted)
        offset = shifted[-1]
    return out


def _final_pnl_per_backtest_day(results: list[BacktestResult]) -> list[float]:
    pnls: list[float] = []
    for r in results:
        lv = equity_levels_from_activity(r.activity_logs)
        if lv:
            pnls.append(lv[-1])
    return pnls


def risk_metrics_full_period(results: list[BacktestResult]) -> RiskMetrics:
    """Sharpe/Sortino use one sample per backtest day; drawdown/Calmar use full stitched path."""
    stitched = stitched_equity_levels(results)
    if not stitched:
        return RiskMetrics(
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            float("nan"),
            float("nan"),
        )

    final_pnl = stitched[-1]
    max_dd_abs, max_dd_pct = max_drawdown_from_levels(stitched)

    day_pnls = _final_pnl_per_backtest_day(results)
    if len(day_pnls) >= 2:
        sharpe = sharpe_from_returns(day_pnls)
        sortino = sortino_from_returns(day_pnls)
    else:
        sharpe, sortino = float("nan"), float("nan")

    ann_sharpe = annualized_sharpe_from_sample_sharpe(sharpe)
    calmar = calmar_from_pnl_and_drawdown(final_pnl, max_dd_abs)
    return RiskMetrics(
        final_pnl=final_pnl,
        sharpe_ratio=sharpe,
        annualized_sharpe=ann_sharpe,
        sortino_ratio=sortino,
        max_drawdown_abs=max_dd_abs,
        max_drawdown_pct=max_dd_pct,
        calmar_ratio=calmar,
    )


def format_metric_value(x: float, *, int_style: bool = False) -> str:
    if math.isnan(x):
        return "n/a"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if int_style:
        return f"{x:,.0f}"
    return f"{x:,.4f}"


def format_risk_metrics_block(m: RiskMetrics, *, indent: str = "  ") -> str:
    if not math.isnan(m.max_drawdown_pct) and not math.isinf(m.max_drawdown_pct):
        dd_pct_display = format_metric_value(m.max_drawdown_pct)
    else:
        dd_pct_display = "n/a"
    lines = [
        f"{indent}final_pnl: {format_metric_value(m.final_pnl, int_style=True)}",
        f"{indent}sharpe_ratio: {format_metric_value(m.sharpe_ratio)}",
        f"{indent}annualized_sharpe: {format_metric_value(m.annualized_sharpe)}",
        f"{indent}sortino_ratio: {format_metric_value(m.sortino_ratio)}",
        f"{indent}max_drawdown_abs: {format_metric_value(m.max_drawdown_abs, int_style=True)}",
        f"{indent}max_drawdown_pct: {dd_pct_display}",
        f"{indent}calmar_ratio: {format_metric_value(m.calmar_ratio)}",
    ]
    return "\n".join(lines)
