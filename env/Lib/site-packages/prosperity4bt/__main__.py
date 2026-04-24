import sys
from collections import defaultdict
from datetime import datetime
from functools import reduce
from importlib import import_module, metadata, reload
from math import log
from pathlib import Path
from typing import Annotated, Any, Optional

from typer import Argument, Option, Typer

from prosperity4bt.data import has_day_data
from prosperity4bt.file_reader import FileReader, FileSystemReader, PackageResourcesReader
from prosperity4bt.metrics import format_risk_metrics_block, risk_metrics_full_period
from prosperity4bt.models import BacktestResult, TradeMatchingMode
from prosperity4bt.open import open_visualizer
from prosperity4bt.runner import run_backtest

INVESTMENT_BUDGET_XIRECS = 50_000.0
MAX_PILLAR_PERCENT = 100.0


def parse_algorithm(algorithm: Path) -> Any:
    sys.path.append(str(algorithm.parent))

    from prosperity4bt import datamodel
    sys.modules["datamodel"] = datamodel

    return import_module(algorithm.stem)


def parse_limit_overrides(limit_opts: list[str]) -> Optional[dict[str, int]]:
    if len(limit_opts) == 0:
        return None
    out: dict[str, int] = {}
    for item in limit_opts:
        if ":" not in item:
            print(f"Error: --limit must be PRODUCT:NUMBER, got {item!r}")
            sys.exit(1)
        sym, num = item.split(":", 1)
        sym = sym.strip()
        num = num.strip()
        if not sym or not num:
            print(f"Error: invalid --limit {item!r}")
            sys.exit(1)
        try:
            out[sym] = int(num)
        except ValueError:
            print(f"Error: invalid limit number in {item!r}")
            sys.exit(1)
    return out


def parse_data(data_root: Optional[Path]) -> FileReader:
    if data_root is not None:
        return FileSystemReader(data_root)
    else:
        return PackageResourcesReader()


def parse_days(file_reader: FileReader, days: list[str]) -> list[tuple[int, int]]:
    parsed_days = []

    for arg in days:
        if "-" in arg:
            round_num, day_num = map(int, arg.split("-", 1))

            if not has_day_data(file_reader, round_num, day_num):
                print(f"Warning: no data found for round {round_num} day {day_num}")
                continue

            parsed_days.append((round_num, day_num))
        else:
            round_num = int(arg)

            parsed_days_in_round = []
            for day_num in range(-5, 100):
                if has_day_data(file_reader, round_num, day_num):
                    parsed_days_in_round.append((round_num, day_num))

            if len(parsed_days_in_round) == 0:
                print(f"Warning: no data found for round {round_num}")
                continue

            parsed_days.extend(parsed_days_in_round)

    if len(parsed_days) == 0:
        print("Error: did not find data for any requested round/day")
        sys.exit(1)

    return parsed_days


def parse_out(out: Optional[Path], no_out: bool) -> Optional[Path]:
    if out is not None:
        return out

    if no_out:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path.cwd() / "backtests" / f"{timestamp}.log"


def print_day_summary(result: BacktestResult) -> None:
    last_timestamp = result.activity_logs[-1].timestamp

    product_lines = []
    total_profit = 0

    for row in reversed(result.activity_logs):
        if row.timestamp != last_timestamp:
            break

        product = row.columns[2]
        profit = row.columns[-1]

        product_lines.append(f"{product}: {profit:,.0f}")
        total_profit += profit

    print(*reversed(product_lines), sep="\n")
    print(f"Total profit: {total_profit:,.0f}")


def merge_results(
    a: BacktestResult, b: BacktestResult, merge_profit_loss: bool, merge_timestamps: bool
) -> BacktestResult:
    sandbox_logs = a.sandbox_logs[:]
    activity_logs = a.activity_logs[:]
    trades = a.trades[:]

    if merge_timestamps:
        a_last_timestamp = a.activity_logs[-1].timestamp
        timestamp_offset = a_last_timestamp + 100
    else:
        timestamp_offset = 0

    sandbox_logs.extend([row.with_offset(timestamp_offset) for row in b.sandbox_logs])
    trades.extend([row.with_offset(timestamp_offset) for row in b.trades])

    if merge_profit_loss:
        profit_loss_offsets = defaultdict(float)
        for row in reversed(a.activity_logs):
            if row.timestamp != a_last_timestamp:
                break

            profit_loss_offsets[row.columns[2]] = row.columns[-1]

        activity_logs.extend(
            [row.with_offset(timestamp_offset, profit_loss_offsets[row.columns[2]]) for row in b.activity_logs]
        )
    else:
        activity_logs.extend([row.with_offset(timestamp_offset, 0) for row in b.activity_logs])

    return BacktestResult(a.round_num, a.day_num, sandbox_logs, activity_logs, trades)


def write_output(output_file: Path, merged_results: BacktestResult) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w+", encoding="utf-8") as file:
        file.write("Sandbox logs:\n")
        for row in merged_results.sandbox_logs:
            file.write(str(row))

        file.write("\n\n\nActivities log:\n")
        file.write(
            "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
        )
        file.write("\n".join(map(str, merged_results.activity_logs)))

        file.write("\n\n\n\n\nTrade History:\n")
        file.write("[\n")
        file.write(",\n".join(map(str, merged_results.trades)))
        file.write("]")


def print_overall_summary(results: list[BacktestResult]) -> None:
    print("Profit summary:")

    total_profit = 0
    for result in results:
        last_timestamp = result.activity_logs[-1].timestamp

        profit = 0
        for row in reversed(result.activity_logs):
            if row.timestamp != last_timestamp:
                break

            profit += row.columns[-1]

        print(f"Round {result.round_num} day {result.day_num}: {profit:,.0f}")
        total_profit += profit

    print(f"Total profit: {total_profit:,.0f}")


def _extract_result_profit(result: BacktestResult) -> float:
    last_timestamp = result.activity_logs[-1].timestamp
    return sum(row.columns[-1] for row in reversed(result.activity_logs) if row.timestamp == last_timestamp)


def _safe_int_bid(raw_bid: Any) -> int:
    try:
        bid = int(raw_bid)
    except (TypeError, ValueError):
        bid = 0
    return max(0, bid)


def _calculate_speed_multiplier_from_rank(rank: int, player_count: int) -> float:
    if player_count <= 1:
        return 0.9
    # Linear interpolation from rank 1 -> 0.9 down to rank N -> 0.1
    return 0.9 - ((rank - 1) * (0.8 / (player_count - 1)))


def _research_value(research_pct: float) -> float:
    return 200_000.0 * log(1.0 + research_pct) / log(1.0 + MAX_PILLAR_PERCENT)


def _scale_value(scale_pct: float) -> float:
    return 7.0 * (scale_pct / MAX_PILLAR_PERCENT)


def format_path(path: Path) -> str:
    cwd = Path.cwd()
    if path.is_relative_to(cwd):
        return str(path.relative_to(cwd))
    else:
        return str(path)


def version_callback(value: bool) -> None:
    if value:
        print(f"prosperity4btest {metadata.version(__package__)}")
        sys.exit(0)


app = Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    algorithm: Annotated[Path, Argument(help="Path to the Python file containing the algorithm to backtest.", show_default=False, exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    days: Annotated[list[str], Argument(help="The days to backtest on. <round>-<day> for a single day, <round> for all days in a round.", show_default=False)],
    merge_pnl: Annotated[bool, Option("--merge-pnl", help="Merge profit and loss across days.")] = False,
    vis: Annotated[bool, Option("--vis", help="Open backtest results in https://jmerle.github.io/imc-prosperity-3-visualizer/ when done.")] = False,
    out: Annotated[Optional[Path], Option(help="File to save output log to (defaults to backtests/<timestamp>.log).", show_default=False, dir_okay=False, resolve_path=True)] = None,
    no_out: Annotated[bool, Option("--no-out", help="Skip saving output log.")] = False,
    data: Annotated[Optional[Path], Option(help="Path to data directory. Must look similar in structure to https://github.com/nabayansaha/imc-prosperity-4-backtester/tree/master/prosperity4bt/resources.", show_default=False, exists=True, file_okay=False, dir_okay=True, resolve_path=True)] = None,
    print_output: Annotated[bool, Option("--print", help="Print the trader's output to stdout while it's running.")] = False,
    match_trades: Annotated[TradeMatchingMode, Option(help="How to match orders against market trades. 'all' matches trades with prices equal to or worse than your quotes, 'worse' matches trades with prices worse than your quotes, 'none' does not match trades against orders at all.")] = TradeMatchingMode.all,
    no_progress: Annotated[bool, Option("--no-progress", help="Don't show progress bars.")] = False,
    original_timestamps: Annotated[bool, Option("--original-timestamps", help="Preserve original timestamps in output log rather than making them increase across days.")] = False,
    limit: Annotated[
        list[str],
        Option(
            "--limit",
            help="Override position limit for a product (PRODUCT:LIMIT). Repeat for multiple products.",
            show_default=False,
        ),
    ] = [],
    round2_access: Annotated[
        str,
        Option(
            "--round2-access",
            help=(
                "How to treat Trader.bid() for round 2 in reported adjusted PnL: "
                "'unknown' (default, no deduction), 'accepted' (subtract bid), "
                "'rejected' (do not subtract bid)."
            ),
            case_sensitive=False,
        ),
    ] = "unknown",
    version: Annotated[bool, Option("--version", "-v", help="Show the program's version number and exit.", is_eager=True, callback=version_callback)] = False,
) -> None:  # fmt: skip
    if out is not None and no_out:
        print("Error: --out and --no-out are mutually exclusive")
        sys.exit(1)

    try:
        trader_module = parse_algorithm(algorithm)
    except ModuleNotFoundError as e:
        print(f"{algorithm} is not a valid algorithm file: {e}")
        sys.exit(1)

    if not hasattr(trader_module, "Trader"):
        print(f"{algorithm} does not expose a Trader class")
        sys.exit(1)

    file_reader = parse_data(data)
    parsed_days = parse_days(file_reader, days)
    output_file = parse_out(out, no_out)

    show_progress_bars = not no_progress and not print_output
    limits_override = parse_limit_overrides(limit)
    round2_access_normalized = round2_access.lower()
    if round2_access_normalized not in {"unknown", "accepted", "rejected"}:
        print("Error: --round2-access must be one of: unknown, accepted, rejected")
        sys.exit(1)

    trader_bid = 0
    if hasattr(trader_module.Trader, "bid"):
        try:
            trader_bid = _safe_int_bid(trader_module.Trader().bid())
        except Exception as e:
            print(f"Warning: failed to call Trader.bid(); defaulting to 0 ({e})")
            trader_bid = 0
    if trader_bid > 0:
        print(f"Trader Market Access Fee bid: {trader_bid:,d}")

    results = []
    for round_num, day_num in parsed_days:
        print(f"Backtesting {algorithm} on round {round_num} day {day_num}")

        reload(trader_module)

        result = run_backtest(
            trader_module.Trader(),
            file_reader,
            round_num,
            day_num,
            print_output,
            match_trades,
            True,
            show_progress_bars,
            limits_override,
        )

        print_day_summary(result)
        if len(parsed_days) > 1:
            print()

        results.append(result)

    if len(parsed_days) > 1:
        print_overall_summary(results)

    print()
    full_metrics = risk_metrics_full_period(results)
    print("Risk metrics (full trading period):")
    print(format_risk_metrics_block(full_metrics))
    round2_profit = sum(_extract_result_profit(r) for r in results if r.round_num == 2)
    if round2_profit != 0 or any(r.round_num == 2 for r in results):
        print("\nRound 2 fee-aware summary:")
        print(f"  round2_profit_before_maf: {round2_profit:,.0f}")
        print(f"  bid(): {trader_bid:,.0f}")
        if round2_access_normalized == "accepted":
            adjusted_round2_profit = round2_profit - trader_bid
            print("  assumed_access: accepted")
            print(f"  round2_profit_after_maf: {adjusted_round2_profit:,.0f}")
        elif round2_access_normalized == "rejected":
            print("  assumed_access: rejected")
            print(f"  round2_profit_after_maf: {round2_profit:,.0f}")
        else:
            print("  assumed_access: unknown")
            print("  round2_profit_after_maf: n/a (set --round2-access accepted|rejected)")

    if output_file is not None:
        merged_results = reduce(lambda a, b: merge_results(a, b, merge_pnl, not original_timestamps), results)
        write_output(output_file, merged_results)
        print(f"\nSuccessfully saved backtest results to {format_path(output_file)}")

    if vis and output_file is not None:
        open_visualizer(output_file)


def main() -> None:
    app()


@app.command("invest")
def invest_cli(
    research: Annotated[float, Option("--research", min=0.0, max=100.0, help="Research allocation percentage (0..100).")],
    scale: Annotated[float, Option("--scale", min=0.0, max=100.0, help="Scale allocation percentage (0..100).")],
    speed: Annotated[float, Option("--speed", min=0.0, max=100.0, help="Speed allocation percentage (0..100).")],
    speed_multiplier: Annotated[
        Optional[float],
        Option(
            "--speed-multiplier",
            min=0.1,
            max=0.9,
            help="Optional direct speed multiplier estimate (0.1..0.9).",
        ),
    ] = None,
    speed_rank: Annotated[
        Optional[int],
        Option("--speed-rank", min=1, help="Optional rank of your speed allocation (1 is highest)."),
    ] = None,
    player_count: Annotated[
        Optional[int],
        Option("--players", min=1, help="Optional number of participating players for rank-based speed multiplier."),
    ] = None,
) -> None:
    total_alloc = research + scale + speed
    if total_alloc > 100.0:
        print(
            f"Error: total allocation exceeds 100% (research+scale+speed={total_alloc:.2f}%). "
            "Reduce allocations."
        )
        sys.exit(1)

    used_budget = INVESTMENT_BUDGET_XIRECS * (total_alloc / 100.0)
    research_out = _research_value(research)
    scale_out = _scale_value(scale)

    if speed_multiplier is not None and (speed_rank is not None or player_count is not None):
        print("Error: choose either --speed-multiplier or (--speed-rank with --players), not both.")
        sys.exit(1)

    if speed_multiplier is None:
        if speed_rank is not None or player_count is not None:
            if speed_rank is None or player_count is None:
                print("Error: both --speed-rank and --players are required for rank-based speed multiplier.")
                sys.exit(1)
            if speed_rank > player_count:
                print("Error: --speed-rank cannot be greater than --players.")
                sys.exit(1)
            speed_multiplier = _calculate_speed_multiplier_from_rank(speed_rank, player_count)
        else:
            speed_multiplier = 0.5

    gross_pnl = research_out * scale_out * speed_multiplier
    net_pnl = gross_pnl - used_budget

    print("Investment outcome:")
    print(f"  allocations_pct: research={research:.2f}, scale={scale:.2f}, speed={speed:.2f}")
    print(f"  budget_used: {used_budget:,.2f}")
    print(f"  research_output: {research_out:,.2f}")
    print(f"  scale_output: {scale_out:,.4f}")
    print(f"  speed_multiplier: {speed_multiplier:.4f}")
    print(f"  gross_pnl: {gross_pnl:,.2f}")
    print(f"  net_pnl: {net_pnl:,.2f}")


if __name__ == "__main__":
    main()
