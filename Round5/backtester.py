#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
#
#   IMC Prosperity 4 — Round 5  |  Offline Backtester  v2.0
#   ─────────────────────────────────────────────────────────────────────────
#   Architecture inspired by jmerle/imc-prosperity-4-backtester, rebuilt
#   from the ground up for Round 5 with full matching engine & risk metrics.
#
#   v2.0 upgrades
#   ─────────────────────────────────────────────────────────────────────────
#   FEATURE 1 — --strategy <file>  : load any arbitrary strategy file by path
#   FEATURE 2 — per-day PnL table  : asset-level PnL printed after every day
#   FEATURE 3 — per-day risk block : Sharpe / Sortino / Drawdown per day
#   FEATURE 4 — dynamic log naming : log file named after the strategy file
#
#   Directory layout expected:
#
#       .
#       ├── backtester.py               ← this file
#       ├── datamodel.py                ← official IMC datamodel (or the stub)
#       ├── trader.py                   ← default strategy (or pass --strategy)
#       └── data/
#           ├── prices_round_5_day_2.csv
#           ├── prices_round_5_day_3.csv
#           ├── prices_round_5_day_4.csv
#           ├── trades_round_5_day_2.csv  (optional but recommended)
#           ├── trades_round_5_day_3.csv
#           └── trades_round_5_day_4.csv
#
#   Usage:
#       # ── FEATURE 1: pass any strategy by filename ──────────────────────
#       python backtester.py --strategy v26_no_fv.py
#       python backtester.py --strategy ntrader-round3-v5.py --days 2 3
#
#       # ── Log generation is OPT-IN (--gen-log flag) ─────────────────────
#       #   Default  : no log file written — simulation + terminal metrics only
#       #   Opt-in   : --gen-log  →  writes  <strategy_basename>.log
#       #   Override : --gen-log --log custom_name.log
#
#       python backtester.py --strategy v26.py                  # fast, no log
#       python backtester.py --strategy v26.py --gen-log        # write v26.log
#       python backtester.py --strategy v26.py --gen-log --log out.log
#       python backtester.py --days 2 3                         # specific days
#       python backtester.py --data-dir ./mydata                # custom folder
#       python backtester.py --all-products                     # all 50 in table
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import sys, os, json, math, time, argparse, importlib.util, io, traceback, re
from copy        import deepcopy
from typing      import Dict, List, Optional, Tuple, Any, DefaultDict
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import numpy  as np
from tqdm import tqdm

# ── datamodel import ──────────────────────────────────────────────────────────
try:
    from datamodel import (
        TradingState, OrderDepth, Order, Trade, Listing, Observation,
    )
    try:
        from datamodel import ConversionObservation
    except ImportError:
        ConversionObservation = None          # older datamodel stubs
except ImportError as _dm_err:
    print(f"✗  Cannot import datamodel: {_dm_err}")
    print("   Place datamodel.py in the same directory as backtester.py")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

POSITION_LIMIT : int = 10          # Round 5: ALL 50 products
TICK_SIZE       : int = 100        # timestamps go 0, 100, 200 … 999900
TICKS_PER_DAY  : int = 10_000     # 0 → 999900 inclusive
OUR_NAME        : str = "SUBMISSION"
SEP             : str = ";"


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 1 — DATA INGESTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Parses pricing and trading CSVs into in-memory lookup structures.

    price_map  : {(day, ts): {product: row_as_dict}}
    trade_map  : {(day, ts): [trade_row_as_dict, ...]}
    ts_by_day  : {day: [sorted list of timestamps]}
    products   : sorted list of all unique product symbols
    """

    PRICE_DTYPES = {
        "day": "int32", "timestamp": "int32",
        "bid_price_1": "float32", "bid_volume_1": "float32",
        "bid_price_2": "float32", "bid_volume_2": "float32",
        "bid_price_3": "float32", "bid_volume_3": "float32",
        "ask_price_1": "float32", "ask_volume_1": "float32",
        "ask_price_2": "float32", "ask_volume_2": "float32",
        "ask_price_3": "float32", "ask_volume_3": "float32",
        "mid_price": "float32",   "profit_and_loss": "float32",
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    # ─────────────────────────────────────────────────────────────────────────
    def load_days(
        self, days: List[int]
    ) -> Tuple[
        Dict[Tuple[int,int], Dict[str, dict]],
        Dict[Tuple[int,int], List[dict]],
        Dict[int, List[int]],
        List[str],
    ]:
        price_frames: List[pd.DataFrame] = []
        trade_frames: List[pd.DataFrame] = []

        for day in days:
            pf = self._load_price_file(day)
            if pf is not None:
                price_frames.append(pf)

            tf = self._load_trade_file(day)
            if tf is not None:
                trade_frames.append(tf)

        if not price_frames:
            raise FileNotFoundError(
                f"No price CSV files found in '{self.data_dir}' for days {days}."
            )

        prices_df = pd.concat(price_frames, ignore_index=True)

        # ── Build price map ───────────────────────────────────────────────
        price_map:  Dict[Tuple[int,int], Dict[str, dict]] = defaultdict(dict)
        ts_by_day:  Dict[int, set] = defaultdict(set)

        # Vectorised conversion → dict of lists → row iteration
        records = prices_df.to_dict("records")
        for row in records:
            day = int(row["day"])
            ts  = int(row["timestamp"])
            price_map[(day, ts)][row["product"]] = row
            ts_by_day[day].add(ts)

        ts_by_day = {d: sorted(ts) for d, ts in ts_by_day.items()}

        # ── Build trade map ───────────────────────────────────────────────
        trade_map: Dict[Tuple[int,int], List[dict]] = defaultdict(list)
        if trade_frames:
            trades_df = pd.concat(trade_frames, ignore_index=True)
            for row in trades_df.to_dict("records"):
                day = int(row.get("day", days[0]))
                ts  = int(row["timestamp"])
                trade_map[(day, ts)].append(row)

        # ── All unique products ───────────────────────────────────────────
        products = sorted(prices_df["product"].unique().tolist())

        return dict(price_map), dict(trade_map), ts_by_day, products

    # ─────────────────────────────────────────────────────────────────────────
    def _load_price_file(self, day: int) -> Optional[pd.DataFrame]:
        path = os.path.join(self.data_dir, f"prices_round_5_day_{day}.csv")
        if not os.path.exists(path):
            print(f"  ⚠  {path} not found — skipping day {day}")
            return None
        df = pd.read_csv(path, sep=SEP, low_memory=False)
        df["day"] = day
        # Coerce all numeric columns; keep 'product' as string
        for col in df.columns:
            if col != "product":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _load_trade_file(self, day: int) -> Optional[pd.DataFrame]:
        path = os.path.join(self.data_dir, f"trades_round_5_day_{day}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, sep=SEP, low_memory=False)
        df["day"] = day
        for col in ["timestamp", "price", "quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 2 — ORDER BOOK & MATCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BookSnapshot:
    """
    Immutable snapshot of one product's order book at one timestamp.
    Constructed once per tick from the CSV row.
    """
    product:   str
    bids:      Dict[int, int]          # {price: volume} — sorted descending
    asks:      Dict[int, int]          # {price: volume} — sorted ascending
    mid_price: Optional[float] = None

    # ── Factory ──────────────────────────────────────────────────────────────
    @classmethod
    def from_csv_row(cls, row: dict) -> "BookSnapshot":
        bids: Dict[int,int] = {}
        asks: Dict[int,int] = {}

        for lvl in (1, 2, 3):
            bp = row.get(f"bid_price_{lvl}")
            bv = row.get(f"bid_volume_{lvl}")
            ap = row.get(f"ask_price_{lvl}")
            av = row.get(f"ask_volume_{lvl}")
            if bp is not None and not math.isnan(float(bp)) and \
               bv is not None and not math.isnan(float(bv)) and int(bv) > 0:
                bids[int(bp)] = int(bv)
            if ap is not None and not math.isnan(float(ap)) and \
               av is not None and not math.isnan(float(av)) and int(av) > 0:
                asks[int(ap)] = int(av)

        mid = row.get("mid_price")
        if mid is None or (isinstance(mid, float) and math.isnan(mid)):
            mid = ((max(bids) + min(asks)) / 2.0) if bids and asks else None
        else:
            mid = float(mid)

        return cls(product=row["product"], bids=bids, asks=asks, mid_price=mid)

    # ── Convert to datamodel.OrderDepth ──────────────────────────────────────
    def to_order_depth(self) -> OrderDepth:
        depth = OrderDepth()
        depth.buy_orders  = dict(self.bids)   # {price: +vol}
        depth.sell_orders = dict(self.asks)   # {price: +vol}
        return depth


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fill:
    """Records a single matched trade from our algorithm's orders."""
    product:   str
    price:     int
    quantity:  int      # positive = bought, negative = sold
    timestamp: int
    day:       int = 0


# ─────────────────────────────────────────────────────────────────────────────

class MatchingEngine:
    """
    Replicates IMC exchange fill logic.

    Rule 1 — Limit Breach (all-or-nothing):
        Compute the theoretical maximum long and minimum short positions that
        would result if EVERY order for a product were fully filled.
        If max_pos > +LIMIT  OR  min_pos < -LIMIT  →  reject ALL orders
        for that product this tick.  Nothing is partially accepted.

    Rule 2 — Passive Fill Price:
        Fills occur at the BOOK price, not the order price.
        (Trader bids 105 against an ask at 100 → fills at 100.)

    Rule 3 — Volume Depletion:
        Each fill subtracts from the working copy of the book so one tick's
        order set cannot consume more volume than the book contains.
    """

    def __init__(self, limit: int = POSITION_LIMIT):
        self.limit = limit

    # ─────────────────────────────────────────────────────────────────────────
    def process(
        self,
        orders:    Dict[str, List[Order]],
        books:     Dict[str, BookSnapshot],
        positions: Dict[str, int],
        timestamp: int,
        day:       int,
    ) -> Tuple[List[Fill], Dict[str, str]]:
        """
        Parameters
        ----------
        orders    : product → list of Order objects from Trader.run()
        books     : product → BookSnapshot at this timestamp
        positions : product → current held position (before this tick)
        timestamp : current tick timestamp
        day       : current day number

        Returns
        -------
        fills      : list of Fill objects (executed trades)
        rejections : product → human-readable rejection reason
        """
        fills:      List[Fill]       = []
        rejections: Dict[str, str]   = {}

        for product, order_list in orders.items():
            if not order_list:
                continue

            cur_pos = positions.get(product, 0)

            # ── Rule 1: Limit Breach Check ───────────────────────────────────
            total_buy  = sum(o.quantity for o in order_list if o.quantity > 0)
            total_sell = sum(o.quantity for o in order_list if o.quantity < 0)

            max_theoretical = cur_pos + total_buy    # if every buy fills
            min_theoretical = cur_pos + total_sell   # if every sell fills (sells are neg)

            if max_theoretical > self.limit or min_theoretical < -self.limit:
                rejections[product] = (
                    f"pos={cur_pos:+d}  "
                    f"max_if_all_buy={max_theoretical:+d}  "
                    f"min_if_all_sell={min_theoretical:+d}  "
                    f"limit=±{self.limit}  →  ALL {len(order_list)} order(s) REJECTED"
                )
                continue

            # ── Get mutable working copy of the book ────────────────────────
            book = books.get(product)
            if book is None:
                continue

            # Work on mutable copies so volume depletion is local to this tick
            working_bids: Dict[int,int] = dict(book.bids)
            working_asks: Dict[int,int] = dict(book.asks)

            # ── Rule 2 & 3: Match orders against book ───────────────────────
            for order in order_list:
                remaining = abs(order.quantity)
                is_buy    = order.quantity > 0

                if is_buy:
                    # BUY: walk asks ascending (cheapest first)
                    for ask_px in sorted(working_asks.keys()):
                        if remaining <= 0:
                            break
                        if order.price < ask_px:
                            # Trader's bid is below this ask → no fill
                            break

                        avail   = working_asks[ask_px]
                        fill_q  = min(remaining, avail)
                        if fill_q > 0:
                            fills.append(Fill(
                                product=product, price=ask_px,
                                quantity=+fill_q,
                                timestamp=timestamp, day=day,
                            ))
                            working_asks[ask_px] -= fill_q
                            remaining            -= fill_q
                            if working_asks[ask_px] <= 0:
                                del working_asks[ask_px]

                else:
                    # SELL: walk bids descending (richest first)
                    for bid_px in sorted(working_bids.keys(), reverse=True):
                        if remaining <= 0:
                            break
                        if order.price > bid_px:
                            # Trader's ask is above this bid → no fill
                            break

                        avail  = working_bids[bid_px]
                        fill_q = min(remaining, avail)
                        if fill_q > 0:
                            fills.append(Fill(
                                product=product, price=bid_px,
                                quantity=-fill_q,
                                timestamp=timestamp, day=day,
                            ))
                            working_bids[bid_px] -= fill_q
                            remaining            -= fill_q
                            if working_bids[bid_px] <= 0:
                                del working_bids[bid_px]

        return fills, rejections


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3A — STATE CONSTRUCTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def build_trading_state(
    trader_data:   str,
    listings:      Dict[str, Listing],
    books:         Dict[str, BookSnapshot],
    own_trades:    Dict[str, List[Trade]],
    market_trades: Dict[str, List[Trade]],
    positions:     Dict[str, int],
    timestamp:     int,
) -> TradingState:
    """
    Construct a TradingState compatible with the official IMC datamodel.
    Handles both constructor signatures seen across competition seasons.
    """
    order_depths = {sym: snap.to_order_depth() for sym, snap in books.items()}

    try:
        obs = Observation(
            plainValueObservations={},
            conversionObservations={},
        )
    except Exception:
        obs = None

    # Try keyword-argument constructor (IMC Prosperity 3 style)
    try:
        state = TradingState(
            traderData    = trader_data,
            listings      = listings,
            order_depths  = order_depths,
            own_trades    = own_trades,
            market_trades = market_trades,
            position      = dict(positions),
            observations  = obs,
        )
    except TypeError:
        # Fall back to positional constructor (older seasons)
        try:
            state = TradingState(
                trader_data, listings, order_depths,
                own_trades, market_trades, dict(positions), obs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot construct TradingState — check datamodel.py: {exc}"
            ) from exc

    # Always set timestamp as an attribute (some constructors don't take it)
    state.timestamp = timestamp
    return state


def build_market_trades(
    trade_rows: List[dict],
) -> Dict[str, List[Trade]]:
    """
    Convert raw trade CSV rows into market_trades dict for TradingState.
    These represent trades between other market participants, NOT our fills.
    """
    result: Dict[str, List[Trade]] = defaultdict(list)
    for row in trade_rows:
        sym = row.get("symbol", "")
        if not sym:
            continue
        px  = row.get("price",    0)
        qty = row.get("quantity", 0)
        if pd.isna(px) or pd.isna(qty):
            continue
        result[sym].append(Trade(
            symbol    = sym,
            price     = int(float(px)),
            quantity  = int(float(qty)),
            buyer     = str(row.get("buyer",  "") or ""),
            seller    = str(row.get("seller", "") or ""),
            timestamp = int(row.get("timestamp", 0)),
        ))
    return dict(result)


def fills_to_own_trades(fills: List[Fill]) -> Dict[str, List[Trade]]:
    """Convert Fill objects to the own_trades dict format."""
    result: Dict[str, List[Trade]] = defaultdict(list)
    for f in fills:
        if f.quantity > 0:
            buyer, seller = OUR_NAME, "MARKET"
        else:
            buyer, seller = "MARKET", OUR_NAME
        result[f.product].append(Trade(
            symbol    = f.product,
            price     = f.price,
            quantity  = abs(f.quantity),
            buyer     = buyer,
            seller    = seller,
            timestamp = f.timestamp,
        ))
    return dict(result)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3B — PER-PRODUCT PnL TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class PnLTracker:
    """
    Tracks cash and position per product; computes mark-to-market PnL.

    cash[product]    = Σ(fill_price × −fill_qty)   [selling earns, buying costs]
    position[product]= current held inventory
    pnl[product]     = cash[product] + position[product] × mid_price
    """

    def __init__(self, products: List[str]):
        self.cash:     Dict[str, float] = defaultdict(float)
        self.position: Dict[str, int]   = defaultdict(int)
        self.products  = products

        # Tick-level total PnL history (for Sharpe / Sortino / drawdown)
        self.pnl_history: List[float] = []

    # ─────────────────────────────────────────────────────────────────────────
    def apply_fills(self, fills: List[Fill]) -> None:
        """Update cash and position from this tick's matched trades."""
        for f in fills:
            # Positive quantity (buy)  → pay cash
            # Negative quantity (sell) → receive cash
            self.cash[f.product]    -= f.price * f.quantity
            self.position[f.product] += f.quantity

    # ─────────────────────────────────────────────────────────────────────────
    def mark_to_market(
        self, mid_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute per-product MTM PnL and record the total for risk metric calc.
        Returns {product: pnl_value}.
        """
        product_pnl: Dict[str, float] = {}
        total = 0.0

        for prod in self.products:
            mid = mid_prices.get(prod)
            pos = self.position.get(prod, 0)
            csh = self.cash.get(prod, 0.0)

            if mid is not None and not math.isnan(mid):
                pnl = csh + pos * mid
            else:
                pnl = csh    # no mid available → realised cash only

            product_pnl[prod] = pnl
            total += pnl

        self.pnl_history.append(total)
        return product_pnl

    # ─────────────────────────────────────────────────────────────────────────
    def total_pnl(self, mid_prices: Dict[str, float]) -> float:
        """Current total PnL across all products."""
        total = 0.0
        for prod in self.products:
            mid = mid_prices.get(prod)
            pos = self.position.get(prod, 0)
            csh = self.cash.get(prod, 0.0)
            if mid is not None and not math.isnan(mid):
                total += csh + pos * mid
            else:
                total += csh
        return total

    # ─────────────────────────────────────────────────────────────────────────
    def final_pnl_per_product(
        self, mid_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Final MTM PnL breakdown per product (used in summary table)."""
        return {
            prod: self.cash.get(prod, 0.0) + self.position.get(prod, 0) * (mid_prices.get(prod) or 0.0)
            for prod in self.products
        }

    # ── FEATURE 2 & 3 ────────────────────────────────────────────────────────
    def snapshot_cumulative_pnl(
        self, mid_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Return the current cumulative MTM PnL per product WITHOUT recording
        it in pnl_history.  Used to snapshot state at the start of each day
        so that (end-of-day - start-of-day) gives the *daily delta* only.
        """
        return {
            prod: self.cash.get(prod, 0.0)
                  + self.position.get(prod, 0) * (mid_prices.get(prod) or 0.0)
            for prod in self.products
        }


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 4A — RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_metrics(pnl_history: List[float]) -> Dict[str, float]:
    """
    Compute Sharpe, Sortino, and Max Drawdown from tick-by-tick total PnL.

    Returns a dict with keys:
        sharpe, sortino, max_drawdown, max_drawdown_pct, total_return
    """
    if len(pnl_history) < 2:
        return {"sharpe": 0.0, "sortino": 0.0,
                "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
                "total_return": 0.0}

    pnl_arr = np.array(pnl_history, dtype=float)
    returns = np.diff(pnl_arr)       # tick-to-tick PnL changes

    mean_r = returns.mean()
    std_r  = returns.std()

    # Sharpe  (non-annualised — competition doesn't have a risk-free rate)
    sharpe = (mean_r / std_r) if std_r > 1e-12 else 0.0

    # Sortino  (downside deviation only)
    neg_returns = returns[returns < 0]
    down_std    = neg_returns.std() if len(neg_returns) > 1 else 1e-12
    sortino     = (mean_r / down_std) if down_std > 1e-12 else 0.0

    # Max Drawdown
    running_max  = np.maximum.accumulate(pnl_arr)
    drawdowns    = pnl_arr - running_max
    max_dd_abs   = float(drawdowns.min())
    peak_at_dd   = float(running_max[drawdowns.argmin()])
    max_dd_pct   = (max_dd_abs / peak_at_dd * 100) if abs(peak_at_dd) > 1e-6 else 0.0

    return {
        "sharpe":          round(sharpe,       4),
        "sortino":         round(sortino,       4),
        "max_drawdown":    round(max_dd_abs,    2),
        "max_drawdown_pct":round(max_dd_pct,   4),
        "total_return":    round(float(pnl_arr[-1]), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 4B — SANDBOX LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class SandboxLogger:
    """
    Writes a sandbox.log file that matches the official IMC visualiser format.

    Format:
        Sandbox logs:
        [T=0] [trader stdout captured here]

        Activities log:
        day;timestamp;product;bid_price_1;...;mid_price;profit_and_loss

        Trade history:
        day;timestamp;symbol;price;quantity;buyer;seller
    """

    ACT_HEADER = (
        "day;timestamp;product;"
        "bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;"
        "ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;"
        "mid_price;profit_and_loss"
    )
    TRADE_HEADER = "day;timestamp;symbol;price;quantity;buyer;seller"

    def __init__(self, path: str):
        self.path        = path
        self._sandbox:   List[str] = []   # [f"[T={ts}] {output}", ...]
        self._activities: List[str] = []
        self._trades:     List[str] = []

    # ─────────────────────────────────────────────────────────────────────────
    def log_tick(
        self,
        day:           int,
        timestamp:     int,
        books:         Dict[str, BookSnapshot],
        product_pnl:   Dict[str, float],
        trader_stdout: str,
        fills:         List[Fill],
        market_trades: Dict[str, List[Trade]],
    ) -> None:
        """Record one tick's data into the log buffers."""

        # ── Sandbox logs (trader print statements) ──────────────────────────
        if trader_stdout.strip():
            for line in trader_stdout.strip().split("\n"):
                self._sandbox.append(f"[T={timestamp}] {line}")

        # ── Activities log (one row per product) ────────────────────────────
        for product, snap in books.items():
            pnl = product_pnl.get(product, 0.0)

            # Serialise bid levels
            bid_levels = sorted(snap.bids.keys(), reverse=True)[:3]
            ask_levels = sorted(snap.asks.keys())[:3]

            def _lv(lst, i):
                return lst[i] if i < len(lst) else ""

            bid_str = SEP.join([
                f"{_lv(bid_levels,0)}{SEP}{snap.bids.get(_lv(bid_levels,0),'') if _lv(bid_levels,0) else ''}",
                f"{_lv(bid_levels,1)}{SEP}{snap.bids.get(_lv(bid_levels,1),'') if _lv(bid_levels,1) else ''}",
                f"{_lv(bid_levels,2)}{SEP}{snap.bids.get(_lv(bid_levels,2),'') if _lv(bid_levels,2) else ''}",
            ])
            ask_str = SEP.join([
                f"{_lv(ask_levels,0)}{SEP}{snap.asks.get(_lv(ask_levels,0),'') if _lv(ask_levels,0) else ''}",
                f"{_lv(ask_levels,1)}{SEP}{snap.asks.get(_lv(ask_levels,1),'') if _lv(ask_levels,1) else ''}",
                f"{_lv(ask_levels,2)}{SEP}{snap.asks.get(_lv(ask_levels,2),'') if _lv(ask_levels,2) else ''}",
            ])

            mid = snap.mid_price if snap.mid_price is not None else ""
            self._activities.append(
                f"{day}{SEP}{timestamp}{SEP}{product}{SEP}"
                f"{bid_str}{SEP}{ask_str}{SEP}"
                f"{mid}{SEP}{pnl:.2f}"
            )

        # ── Trade history (our fills + market trades) ────────────────────────
        for f in fills:
            buyer  = OUR_NAME if f.quantity > 0 else "MARKET"
            seller = "MARKET" if f.quantity > 0 else OUR_NAME
            self._trades.append(
                f"{day}{SEP}{timestamp}{SEP}{f.product}{SEP}"
                f"{f.price}{SEP}{abs(f.quantity)}{SEP}{buyer}{SEP}{seller}"
            )

        for sym, tlist in market_trades.items():
            for tr in tlist:
                self._trades.append(
                    f"{day}{SEP}{timestamp}{SEP}{sym}{SEP}"
                    f"{tr.price}{SEP}{tr.quantity}{SEP}"
                    f"{tr.buyer or ''}{SEP}{tr.seller or ''}"
                )

    # ─────────────────────────────────────────────────────────────────────────
    def save(self) -> None:
        """Flush all buffers to the log file."""
        lines = [
            "Sandbox logs:",
            *self._sandbox,
            "",
            "Activities log:",
            self.ACT_HEADER,
            *self._activities,
            "",
            "Trade history:",
            self.TRADE_HEADER,
            *self._trades,
            "",
        ]
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        # ── FEATURE 4: confirm the exact (dynamically-named) log path written
        print(f"\n  ✓  Log written → {os.path.abspath(self.path)}"
              f"  ({len(self._activities):,} activity rows, "
              f"{len(self._trades):,} trade rows)")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3C — MAIN BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Orchestrates the full simulation.

    For each tick in each requested day:
      1. Build a BookSnapshot per product from the CSV row.
      2. Construct TradingState with order_depths, positions, own/market trades.
      3. Call Trader.run(state) — capture print output and the returned orders.
      4. Pass orders to MatchingEngine → get fills + rejections.
      5. Apply fills to PnLTracker.  Compute MTM PnL.
      6. Log everything to SandboxLogger.
      7. Advance to next tick.
    """

    def __init__(
        self,
        data_dir:      str,
        days:          List[int],
        trader_cls,                   # the Trader class (uninstantiated)
        no_log:        bool = False,
        log_path:      str  = "sandbox.log",
        show_all:      bool = False,
        # ── FEATURE 1 & 4: carry the strategy filename for banner + log naming
        strategy_name: str  = "trader.py",
    ):
        self.data_dir      = data_dir
        self.days          = days
        self.trader        = trader_cls()     # instantiate once
        self.no_log        = no_log
        self.log_path      = log_path
        self.show_all      = show_all
        self.strategy_name = strategy_name   # Feature 1: for banner display

        self.engine    = MatchingEngine(limit=POSITION_LIMIT)
        self.loader    = DataLoader(data_dir)
        self.logger    = SandboxLogger(log_path) if not no_log else None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> Dict:
        """
        Run the full multi-day simulation.
        Returns a dict with final PnL summary and risk metrics.
        """
        # ── FEATURE 1: pass strategy_name into banner ─────────────────────
        _print_banner(self.days, strategy_name=self.strategy_name)

        print("  Loading CSV data …")
        t0 = time.time()
        price_map, trade_map, ts_by_day, products = self.loader.load_days(self.days)
        print(f"  Data loaded in {time.time()-t0:.2f}s  "
              f"({len(products)} products, "
              f"{sum(len(v) for v in ts_by_day.values())} ticks)\n")

        tracker  = PnLTracker(products)
        listings = {p: Listing(symbol=p, product=p) for p in products}

        # Persistent across the ENTIRE multi-day run (matches competition behaviour)
        trader_data: str           = ""
        positions:   Dict[str,int] = defaultdict(int)
        prev_fills:  List[Fill]    = []       # own_trades from previous tick

        last_mid_prices: Dict[str, float] = {}

        for day in self.days:
            timestamps = ts_by_day.get(day, [])
            if not timestamps:
                print(f"  ⚠  No ticks for day {day}, skipping.")
                continue

            # ── FEATURE 2 & 3: snapshot cumulative PnL BEFORE this day starts
            # so we can compute the per-day DELTA at the end of the day loop.
            pnl_snapshot_start_of_day: Dict[str, float] = tracker.snapshot_cumulative_pnl(
                last_mid_prices
            )
            # Record where in pnl_history this day starts so we can slice it
            # for per-day risk metrics (Feature 3).
            tick_history_start_idx: int = len(tracker.pnl_history)

            desc = f"  Day {day}"
            with tqdm(
                total=len(timestamps),
                desc=desc,
                unit="tick",
                bar_format="{desc}: {percentage:3.0f}%|{bar:30}| "
                           "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ncols=88,
            ) as pbar:
                for ts in timestamps:
                    key = (day, ts)

                    # ── 1. Gather book snapshots ─────────────────────────────
                    price_rows = price_map.get(key, {})
                    books: Dict[str, BookSnapshot] = {}
                    mid_prices: Dict[str, float]   = {}

                    for prod, row in price_rows.items():
                        snap = BookSnapshot.from_csv_row(row)
                        books[prod] = snap
                        if snap.mid_price is not None:
                            mid_prices[prod] = snap.mid_price

                    last_mid_prices.update(mid_prices)

                    # ── 2. Build market_trades from trade CSV ────────────────
                    raw_trades    = trade_map.get(key, [])
                    market_trades = build_market_trades(raw_trades)

                    # ── 3. Convert previous tick's fills → own_trades ────────
                    own_trades = fills_to_own_trades(prev_fills)

                    # ── 4. Construct TradingState ────────────────────────────
                    state = build_trading_state(
                        trader_data   = trader_data,
                        listings      = listings,
                        books         = books,
                        own_trades    = own_trades,
                        market_trades = market_trades,
                        positions     = dict(positions),
                        timestamp     = ts,
                    )

                    # ── 5. Call Trader.run() — capture stdout ────────────────
                    stdout_capture = io.StringIO()
                    orders: Dict[str, List[Order]] = {}
                    new_trader_data = trader_data

                    _saved_stdout = sys.stdout
                    sys.stdout = stdout_capture
                    try:
                        result = self.trader.run(state)
                        if isinstance(result, tuple) and len(result) >= 1:
                            orders = result[0] or {}
                            if len(result) >= 3:
                                new_trader_data = result[2] or ""
                    except Exception as exc:
                        # Trader threw an error — log it but keep simulating
                        sys.stdout = _saved_stdout
                        err_msg = traceback.format_exc()
                        stdout_capture.write(f"[EXCEPTION] {err_msg}")
                    finally:
                        sys.stdout = _saved_stdout

                    trader_stdout  = stdout_capture.getvalue()
                    trader_data    = new_trader_data

                    # ── 6. Run matching engine ────────────────────────────────
                    fills, rejections = self.engine.process(
                        orders    = orders,
                        books     = books,
                        positions = positions,
                        timestamp = ts,
                        day       = day,
                    )

                    # Log rejections to sandbox
                    for prod, reason in rejections.items():
                        trader_stdout += f"[REJECTED {prod}] {reason}\n"

                    # ── 7. Update positions and PnL ───────────────────────────
                    tracker.apply_fills(fills)
                    for f in fills:
                        positions[f.product] = tracker.position[f.product]

                    product_pnl = tracker.mark_to_market(last_mid_prices)

                    # ── 8. Log to sandbox ────────────────────────────────────
                    if self.logger:
                        self.logger.log_tick(
                            day           = day,
                            timestamp     = ts,
                            books         = books,
                            product_pnl   = product_pnl,
                            trader_stdout = trader_stdout,
                            fills         = fills,
                            market_trades = market_trades,
                        )

                    prev_fills = fills
                    pbar.update(1)

            # ── FEATURE 2: per-asset daily PnL delta ──────────────────────────
            # Compute end-of-day cumulative PnL, then subtract start-of-day
            # snapshot so every value represents THIS day's contribution only.
            pnl_snapshot_end_of_day = tracker.snapshot_cumulative_pnl(last_mid_prices)
            daily_pnl_delta: Dict[str, float] = {
                prod: pnl_snapshot_end_of_day.get(prod, 0.0)
                      - pnl_snapshot_start_of_day.get(prod, 0.0)
                for prod in products
            }

            # ── FEATURE 3: per-day risk metrics ───────────────────────────────
            # Slice pnl_history to only the ticks that ran during this day.
            daily_tick_pnl_slice: List[float] = tracker.pnl_history[
                tick_history_start_idx:
            ]
            daily_risk = compute_risk_metrics(daily_tick_pnl_slice)

            # ── Print per-day summary (Features 2 & 3 combined) ───────────────
            _print_daily_summary(
                day      = day,
                daily_pnl= daily_pnl_delta,
                risk     = daily_risk,
                show_all = self.show_all,
            )

        # ── Final PnL snapshot ─────────────────────────────────────────────
        final_pnl_per_product = tracker.final_pnl_per_product(last_mid_prices)

        # ── Risk metrics ───────────────────────────────────────────────────
        risk = compute_risk_metrics(tracker.pnl_history)

        # ── Print summary ──────────────────────────────────────────────────
        _print_summary(
            final_pnl_per_product,
            risk,
            products,
            show_all=self.show_all,
        )

        # ── Write sandbox.log ─────────────────────────────────────────────
        if self.logger:
            self.logger.save()

        return {
            "per_product_pnl": final_pnl_per_product,
            "risk_metrics":    risk,
            "pnl_history":     tracker.pnl_history,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

# ── FEATURE 1: banner now shows the loaded strategy filename ─────────────────
def _print_banner(days: List[int], strategy_name: str = "trader.py") -> None:
    w = 62
    print()
    print("┌" + "─" * w + "┐")
    print(f"│{'IMC Prosperity 4  ·  Round 5  ·  Offline Backtester v2.0':^{w}}│")
    print(f"│{'Architecture: Ignacio Pinazo Orihuela':^{w}}│")
    print(f"│{('Strategy: ' + strategy_name):^{w}}│")
    print(f"│{('Days: ' + ', '.join(str(d) for d in days)):^{w}}│")
    print("└" + "─" * w + "┘")
    print()


def _fmt_pnl(v: float) -> str:
    """Format a PnL value with sign and comma-separator."""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:>14,.2f}"


# ── FEATURE 2 & 3 ─────────────────────────────────────────────────────────────
def _print_daily_summary(
    day:        int,
    daily_pnl:  Dict[str, float],   # per-product PnL delta for THIS day only
    risk:       Dict[str, float],   # Sharpe / Sortino / drawdown for THIS day
    show_all:   bool = False,
) -> None:
    """
    Print the per-asset PnL breakdown for a single completed day followed
    by that day's risk metrics block.  Called at the end of each day loop.

    FEATURE 2 — daily_pnl is the delta (end-of-day minus start-of-day) so it
    shows exactly what the strategy earned/lost on that calendar day.

    FEATURE 3 — risk is computed only from the tick-slice for that day, giving
    a per-day Sharpe, Sortino, and Max Drawdown.
    """
    w = 58
    total = sum(daily_pnl.values())

    print()
    print("  ┌" + "─" * w + "┐")
    print(f"  │{'  DAY ' + str(day) + ' — PnL BREAKDOWN':^{w}}│")
    print("  ├" + "─" * w + "┤")

    # Sort by absolute daily PnL, show only active products unless show_all
    sorted_items = sorted(daily_pnl.items(), key=lambda x: abs(x[1]), reverse=True)
    items_to_show = sorted_items if show_all else [
        item for item in sorted_items if abs(item[1]) > 0.01
    ][:50]

    if not items_to_show:
        print(f"  │{'  (no trades this day)':^{w}}│")
    else:
        for prod, pnl in items_to_show:
            col = "\033[92m" if pnl >= 0 else "\033[91m"
            rst = "\033[0m"
            pnl_str = _fmt_pnl(pnl)
            row     = f"  {prod:<32}{pnl_str}"
            # pad to consistent width accounting for invisible ANSI bytes
            print(f"  │{col}{row}{rst}  │")

    print("  ├" + "─" * w + "┤")
    col = "\033[92m" if total >= 0 else "\033[91m"
    rst = "\033[0m"
    total_row = f"  {'Day ' + str(day) + ' Total':<32}{_fmt_pnl(total)}"
    print(f"  │{col}{total_row}{rst}  │")
    print("  └" + "─" * w + "┘")

    # ── FEATURE 3: per-day risk metrics ──────────────────────────────────────
    print(f"  Day {day} Risk Metrics")
    print("  " + "─" * 40)
    print(f"  Sharpe Ratio       :  {risk['sharpe']:>10.4f}")
    print(f"  Sortino Ratio      :  {risk['sortino']:>10.4f}")
    mdd     = risk['max_drawdown']
    mdd_pct = risk['max_drawdown_pct']
    print(f"  Max Drawdown       :  {mdd:>14,.2f}  ({mdd_pct:.2f}%)")
    print(f"  Day PnL            :  {_fmt_pnl(total)}")
    print()


def _print_summary(
    per_product: Dict[str, float],
    risk:        Dict[str, float],
    products:    List[str],
    show_all:    bool = False,
) -> None:
    w = 58

    print()
    print("╔" + "═" * w + "╗")
    print(f"║{'  FINAL PnL SUMMARY':^{w}}║")
    print("╠" + "═" * w + "╣")

    # Sort by absolute PnL descending
    sorted_items = sorted(per_product.items(), key=lambda x: abs(x[1]), reverse=True)

    # If show_all, print everything; else print non-zero + top-10
    items_to_show = sorted_items if show_all else [
        item for item in sorted_items if abs(item[1]) > 0.01
    ][:50]

    for prod, pnl in items_to_show:
        col = "\033[92m" if pnl >= 0 else "\033[91m"  # green / red
        rst = "\033[0m"
        pnl_str = _fmt_pnl(pnl)
        row = f"  {prod:<32}{pnl_str}"
        print(f"║{col}{row}{rst:<{w - len(row) + len(row)}}║")

    print("╠" + "═" * w + "╣")
    total = sum(per_product.values())
    col   = "\033[92m" if total >= 0 else "\033[91m"
    rst   = "\033[0m"
    total_str = _fmt_pnl(total)
    total_row = f"  {'TOTAL PROFIT':<32}{total_str}"
    print(f"║{col}{total_row}{rst}  ║")
    print("╚" + "═" * w + "╝")

    print()
    print("  Risk Metrics")
    print("  " + "─" * 40)
    print(f"  Sharpe Ratio       :  {risk['sharpe']:>10.4f}")
    print(f"  Sortino Ratio      :  {risk['sortino']:>10.4f}")
    mdd = risk['max_drawdown']
    mdd_pct = risk['max_drawdown_pct']
    print(f"  Max Drawdown       :  {mdd:>14,.2f}  ({mdd_pct:.2f}%)")
    print(f"  Final PnL          :  {_fmt_pnl(risk['total_return'])}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  TRADER LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_trader(path: str):
    """
    Dynamically import a trader file and return its Trader class.
    The trader file must define `class Trader` with `def run(self, state)`.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Trader file not found: {abs_path}")

    spec   = importlib.util.spec_from_file_location("trader_module", abs_path)
    module = importlib.util.module_from_spec(spec)

    # Make datamodel available in the trader's namespace
    import datamodel as _dm
    module.datamodel = _dm                # legacy: `from datamodel import ...`
    sys.modules["trader_module"] = module

    spec.loader.exec_module(module)

    if not hasattr(module, "Trader"):
        raise AttributeError(
            f"No class 'Trader' found in {path}. "
            "Ensure your file defines `class Trader:` with `def run(self, state)`."
        )

    trader_cls = module.Trader
    if not hasattr(trader_cls, "run"):
        raise AttributeError(
            f"Trader class in {path} has no `run` method."
        )

    return trader_cls


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

# ── parse_args ────────────────────────────────────────────────────────────────
# Key change: log generation is now OPT-IN via --gen-log (action='store_true').
# The old --no-log flag has been removed; silence is the new default.
# --log is kept as an optional path override that only takes effect when
# --gen-log is also present.
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="IMC Prosperity 3 Round 5 — Offline Backtester v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: simulation + terminal output only — no file written
  python backtester.py --strategy v26_no_fv.py
  python backtester.py --strategy v26_no_fv.py --days 2 3

  # Opt-in log generation with --gen-log
  python backtester.py --strategy v26_no_fv.py --gen-log
  #   → writes  v26_no_fv.log  (auto-named from strategy basename)

  # Opt-in with an explicit output path override
  python backtester.py --strategy v26_no_fv.py --gen-log --log results/run1.log

  # Other flags work the same as before
  python backtester.py --days 2 3 4 --data-dir ./data --all-products
        """,
    )
    ap.add_argument(
        "--days", nargs="+", type=int, default=[2, 3, 4],
        help="Days to simulate (default: 2 3 4)",
    )
    ap.add_argument(
        "--data-dir", default="data",
        help="Directory containing the CSV files (default: ./data)",
    )

    # ── FEATURE 1: primary strategy argument ─────────────────────────────────
    ap.add_argument(
        "--strategy",
        default=None,
        metavar="FILE",
        help=(
            "Path to the strategy file to backtest "
            "(e.g. --strategy v26_no_fv.py). "
            "Falls back to --trader, then 'trader.py'."
        ),
    )

    # ── Legacy alias ──────────────────────────────────────────────────────────
    ap.add_argument(
        "--trader",
        default=None,
        metavar="FILE",
        help="[DEPRECATED] Use --strategy instead.",
    )

    # ── OPT-IN LOG FLAG ───────────────────────────────────────────────────────
    # When absent  → no log file is constructed or written (default, fast path).
    # When present → a .log file is generated and saved using the dynamic naming
    #                convention: <strategy_basename>.log  (or --log override).
    ap.add_argument(                            # <── NEW FLAG (opt-in)
        "--gen-log",
        action="store_true",
        default=False,
        help=(
            "Generate and save a visualiser-compatible log file. "
            "Off by default to avoid unnecessary disk I/O. "
            "Output is named <strategy_basename>.log unless --log is also given."
        ),
    )

    # ── Optional explicit path override (only used when --gen-log is set) ─────
    ap.add_argument(
        "--log",
        default=None,
        metavar="FILE",
        help=(
            "Override the output log path. "
            "Only takes effect when --gen-log is also passed. "
            "Defaults to <strategy_basename>.log."
        ),
    )

    ap.add_argument(
        "--all-products", action="store_true",
        help="Show all 50 products in per-day and final summary tables.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve strategy file path ────────────────────────────────────────────
    # Priority: --strategy  >  --trader (deprecated)  >  "trader.py" default
    if args.strategy:
        strategy_path = args.strategy
    elif args.trader:
        print("  ⚠  --trader is deprecated; please use --strategy instead.")
        strategy_path = args.trader
    else:
        strategy_path = "trader.py"

    strategy_name = os.path.basename(strategy_path)   # e.g. "ntrader-round3-v5.py"

    # ── OPT-IN LOG PATH RESOLUTION ────────────────────────────────────────────
    # Gate: only derive a log_path when the user explicitly passed --gen-log.
    # Without the flag, log_path stays None and no SandboxLogger is created,
    # bypassing all string construction and file I/O entirely.
    #
    # With --gen-log:
    #   • --log <path>  →  honour the explicit override                (custom)
    #   • (no --log)    →  auto-derive "<strategy_stem>.log"           (default)
    if args.gen_log:                                   # <── OPT-IN GATE
        if args.log:
            log_path = args.log                        # explicit path override
        else:
            stem     = os.path.splitext(strategy_name)[0]   # "ntrader-round3-v5"
            log_path = f"{stem}.log"                         # "ntrader-round3-v5.log"
    else:
        log_path = None                                # no log — fast path

    # ── Load strategy ─────────────────────────────────────────────────────────
    try:
        trader_cls = load_trader(strategy_path)
        print(f"  ✓  Strategy loaded : {os.path.abspath(strategy_path)}")
        if args.gen_log:                               # <── only print when logging
            print(f"  ✓  Log will write  : {os.path.abspath(log_path)}")
        else:
            print(f"  ℹ  Log generation  : disabled (pass --gen-log to enable)")
    except (FileNotFoundError, AttributeError) as e:
        print(f"  ✗  {e}")
        sys.exit(1)

    # ── Run backtester ────────────────────────────────────────────────────────
    # no_log = not args.gen_log  →  True when flag absent, False when present.
    # Backtester.__init__ creates SandboxLogger only when no_log is False,
    # so the entire logging pipeline is skipped on the default (no-flag) path.
    bt = Backtester(
        data_dir      = args.data_dir,
        days          = args.days,
        trader_cls    = trader_cls,
        no_log        = not args.gen_log,              # <── OPT-IN INVERSION
        log_path      = log_path or "sandbox.log",     # fallback never reached when no_log=True
        show_all      = args.all_products,
        strategy_name = strategy_name,
    )

    try:
        bt.run()
    except FileNotFoundError as e:
        print(f"\n  ✗  Data error: {e}")
        print("  Ensure CSV files are in the --data-dir directory.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()