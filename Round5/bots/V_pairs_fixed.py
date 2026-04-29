# IMC Prosperity 4 — Round 5  V_pairs_fixed (frozen-β walk-forward pair trader)
#
# Replacement for round5_trader.py. Selected pair(s) come from pair_selector.py
# trained on Day 2 only and validated on Days 3-4 (no peeking).
#
# Conservative knobs (per plan):
#   - z entry 2.0 / exit 0.5 (was 1.5 / 0.3)
#   - frozen β (no rolling refresh — that bled in regime shifts)
#   - cost gate: enter only if |z|·σ·(1+|β|) > 3 × round-trip cost
#   - hard daily stop-loss per pair
#   - qty_per_leg = 3, POS_LIMIT = ±10, all Order prices int()-wrapped

import json, math
from typing import List, Tuple
from datamodel import OrderDepth, TradingState, Order


# ─── Embedded pair config (output of pair_selector.py, train_day=2, holdout=[3,4]) ───
# To regenerate: python3 "ROUND 5/STRATEGIES/pair_selector.py"
SELECTED_PAIRS: List[Tuple[str, str, float]] = [
    # (Y, X, beta_frozen)
    ("TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_GRAPHITE_MIST", 0.70398),
]


# ─── Hyperparameters (locked) ───────────────────────────────────────────────
POS_LIMIT          = 10
QTY_PER_LEG        = 3
ENTRY_Z            = 2.0
EXIT_Z             = 0.5
Z_WIN              = 300
MIN_HIST_FOR_TRADE = 100
COST_GATE_MULT     = 3.0      # require |z|·σ·(1+|β|) > 3 × round-trip cost
BA_WIN             = 100      # rolling window for bid-ask cost estimate
MAX_HOLD_TICKS     = 1500     # halt pair if same-sign position held this long without exit


def _mid(state: TradingState, p: str):
    od = state.order_depths.get(p)
    if od is None or not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0


def _best_bid(state: TradingState, p: str):
    od = state.order_depths.get(p)
    return int(max(od.buy_orders)) if od and od.buy_orders else None


def _best_ask(state: TradingState, p: str):
    od = state.order_depths.get(p)
    return int(min(od.sell_orders)) if od and od.sell_orders else None


def _ba_spread(state: TradingState, p: str):
    bb = _best_bid(state, p)
    ba = _best_ask(state, p)
    if bb is None or ba is None:
        return None
    return ba - bb


def _hedge_clip(hedge: int, x_pos: int, side: str) -> int:
    """Clamp hedge so that the resulting X-leg position stays within ±POS_LIMIT.

    side = 'long_spread'  → X order qty = -hedge   (sell when hedge>0, buy when hedge<0)
    side = 'short_spread' → X order qty = +hedge   (buy  when hedge>0, sell when hedge<0)
    """
    if side == "long_spread":
        # New x_pos = x_pos - hedge ; require -POS_LIMIT ≤ new ≤ +POS_LIMIT
        if hedge > 0:
            return max(0, min(hedge, x_pos + POS_LIMIT))
        elif hedge < 0:
            return min(0, max(hedge, x_pos - POS_LIMIT))
        return 0
    else:  # short_spread: new x_pos = x_pos + hedge
        if hedge > 0:
            return max(0, min(hedge, POS_LIMIT - x_pos))
        elif hedge < 0:
            return min(0, max(hedge, -(POS_LIMIT + x_pos)))
        return 0


class Trader:
    def run(self, state: TradingState):
        td_raw = getattr(state, "traderData", "") or ""
        try:
            td = json.loads(td_raw) if td_raw else {}
        except Exception:
            td = {}

        result = {}

        for Y, X, beta in SELECTED_PAIRS:
            key = f"{Y}|{X}"
            ps = td.get(key, {
                "sp_hist": [],
                "ba_y_hist": [], "ba_x_hist": [],
                "ticks_in_trade": 0,     # consecutive ticks with non-zero pair position
                "tick": 0,
                "halted": False,
            })

            y_mid = _mid(state, Y)
            x_mid = _mid(state, X)
            if y_mid is None or x_mid is None:
                td[key] = ps
                continue

            ps["tick"] += 1

            # ── Spread + z-score ──
            sp_now = y_mid - beta * x_mid
            ps["sp_hist"].append(sp_now)
            if len(ps["sp_hist"]) > Z_WIN:
                ps["sp_hist"] = ps["sp_hist"][-Z_WIN:]

            # ── Track bid-ask cost estimate ──
            bay = _ba_spread(state, Y)
            bax = _ba_spread(state, X)
            if bay is not None:
                ps["ba_y_hist"].append(bay)
                if len(ps["ba_y_hist"]) > BA_WIN:
                    ps["ba_y_hist"] = ps["ba_y_hist"][-BA_WIN:]
            if bax is not None:
                ps["ba_x_hist"].append(bax)
                if len(ps["ba_x_hist"]) > BA_WIN:
                    ps["ba_x_hist"] = ps["ba_x_hist"][-BA_WIN:]

            td[key] = ps

            if ps["halted"]:
                continue
            if len(ps["sp_hist"]) < MIN_HIST_FOR_TRADE:
                continue
            if not ps["ba_y_hist"] or not ps["ba_x_hist"]:
                continue

            mu  = sum(ps["sp_hist"]) / len(ps["sp_hist"])
            var = sum((s - mu) ** 2 for s in ps["sp_hist"]) / len(ps["sp_hist"])
            sig = math.sqrt(var) if var > 1e-12 else 1e-9
            z   = (sp_now - mu) / sig

            avg_ba_y = sum(ps["ba_y_hist"]) / len(ps["ba_y_hist"])
            avg_ba_x = sum(ps["ba_x_hist"]) / len(ps["ba_x_hist"])
            rt_cost  = 2.0 * (avg_ba_y + abs(beta) * avg_ba_x)
            expected_capture = abs(z) * sig * (1.0 + abs(beta))
            cost_ok = expected_capture > COST_GATE_MULT * rt_cost

            y_pos = state.position.get(Y, 0)
            x_pos = state.position.get(X, 0)
            y_ords, x_ords = [], []

            # ── ENTRY: SHORT spread (z > +ENTRY_Z) ──
            if z > ENTRY_Z and cost_ok and y_pos > -POS_LIMIT:
                qty_y = min(QTY_PER_LEG, y_pos + POS_LIMIT)
                hedge = _hedge_clip(int(round(beta * qty_y)), x_pos, "short_spread")
                y_bid = _best_bid(state, Y)
                if qty_y > 0 and y_bid is not None:
                    y_ords.append(Order(Y, int(y_bid), -qty_y))
                    if hedge > 0:
                        x_ask = _best_ask(state, X)
                        if x_ask is not None:
                            x_ords.append(Order(X, int(x_ask), int(hedge)))
                    elif hedge < 0:
                        x_bid = _best_bid(state, X)
                        if x_bid is not None:
                            x_ords.append(Order(X, int(x_bid), int(hedge)))

            # ── ENTRY: LONG spread (z < -ENTRY_Z) ──
            elif z < -ENTRY_Z and cost_ok and y_pos < POS_LIMIT:
                qty_y = min(QTY_PER_LEG, POS_LIMIT - y_pos)
                hedge = _hedge_clip(int(round(beta * qty_y)), x_pos, "long_spread")
                y_ask = _best_ask(state, Y)
                if qty_y > 0 and y_ask is not None:
                    y_ords.append(Order(Y, int(y_ask), qty_y))
                    if hedge > 0:
                        x_bid = _best_bid(state, X)
                        if x_bid is not None:
                            x_ords.append(Order(X, int(x_bid), -int(hedge)))
                    elif hedge < 0:
                        x_ask = _best_ask(state, X)
                        if x_ask is not None:
                            x_ords.append(Order(X, int(x_ask), -int(hedge)))

            # ── EXIT: |z| < EXIT_Z → flatten both legs at touch ──
            elif abs(z) < EXIT_Z:
                if y_pos > 0:
                    p = _best_bid(state, Y)
                    if p is not None:
                        y_ords.append(Order(Y, int(p), -y_pos))
                elif y_pos < 0:
                    p = _best_ask(state, Y)
                    if p is not None:
                        y_ords.append(Order(Y, int(p), -y_pos))
                if x_pos > 0:
                    p = _best_bid(state, X)
                    if p is not None:
                        x_ords.append(Order(X, int(p), -x_pos))
                elif x_pos < 0:
                    p = _best_ask(state, X)
                    if p is not None:
                        x_ords.append(Order(X, int(p), -x_pos))

            if y_ords:
                result.setdefault(Y, []).extend(y_ords)
            if x_ords:
                result.setdefault(X, []).extend(x_ords)

            # ── Stop-loss: position-time halt ──
            # If we've been holding pair position for too long without exiting, the spread
            # has drifted against us. Halt this pair for the remainder of traderData lifetime.
            if y_pos != 0 or x_pos != 0:
                ps["ticks_in_trade"] += 1
            else:
                ps["ticks_in_trade"] = 0
            if ps["ticks_in_trade"] > MAX_HOLD_TICKS:
                ps["halted"] = True

            td[key] = ps

        return result, 0, json.dumps(td)
