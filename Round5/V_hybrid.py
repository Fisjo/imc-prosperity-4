# IMC Prosperity 4 — Round 5  V_hybrid  (MM baseline + selective validated pairs)
#
# Combines:
#   - V_mm.py logic on SNACKPACK + OXYGEN_SHAKE families  (proven +40k in 3-day backtest)
#   - V_pairs_fixed.py logic on the walk-forward-validated TRANSLATOR pair
#
# The MM whitelist excludes TRANSLATOR, so pair legs and MM products never overlap;
# position-limit accounting is automatically separate via state.position[product].

import json, math
from typing import List, Tuple
from datamodel import OrderDepth, TradingState, Order


# ─── MM config (lifted from V_mm.py) ──────────────────────────────────────────
POS_LIMIT      = 10
EMA_SPAN       = 200
EMA_ALPHA      = 2.0 / (EMA_SPAN + 1.0)
MAX_QUOTE_SIZE = 5
SKEW_FACTOR    = 0.2
MIN_BA_TO_MM   = 10
MM_FAMILIES    = {"SNACKPACK", "OXYGEN_SHAKE"}

MM_PRODUCTS: List[str] = [
    "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH",
    "SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA",
]


# ─── Pair config (lifted from V_pairs_fixed.py / validated_pairs.json) ───────
SELECTED_PAIRS: List[Tuple[str, str, float]] = [
    ("TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_GRAPHITE_MIST", 0.70398),
]
PAIR_QTY_PER_LEG    = 3
PAIR_ENTRY_Z        = 2.0
PAIR_EXIT_Z         = 0.5
PAIR_Z_WIN          = 300
PAIR_MIN_HIST       = 100
PAIR_COST_GATE_MULT = 3.0
PAIR_BA_WIN         = 100
PAIR_MAX_HOLD_TICKS = 1500


# ─── Helpers ──────────────────────────────────────────────────────────────────
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
    if side == "long_spread":
        if hedge > 0:   return max(0, min(hedge, x_pos + POS_LIMIT))
        elif hedge < 0: return min(0, max(hedge, x_pos - POS_LIMIT))
        return 0
    else:  # short_spread
        if hedge > 0:   return max(0, min(hedge, POS_LIMIT - x_pos))
        elif hedge < 0: return min(0, max(hedge, -(POS_LIMIT + x_pos)))
        return 0


class Trader:
    def run(self, state: TradingState):
        td_raw = getattr(state, "traderData", "") or ""
        try:
            td = json.loads(td_raw) if td_raw else {}
        except Exception:
            td = {}

        ema_store  = td.get("ema", {})
        pair_store = td.get("pair", {})

        result = {}

        # ─── Layer 1: Market making on whitelist (SNACKPACK + OXYGEN_SHAKE) ───
        for prod in MM_PRODUCTS:
            od = state.order_depths.get(prod)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue
            best_bid = int(max(od.buy_orders))
            best_ask = int(min(od.sell_orders))
            mid      = (best_bid + best_ask) / 2.0
            ba       = best_ask - best_bid
            if ba <= 0:
                continue

            prev = ema_store.get(prod)
            fair = mid if prev is None else (EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * float(prev))
            ema_store[prod] = fair

            if ba < MIN_BA_TO_MM:
                continue

            pos  = int(state.position.get(prod, 0))
            edge = max(1, ba // 2 - 1)
            skew = int(round(SKEW_FACTOR * pos))

            bid_px = int(round(fair)) - edge - skew
            ask_px = int(round(fair)) + edge - skew
            bid_px = min(bid_px, best_ask - 1)
            ask_px = max(ask_px, best_bid + 1)
            if ask_px <= bid_px:
                ask_px = bid_px + 1

            buy_size  = min(MAX_QUOTE_SIZE, POS_LIMIT - pos)
            sell_size = min(MAX_QUOTE_SIZE, POS_LIMIT + pos)

            orders: List[Order] = []
            if buy_size > 0:
                orders.append(Order(prod, int(bid_px), int(buy_size)))
            if sell_size > 0:
                orders.append(Order(prod, int(ask_px), -int(sell_size)))

            take_thr = max(2, edge)
            if best_ask < fair - take_thr and (POS_LIMIT - pos) > 0:
                take_qty = min(POS_LIMIT - pos, abs(int(od.sell_orders.get(best_ask, MAX_QUOTE_SIZE))))
                if take_qty > 0:
                    orders.append(Order(prod, int(best_ask), int(take_qty)))
            if best_bid > fair + take_thr and (POS_LIMIT + pos) > 0:
                take_qty = min(POS_LIMIT + pos, int(od.buy_orders.get(best_bid, MAX_QUOTE_SIZE)))
                if take_qty > 0:
                    orders.append(Order(prod, int(best_bid), -int(take_qty)))

            if orders:
                result[prod] = orders

        # ─── Layer 2: Validated pair trading (TRANSLATOR_ASTRO_BLACK / GRAPHITE_MIST) ───
        for Y, X, beta in SELECTED_PAIRS:
            key = f"{Y}|{X}"
            ps = pair_store.get(key, {
                "sp_hist": [],
                "ba_y_hist": [], "ba_x_hist": [],
                "ticks_in_trade": 0,
                "halted": False,
            })

            y_mid = _mid(state, Y)
            x_mid = _mid(state, X)
            if y_mid is None or x_mid is None:
                pair_store[key] = ps
                continue

            sp_now = y_mid - beta * x_mid
            ps["sp_hist"].append(sp_now)
            if len(ps["sp_hist"]) > PAIR_Z_WIN:
                ps["sp_hist"] = ps["sp_hist"][-PAIR_Z_WIN:]

            bay = _ba_spread(state, Y)
            bax = _ba_spread(state, X)
            if bay is not None:
                ps["ba_y_hist"].append(bay)
                if len(ps["ba_y_hist"]) > PAIR_BA_WIN:
                    ps["ba_y_hist"] = ps["ba_y_hist"][-PAIR_BA_WIN:]
            if bax is not None:
                ps["ba_x_hist"].append(bax)
                if len(ps["ba_x_hist"]) > PAIR_BA_WIN:
                    ps["ba_x_hist"] = ps["ba_x_hist"][-PAIR_BA_WIN:]

            pair_store[key] = ps

            if ps["halted"]:
                continue
            if len(ps["sp_hist"]) < PAIR_MIN_HIST:
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
            cost_ok  = abs(z) * sig * (1.0 + abs(beta)) > PAIR_COST_GATE_MULT * rt_cost

            y_pos = state.position.get(Y, 0)
            x_pos = state.position.get(X, 0)
            y_ords, x_ords = [], []

            if z > PAIR_ENTRY_Z and cost_ok and y_pos > -POS_LIMIT:
                qty_y = min(PAIR_QTY_PER_LEG, y_pos + POS_LIMIT)
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

            elif z < -PAIR_ENTRY_Z and cost_ok and y_pos < POS_LIMIT:
                qty_y = min(PAIR_QTY_PER_LEG, POS_LIMIT - y_pos)
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

            elif abs(z) < PAIR_EXIT_Z:
                if y_pos > 0:
                    p = _best_bid(state, Y)
                    if p is not None: y_ords.append(Order(Y, int(p), -y_pos))
                elif y_pos < 0:
                    p = _best_ask(state, Y)
                    if p is not None: y_ords.append(Order(Y, int(p), -y_pos))
                if x_pos > 0:
                    p = _best_bid(state, X)
                    if p is not None: x_ords.append(Order(X, int(p), -x_pos))
                elif x_pos < 0:
                    p = _best_ask(state, X)
                    if p is not None: x_ords.append(Order(X, int(p), -x_pos))

            if y_ords:
                result.setdefault(Y, []).extend(y_ords)
            if x_ords:
                result.setdefault(X, []).extend(x_ords)

            # Position-time stop-loss
            if y_pos != 0 or x_pos != 0:
                ps["ticks_in_trade"] += 1
            else:
                ps["ticks_in_trade"] = 0
            if ps["ticks_in_trade"] > PAIR_MAX_HOLD_TICKS:
                ps["halted"] = True
            pair_store[key] = ps

        td["ema"]  = ema_store
        td["pair"] = pair_store
        return result, 0, json.dumps(td)
