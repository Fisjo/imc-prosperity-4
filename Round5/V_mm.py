# IMC Prosperity 4 — Round 5  V_mm  (wide-spread market maker, 50 products)
#
# Captures half-spread on every product independently. The 50-product universe
# combined with ±10 limit per product gives huge aggregate capacity. Many R5
# products (SNACKPACK, OXYGEN_SHAKE) carry BA spreads of 15-18 — that is the
# real edge in this round.
#
# Design (per plan, conservative tier):
#   - fair value: EMA of mid (span = 200)
#   - quotes:    bid = floor(fair) - edge,  ask = ceil(fair) + edge
#                edge = max(1, floor(ba/2) - 1)  → sit one tick inside the touch
#   - skew:      shift quotes by round(0.05 × position)
#   - take:      cross if ask < fair - take_thr  /  bid > fair + take_thr
#   - sizing:    min(8, POS_LIMIT - |pos|)
#   - skip product if order book is one-sided that tick (avoids float-fallback bug)
#   - all Order prices wrapped with int(round(...))

import json, math
from typing import List
from datamodel import OrderDepth, TradingState, Order


POS_LIMIT      = 10
EMA_SPAN       = 200        # smooth fair value
EMA_ALPHA      = 2.0 / (EMA_SPAN + 1.0)
MAX_QUOTE_SIZE = 5          # leaves room for both bid + ask within ±10 limit
SKEW_FACTOR    = 0.2        # pos=10 → skew=2 ticks (light pressure; high values broke volatile families)
MIN_BA_TO_MM   = 10         # narrow-spread products (MICROCHIP/ROBOT) get adversely selected

# Family whitelist — empirically profitable families in the test run with these params.
# Trimmed: PEBBLES + SLEEP_POD lose with non-zero skew (high σ amplifies skew miss).
MM_FAMILIES = {"SNACKPACK", "OXYGEN_SHAKE"}


# Universe (50 products) — listed explicitly so we don't rely on state's order_depths key set.
ALL_PRODUCTS: List[str] = [
    # GALAXY_SOUNDS
    "GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    # MICROCHIP
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE",
    "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE",
    # OXYGEN_SHAKE
    "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH",
    # PANEL
    "PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4",
    # PEBBLES
    "PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS",
    # ROBOT
    "ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY",
    "ROBOT_MOPPING", "ROBOT_VACUUMING",
    # SLEEP_POD
    "SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE",
    # SNACKPACK
    "SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA",
    # TRANSLATOR
    "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE",
    # UV_VISOR
    "UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE",
    "UV_VISOR_RED", "UV_VISOR_YELLOW",
]


class Trader:
    def run(self, state: TradingState):
        td_raw = getattr(state, "traderData", "") or ""
        try:
            td = json.loads(td_raw) if td_raw else {}
        except Exception:
            td = {}

        ema_store = td.get("ema", {})

        # Build the active universe from ALL_PRODUCTS, filtered by family whitelist.
        active_products = [
            p for p in ALL_PRODUCTS
            if any(p.startswith(f + "_") for f in MM_FAMILIES)
        ]

        result = {}

        for prod in active_products:
            od = state.order_depths.get(prod)
            if od is None or not od.buy_orders or not od.sell_orders:
                # one-sided book → skip (and do NOT update EMA from a half-book)
                continue

            best_bid = int(max(od.buy_orders))
            best_ask = int(min(od.sell_orders))
            mid      = (best_bid + best_ask) / 2.0
            ba       = best_ask - best_bid
            if ba <= 0:
                continue

            # ── EMA fair value (always update — even if we don't quote, for re-entry) ──
            prev = ema_store.get(prod)
            fair = mid if prev is None else (EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * float(prev))
            ema_store[prod] = fair

            if ba < MIN_BA_TO_MM:
                continue   # too tight to MM profitably; skip

            pos = int(state.position.get(prod, 0))
            edge = max(1, ba // 2 - 1)              # one tick inside the touch when wide
            skew = int(round(SKEW_FACTOR * pos))    # long → lower both sides; short → raise

            # Quotes — keep inside the existing touch but step away when inventoried.
            bid_px = int(round(fair)) - edge - skew
            ask_px = int(round(fair)) + edge - skew

            # Don't cross self — if our bid is above current best ask, we'd be a taker;
            # IMC accepts that, but we want passive quotes here. Clamp to one tick inside.
            bid_px = min(bid_px, best_ask - 1)
            ask_px = max(ask_px, best_bid + 1)
            # And don't make a no-op (bid >= ask).
            if ask_px <= bid_px:
                ask_px = bid_px + 1

            # Sizes: leave room for both quotes within ±POS_LIMIT.
            buy_size  = min(MAX_QUOTE_SIZE, POS_LIMIT - pos)
            sell_size = min(MAX_QUOTE_SIZE, POS_LIMIT + pos)

            orders: List[Order] = []
            if buy_size > 0:
                orders.append(Order(prod, int(bid_px), int(buy_size)))
            if sell_size > 0:
                orders.append(Order(prod, int(ask_px), -int(sell_size)))

            # ── Aggressive take if a stale level exists across our fair ──
            take_thr = max(2, edge)
            if best_ask < fair - take_thr and (POS_LIMIT - pos) > 0:
                take_qty = min(POS_LIMIT - pos, int(od.sell_orders[best_ask] * -1) if best_ask in od.sell_orders else MAX_QUOTE_SIZE)
                if take_qty > 0:
                    orders.append(Order(prod, int(best_ask), int(take_qty)))
            if best_bid > fair + take_thr and (POS_LIMIT + pos) > 0:
                take_qty = min(POS_LIMIT + pos, int(od.buy_orders[best_bid]) if best_bid in od.buy_orders else MAX_QUOTE_SIZE)
                if take_qty > 0:
                    orders.append(Order(prod, int(best_bid), -int(take_qty)))

            if orders:
                result[prod] = orders

        td["ema"] = ema_store
        return result, 0, json.dumps(td)
