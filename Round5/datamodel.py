"""
Official IMC Prosperity 4 – datamodel.py  (Round 5 compatible)
This is the canonical datamodel shipped with the competition SDK.
Drop the official version from the competition portal here if you have it;
this file is a complete, API-compatible stub.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# ── Type aliases ──────────────────────────────────────────────────────────────
Symbol  = str
Product = str
UserId  = str


# ── Order ─────────────────────────────────────────────────────────────────────
class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol   = symbol
        self.price    = price
        self.quantity = quantity   # positive = BUY, negative = SELL

    def __str__(self) -> str:
        return (f"Order('{self.symbol}', {self.price}, {self.quantity})")

    def __repr__(self) -> str:
        return self.__str__()


# ── OrderDepth ────────────────────────────────────────────────────────────────
class OrderDepth:
    def __init__(self) -> None:
        self.buy_orders:  Dict[int, int] = {}   # {price: +volume}  (bids)
        self.sell_orders: Dict[int, int] = {}   # {price: +volume}  (asks)

    def __repr__(self) -> str:
        return f"OrderDepth(bids={self.buy_orders}, asks={self.sell_orders})"


# ── Trade ─────────────────────────────────────────────────────────────────────
class Trade:
    def __init__(
        self,
        symbol:    Symbol,
        price:     int,
        quantity:  int,
        buyer:     Optional[UserId] = None,
        seller:    Optional[UserId] = None,
        timestamp: int = 0,
    ) -> None:
        self.symbol    = symbol
        self.price     = price
        self.quantity  = quantity
        self.buyer     = buyer
        self.seller    = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return (
            f"Trade('{self.symbol}', {self.price}, {self.quantity}, "
            f"'{self.buyer}', '{self.seller}', {self.timestamp})"
        )

    def __repr__(self) -> str:
        return self.__str__()


# ── Listing ───────────────────────────────────────────────────────────────────
@dataclass
class Listing:
    symbol:       Symbol
    product:      Product
    denomination: str = "SEASHELLS"


# ── Observations ──────────────────────────────────────────────────────────────
@dataclass
class ConversionObservation:
    bidPrice:            float = 0.0
    askPrice:            float = 0.0
    transportFees:       float = 0.0
    exportTariff:        float = 0.0
    importTariff:        float = 0.0
    sugarPrice:          float = 0.0
    sunlightIndex:       float = 0.0


class Observation:
    def __init__(
        self,
        plainValueObservations:  Dict[Product, int]                     = None,
        conversionObservations:  Dict[Product, ConversionObservation]   = None,
    ) -> None:
        self.plainValueObservations  = plainValueObservations  or {}
        self.conversionObservations  = conversionObservations  or {}


# ── TradingState ──────────────────────────────────────────────────────────────
class TradingState:
    def __init__(
        self,
        traderData:    str,
        listings:      Dict[Symbol, Listing],
        order_depths:  Dict[Symbol, OrderDepth],
        own_trades:    Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position:      Dict[Product, int],
        observations:  Observation,
    ) -> None:
        self.traderData    = traderData
        self.listings      = listings
        self.order_depths  = order_depths
        self.own_trades    = own_trades
        self.market_trades = market_trades
        self.position      = position
        self.observations  = observations
        self.timestamp: int = 0    # set externally by the backtester

    def toJSON(self) -> str:
        return "{}"   # stub – not required for backtesting

    def __repr__(self) -> str:
        return f"TradingState(t={self.timestamp}, pos={self.position})"