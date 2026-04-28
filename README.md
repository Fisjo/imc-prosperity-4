# IMC Prosperity 4

> **IMC Prosperity 4** is IMC Trading's annual global algorithmic and manual trading competition for STEM students, running from April 14 to April 30, 2026. Teams of up to five compete across five rounds — each consisting of one algorithmic challenge and one manual challenge — maximising profit in the competition's virtual currency, **XIRECs**.

---

## Results

| Round | Algo | Manual | Global rank | Spain rank |
|-------|------|--------|-------------|------------|
| Tutorial |  |  | **416** (Unofficial) | - |
| Round 1 | 102763 (203rd) | 87995 (1st) | **177** (top 0.8%)| **5**  |
| Round 2 | 0 (strategic) | 24233 (safe bet) | **3882** | **30** |
| Round 3 | 59993 (575th) | 70895 (430th) | **603** | **5**|
| Round 4 | 186103 (84th) | 34242 (586th) | **232** | **1** |

- Reached **global #2** for a full day during the First Round.
- Secured the **optimal manual answer** in the First Round.
- Round 2: only ~10K XIRECs short of the advancement threshold after Round 1 — we submitted an empty algo trader and deployed the guaranteed minimax strategy on the manual challenge to bank a safe +24K profit and guarantee Phase 2 while studying for next challenges.

---

## Table of Contents

- [Competition structure](#competition-structure)
- [Repository structure](#repository-structure)
- [Algorithmic trading](#algorithmic-trading)
  - [Tutorial Round](#tutorial-round--emeralds--tomatoes)
  - [Round 1](#round-1--ash_coated_osmium--intarian_pepper_root)
- [Manual challenges](#manual-challenges)
  - [Round 2 — Invest & Expand](#round-2--invest--expand)
- [Setup & usage](#setup--usage)
- [Dependencies](#dependencies)

---

## Competition structure

Five rounds of ~2 days each. Every round introduces new tradeable products and increases market complexity. Each team submits a `Trader` class in Python evaluated independently against the full field.

| Round | Products | Core theme |
|-------|----------|------------|
| Tutorial | `EMERALDS`, `TOMATOES` | Market making fundamentals |
| Round 1 | `ASH_COATED_OSMIUM`, `INTARIAN_PEPPER_ROOT` | Stationary MM + trend following |
| Round 2 | TBD | TBD |
| Round 3 | TBD | TBD |
| Round 4 | TBD | TBD |
| Round 5 | TBD | TBD |

---

## Repository structure

```
imc-prosperity-4/
│
├── bots/
│   ├── TutorialRound/
│   │   ├── prosperity4_trader_v15_1.py   # AS-skew MM — γ=7, Sharpe 6.77
│   │   ├── prosperity4_trader_v21.py     # Aggressive phase-3 with spoof radar
│   │   └── prosperity4_trader_v22.py     # Final submission — phased warmup + V21 aggression
│   └── Round1/
│       ├── prosperity4-round1-v1.py      # Initial framework (Take/Clear/Make + trend)
│       └── prosperity4-round1-v7.py      # Final submission — volume imbalance FV + multi-level
│
├── logs/
│   ├── AryanV1/                          # Backtest logs — AryanV1 variant
│   ├── BestV-1/                          # BestV-1 benchmark run
│   ├── BestV0/                           # BestV0 benchmark run (Round 1 v7)
│   ├── ManuV2/                           # ManuV2 variant evaluation
│   └── logTraderV15/                     # V15 tutorial backtest (baseline)
│
├── manual-challenge/
│   ├── invest-optimizer.py               # Full PnL optimizer — differential evolution + SLSQP
│   └── visualizers/
│       ├── invest_expand_optimizer.html      # Interactive allocation explorer
│       ├── strategy_comparison.html          # Strategy comparison dashboard
│       └── assumption_sensitivity_full.html  # 4-tab sensitivity analysis
│
├── ROUND_1.zip                           # Platform export — Round 1
├── TUTORIAL_ROUND_1.zip                  # Platform export — Tutorial Round
└── README.md
```

---

## Algorithmic trading

All bots implement the standard Prosperity interface:

```python
class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        ...
        return result, conversions, trader_data
```

State is fully serialised in `traderData` (JSON string) to persist EMA values, tick counters, and accumulation flags across calls.

---

### Tutorial Round — EMERALDS & TOMATOES

**Products.**

- `EMERALDS` — stable fair value anchored at **10,000 XIRECs**, near-zero drift.
- `TOMATOES` — mean-reverting price with noisy short-term moves and exploitable order book structure.

**Version history.**

| Version | Key change | Backtest PnL | Sharpe |
|---------|-----------|-------------|--------|
| V15.1 | Avellaneda-Stoikov skew, γ=7, unwind at pos>35 | +30,749 | 6.77 |
| V21 | Aggressive phase-3: γ=5, take threshold 2.0 ticks, spoof-conditional unwind | Higher raw PnL | Higher drawdown |
| **V22** (final) | Phased warmup: V15.1 conservative for ticks 1–1000, V21 aggression from tick 1001+ | Best of both | Controlled |

**Final bot — V22.**

V22 runs a three-phase regime on TOMATOES:

```
Phase 1  (ticks   1–  50):  EMA warm-up only — sniper active, no maker or unwind
Phase 2  (ticks  51–1000):  Conservative AS market making (γ=7, unwind thresh=35)
Phase 3  (ticks 1001+   ):  Aggressive mode — γ=5, threshold 2.0 ticks, spoof-aware unwind
```

The phased design addresses a core trade-off: V21's aggression maximises edge in stable conditions but has a 33% max drawdown before the spoof signal is reliable. V22 preserves V15.1's conservative behaviour during the critical early ticks (when submissions are being evaluated and spoof patterns are not yet established), then unlocks V21's edge once the market has been observed for 1,000 ticks.

*Sniper.* Crosses the spread when the best quote deviates from EMA fair value by more than a threshold — 3.0 ticks in Phase 2, 2.0 ticks in Phase 3.

*Avellaneda-Stoikov maker.* The reservation price skews the mid-market quote to account for inventory risk:

```
reservation = fair − γ × (position / position_limit)
```

A long position pushes the reservation downward, making asks more competitive and discouraging additional buying. γ=7 in Phase 2, γ=5 in Phase 3.

*Spoof radar (Phase 3 only).* Order book imbalance is computed from levels 2–3 on each side, filtering out the noisy best quote:

```python
spoof_imb = (bids_L2_3 − asks_L2_3) / (bids_L2_3 + asks_L2_3)
```

A strong negative imbalance (sell-side pressure from non-best levels) shifts the dynamic fair value upward by +2 ticks; a strong positive imbalance shifts it down by −1.5 ticks. Unwind orders are only sent when the spoof signal confirms the intended direction, avoiding adverse inventory-increasing fills.

*Unwind.* Phase 2: aggressive spread-crossing when |position| > 35. Phase 3: conditional on spoof signal alignment and spread ≤ 8 ticks.

**EMERALDS.** Takes mispriced levels against the anchor fair value of 10,000. Quotes with a 2-tick edge and a light inventory skew factor. This logic was fixed at V6 and left unchanged across all 22 iterations — mean position ≈ −0.07 in backtests.

---

### Round 1 — ASH_COATED_OSMIUM & INTARIAN_PEPPER_ROOT

**Products.**

- `ASH_COATED_OSMIUM` — stationary fair value near **10,001 XIRECs**. Pure market making with a volume imbalance overlay.
- `INTARIAN_PEPPER_ROOT` — deterministic linear price drift: `FV = base + 0.001 × timestamp`. Each full trading day adds approximately **+1,000 XIRECs** to fair value.

**Version history.**

| Version | Key change |
|---------|-----------|
| V1 | Take/Clear/Make for Osmium; accumulate-and-hold for Pepper |
| **V7** (final) | Volume imbalance FV shift for Osmium (measured 0.50 correlation with next-tick returns); multi-level taking; position-aware quote shift; mean-reversion-aware Pepper MM |

**Final bot — V7.**

*ASH_COATED_OSMIUM.*

Dynamic fair value using the full-book volume imbalance signal:

```python
imbalance = (total_bid_vol − total_ask_vol) / (total_bid_vol + total_ask_vol)
```

This signal has a measured **0.50 correlation with next-tick returns**, providing a directional edge on top of standard market making. Position-dependent quote shifting adjusts both bid and ask simultaneously, accelerating inventory reversion without requiring emergency spread-crossing:

```python
pos_shift = round((pos / limit) × 3)   # ±3 ticks at max inventory
our_bid   = min(best_bid + 1,  FV − 1 − pos_shift)
our_ask   = max(best_ask − 1,  FV + 1 − pos_shift)
```

Multi-level taking sweeps all mispriced levels in a single tick, improving fill quality relative to the V1 single-level approach.

*INTARIAN_PEPPER_ROOT.*

The linear drift is calibrated at initialisation against the first observed mid-price:

```python
pepper_base = mid_0 − 0.001 × timestamp_0
FV(t)       = pepper_base + 0.001 × t
```

**Phase 1 — Accumulation.** Sweeps L1 ask liquidity until the position limit of 80 is reached, then places an aggressive bid one tick above the best bid to fill remaining capacity.

**Phase 2 — Hold and earn.** Holds the full long position and captures deterministic appreciation (~80,000 XIRECs over a full session). A sell-side MM overlay provides incremental edge: asks are placed at `max(best_ask − 1, FV + 3)`, generating fills only when the market temporarily overshoots fair value, while keeping the minimum holding position at 40 units.

---

## Manual challenges

### Round 2 — Invest & Expand

**Challenge.** Teams received 50,000 XIRECs to distribute across three pillars — Research, Scale, and Speed — as percentages summing to ≤ 100%. The payoff formula is:

```
PnL = Research(r%) × Scale(s%) × Speed_multiplier − Budget_Used
```

Where:

```
Research(x) = 200,000 × ln(1 + x) / ln(101)    # logarithmic, diminishing returns
Scale(x)    = 7x / 100                           # linear
Speed_mult  = 0.1 + 0.8 × CDF(your_speed; μ, σ) # rank-based across all teams
```

**The core difficulty.** Speed is rank-based — its value is determined by where every other team invests, making this a game-theoretic problem. The optimal allocation depends on an unobservable crowd distribution.

**Our analysis.**

*Stage 1 — Inner optimum.* For any fixed Speed budget, the optimal Research/Scale split satisfies:

```
d/dx [ln(1+x) · (C−x)] = 0   ⟹   (C−x)/(1+x) = ln(1+x)
```

This yields Research ≈ 13% of the remaining budget (after Speed), with Scale absorbing the rest. Research's logarithmic curve flattens rapidly — investing 60% only doubles the output of 13% — making Scale the better marginal investment at almost any budget level.

*Stage 2 — Probabilistic Speed modelling.* We modelled competitor Speed allocations as Normal, Student-t (df=2), and Cauchy (df=1) distributions centred on an estimated crowd median. All three distributions converge to the same allocation:

| Distribution | Research | Scale | Speed | Net PnL |
|-------------|----------|-------|-------|---------|
| Gaussian (μ=40%, σ=8%) | 13.2% | 37.7% | 49.1% | 192,103 XIRECs |
| Student-t df=2 | 13.4% | 38.5% | 48.1% | 178,288 XIRECs |
| Cauchy df=1 | 13.6% | 39.2% | 47.2% | 168,898 XIRECs |

*Stage 3 — Sensitivity analysis.* Because the crowd median is unobservable, we stress-tested across μ ∈ [10%, 70%] and σ ∈ [4%, 25%]. The probabilistic model dominates for μ ∈ [30%, 55%] but degrades if the crowd is passive (μ < 25%). A Monte Carlo simulation over 50,000 draws from a meta-distribution over (μ, σ) confirmed a **94.5% probability of positive PnL** and a mean of ~159K XIRECs for the probabilistic allocation.

**Our Round 2 decision.** Being only ~10K XIRECs short of the advancement threshold, we opted for certainty over optimisation: empty algo trader + **minimax manual allocation**:

```
Research: 23%  ·  Scale: 77%  ·  Speed: 0%
```

This guarantees approximately **+24,000 XIRECs profit regardless of any competitor's behaviour**, by accepting the 0.1× Speed multiplier floor and directing the full budget to the fully deterministic Research × Scale product. Expected PnL variance: zero.

**Tools.**

| File | Description |
|------|-------------|
| `manual-challenge/invest-optimizer.py` | Full optimizer — differential evolution + SLSQP refinement, three distribution models, sensitivity grid, Monte Carlo |
| `manual-challenge/visualizers/assumption_sensitivity_full.html` | Interactive 4-tab dashboard: assumption explorer, PnL heatmap, strategy comparison, minimax analysis |
| `manual-challenge/visualizers/invest_expand_optimizer.html` | Allocation explorer with live PnL and speed multiplier calculation |
| `manual-challenge/visualizers/strategy_comparison.html` | Head-to-head PnL comparison of four candidate strategies across crowd assumptions |

```bash
# Run the full analysis
python manual-challenge/invest-optimizer.py

# Adjust the crowd assumption at the bottom of the file:
# CROWD_MEDIAN = 40.0   ← estimated average competitor Speed allocation (%)
# SPREAD       = 8.0    ← estimated standard deviation
```

---

## Setup & usage

```bash
git clone https://github.com/Fisjo/imc-prosperity-4.git
cd imc-prosperity-4
pip install numpy scipy
```

The bot files are self-contained and require only the `datamodel` module provided by the Prosperity platform at submission time. The manual challenge tools run independently with standard scientific Python libraries.

To open the interactive visualisers, navigate to `manual-challenge/visualizers/` and open any `.html` file directly in a browser — no server required.

---

## Dependencies

```
numpy>=1.24
scipy>=1.11
```

---

## Disclaimer

This repository is for educational and portfolio purposes. All strategies and analyses were developed independently by the team. The competition is hosted by IMC Trading at [prosperity.imc.com](https://prosperity.imc.com). This project is not affiliated with or endorsed by IMC Trading.

---

*IMC Prosperity 4 · April 14–30, 2026*