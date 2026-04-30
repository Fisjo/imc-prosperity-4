# Trading Strategy: Intra-Basket Hybrid Approach (Round 5)

This document outlines the master execution plan for the Round 5 simulated market. By moving away from static means and embracing a **Dynamic Causal Model**, we aim to maximize alpha while strictly respecting the $\pm 10$ position limit per asset.

---

## I. Strategic Foundation

The market is non-uniform, consisting of four distinct structural behaviors. Our strategy replaces "one-size-fits-all" pairs trading with a **Hybrid Intra-Basket Approach**:

1.  **Strict Identity Arbitrage:** Deterministic mathematical relationships with instant reversion but razor-thin margins.
2.  **Compact Relationship Spaces:** Strongly coupled groups with high variance but tight co-movement.
3.  **Secondary Level Relationships:** Long-term block movements with significant short-term independent noise.
4.  **Random Walks (Noise):** Unstructured assets that must be blacklisted to avoid inventory traps.

---

## II. Mathematical Framework

### 1. Spread Construction & L1-Normalization
To compare risk across different basket sizes (e.g., a 2-asset pair vs. a 4-asset basket), we use **L1-Normalization**:
$$\sum |w_i| = 1$$
This ensures that "1 unit of spread" consistently consumes exactly 1 unit of global inventory capacity, regardless of the number of legs involved.

### 2. Dynamic Fair Value (Kalman-Style Filter)
Static means lead to "false positives" during regime shifts. We employ a **Dynamic Fair Value ($m_t$)** updated at every tick:
* **Pre-Update Innovation:** $u_t = s_t - m_{t|t-1}$ (Operating on the residual between current price and yesterday's prediction).
* **Adjustment Factor ($K$):** $m_{t+1} = m_t + K(s_t - m_t)$.
* The parameter $K$ (ranging from $0.001$ to $0.005$) determines the adaptation speed to new structural drifts.

---

## III. Tactical Execution by Category

### 1. SNACKPACK (Primary Profit Engine)
* **Behavior:** Most profitable category with the highest mathematical compression.
* **Execution:** Overlapping Dynamic Pairs.
    * **Chocolate vs. Vanilla** ($K  pprox 0.002 - 0.005$)
    * **Pistachio vs. Strawberry** ($K  pprox 0.002 - 0.005$)
    * **Raspberry vs. Strawberry** ($K  pprox 0.002 - 0.005$)
* **Netting Rule:** Sum position intentions for overlapping assets (e.g., Strawberry) before routing orders to stay within $\pm 10$ limits.
* **Entry:** Z-Score $> 1.5$ to $2.0$.

### 2. MICROCHIPS (Structural Rotation)
* **Behavior:** Strong long-term coupling but high daily noise. Static $ eta$ is ineffective here.
* **Execution:** 4-Leg Basket (`MICRO_LVL4`: Circle, Oval, Rectangle, Triangle).
* **Entry:** Conservative approach; Z-Score $> 2.0$.
* **Note:** Use the Oval/Triangle pair only as a confirmation filter, not a primary trigger.

### 3. SLEEP PODS & TRANSLATORS (Tactical Secondary Pairs)
* **Behavior:** Slower reversion cycles. Noise takes longer to dissipate.
* **Execution:** Slow Dynamic Models ($K = 0.002$).
    * **Sleep Pods:** Polyester/Cotton and Suede/Polyester/Cotton combination.
    * **Translators:** Charcoal/Void_Blue and the 4-leg `TRANS_SEG` basket.
* **Limit Management:** Fractional entry scaling (e.g., blocks of 3-4 units) to avoid tying up inventory in slow-moving trades.

### 4. OXYGEN SHAKES (Clean Rotation)
* **Behavior:** Statistical signatures are clearer in 4-asset combinations than in pure pairs.
* **Execution:** `OXY_ECG` Basket (Morning Breath, Evening Breath, Chocolate, Garlic).
* **Entry:** Z-Score $> 2.0$ to ensure the move covers the cost of crossing the bid-ask spread.

### 5. PEBBLES (Passive Market Making)
* **Behavior:** Programmed identity where $\sum 	ext{Pebbles} = 50,000$.
* **Execution:** **Passive Liquidity Provision only.**
    * **Do Not Cross the Spread:** The cost of "taking" liquidity on 5 legs is $ pprox 30x$ the standard deviation of the identity.
    * **Tactic:** Calculate Fair Value for one Pebble ($50,000 - \sum 	ext{Others}$) and place limit orders. Let market noise fill your orders at the "fair" price.

### 6. BLACKLIST (UV Visors, Panels, Robots, Galaxy)
* **Status:** **Strictly Excluded.**
* **Reasoning:** Historical analysis shows zero real cointegration or high rates of false positives. Trading these assets creates unjustified directional exposure and wastes inventory.

---

## IV. Risk & Limit Management

* **Formation-only Scaling:** $\mu$ and $\sigma$ for Z-Scores are frozen based on Day 2 & 3 data to prevent "volatility masking" during Day 4.
* **Limit Breach Protection:** 1. Calculate target position: $Z 	ext{ conviction} 
ightarrow 	ext{Spread Units} 	imes w_i$.
    2. Check against current inventory ($\pm 10$).
    3. If a breach is detected, **Clip** the offending leg and **Rescale** the other legs proportionally to maintain delta-neutrality.
* **Exit Strategy:** Flatten positions when the pre-update innovation Z-score returns to $|Z| < 0.5$.