# Round 5 Group Mining Report

Days analyzed: 2, 3, 4

## Group Ranking

| Rank | Group | Inefficiency Score | Chosen Structure | Structure Kind | Median R^2 | Best Lead-Lag |
| --- | --- | ---: | --- | --- | ---: | --- |
| 1 | PEBBLES | 173.71 | `residual(PEBBLES_XL vs formula)` | formula | 0.999996 | PEBBLES_M -> PEBBLES_XL @ 5 ticks (-0.0177) |
| 2 | TRANSLATOR | 141.06 | `residual(TRANSLATOR_GRAPHITE_MIST vs formula)` | formula | 0.740908 | TRANSLATOR_ECLIPSE_CHARCOAL -> TRANSLATOR_SPACE_GRAY @ 1 ticks (-0.0221) |
| 3 | UV | 138.61 | `residual(UV_VISOR_AMBER vs formula)` | formula | 0.645842 | UV_VISOR_MAGENTA -> UV_VISOR_RED @ 8 ticks (-0.0153) |
| 4 | OXYGEN | 134.73 | `residual(OXYGEN_SHAKE_CHOCOLATE vs formula)` | formula | 0.615867 | OXYGEN_SHAKE_EVENING_BREATH -> OXYGEN_SHAKE_MORNING_BREATH @ 5 ticks (-0.0190) |
| 5 | ROBOT | 108.95 | `residual(ROBOT_LAUNDRY vs formula)` | formula | 0.603458 | ROBOT_VACUUMING -> ROBOT_MOPPING @ 3 ticks (-0.0180) |
| 6 | PANEL | 102.58 | `residual(PANEL_1X2 vs formula)` | formula | 0.720481 | PANEL_2X4 -> PANEL_1X2 @ 10 ticks (0.0234) |
| 7 | GALAXY | 83.10 | `residual(GALAXY_SOUNDS_DARK_MATTER vs formula)` | formula | 0.616499 | GALAXY_SOUNDS_SOLAR_FLAMES -> GALAXY_SOUNDS_PLANETARY_RINGS @ 2 ticks (-0.0175) |
| 8 | MICROCHIP | 81.73 | `residual(MICROCHIP_OVAL vs formula)` | formula | 0.784610 | MICROCHIP_CIRCLE -> MICROCHIP_OVAL @ 5 ticks (0.0193) |
| 9 | SLEEP | 76.10 | `residual(SLEEP_POD_NYLON vs formula)` | formula | 0.686500 | SLEEP_POD_NYLON -> SLEEP_POD_LAMB_WOOL @ 6 ticks (-0.0171) |
| 10 | SNACKPACK | 72.84 | `residual(SNACKPACK_RASPBERRY vs formula)` | formula | 0.900071 | SNACKPACK_VANILLA -> SNACKPACK_PISTACHIO @ 1 ticks (-0.0212) |

## Strongest Cross-Group Lead-Lag

`OXYGEN` leads `SNACKPACK` by 1 ticks with correlation -0.0692 and t-stat -12.01.

## Top 3 Groups

### 1. PEBBLES

Universe: PEBBLES_L, PEBBLES_M, PEBBLES_S, PEBBLES_XL, PEBBLES_XS

Best pair spread: `PEBBLES_S - PEBBLES_XS` | median p-value 0.142116 | max p-value 0.330644 | half-life 574.92 ticks.
Best basket spread: `PEBBLES_L + PEBBLES_XL - 2*PEBBLES_S` | median p-value 0.16565 | max p-value 0.787567 | half-life 1097.04 ticks.
Best hidden formula: `PEBBLES_XL = 49999.939043 + -0.999986*PEBBLES_L + -1.000000*PEBBLES_M + -1.000021*PEBBLES_S + -0.999994*PEBBLES_XS` with median R² 0.999996 and pooled R² 0.999998.
Formula residual stationarity: median p-value 0 | max p-value 0 | half-life 0.70 ticks.
Strongest within-group lead-lag: `PEBBLES_M` -> `PEBBLES_XL` at 5 ticks with correlation -0.0177.
Suppressed-volatility product: `PEBBLES_S` with volatility ratio 0.9979 and non-zero return frequency 0.9812.
Average L1 depth across the group: 12.60. Average trade size: 3.54.

Strategy:
- Signal: Rolling z-score of the residual for `PEBBLES_XL` against its group fair-value formula using a 50-tick window.
- Entry: If residual z > +2.0, short the target leg and buy the hedge basket implied by the formula. If residual z < -2.0, buy the target leg and short the hedge basket.
- Exit: Take profit when |z| < 0.5 or when the spread crosses its rolling mean.
- Stop: Cut the trade if |z| > 3.5 or if the spread fails to mean-revert within the max-hold window.
- Holding: Use a max holding horizon of 20 ticks, roughly 3x the estimated half-life.
- Size: Theoretical regression hedge weights: PEBBLES_L:+1.000, PEBBLES_M:+1.000, PEBBLES_S:+1.000, PEBBLES_XL:+1, PEBBLES_XS:+1.000. Capped integer implementation: PEBBLES_L:+10, PEBBLES_M:+10, PEBBLES_S:+10, PEBBLES_XL:+10, PEBBLES_XS:+10.

### 2. TRANSLATOR

Universe: TRANSLATOR_ASTRO_BLACK, TRANSLATOR_ECLIPSE_CHARCOAL, TRANSLATOR_GRAPHITE_MIST, TRANSLATOR_SPACE_GRAY, TRANSLATOR_VOID_BLUE

Best pair spread: `TRANSLATOR_ASTRO_BLACK - TRANSLATOR_GRAPHITE_MIST` | median p-value 0.111658 | max p-value 0.479014 | half-life 1226.98 ticks.
Best basket spread: `TRANSLATOR_ECLIPSE_CHARCOAL + TRANSLATOR_GRAPHITE_MIST - 2*TRANSLATOR_ASTRO_BLACK` | median p-value 0.119519 | max p-value 0.294728 | half-life 664.24 ticks.
Best hidden formula: `TRANSLATOR_GRAPHITE_MIST = 11826.613181 + -0.194594*TRANSLATOR_ASTRO_BLACK + -0.123566*TRANSLATOR_ECLIPSE_CHARCOAL + 0.083246*TRANSLATOR_SPACE_GRAY + 0.047134*TRANSLATOR_VOID_BLUE` with median R² 0.740908 and pooled R² 0.047848.
Formula residual stationarity: median p-value 5.94237e-05 | max p-value 0.00805319 | half-life 142.86 ticks.
Strongest within-group lead-lag: `TRANSLATOR_ECLIPSE_CHARCOAL` -> `TRANSLATOR_SPACE_GRAY` at 1 ticks with correlation -0.0221.
Suppressed-volatility product: `TRANSLATOR_SPACE_GRAY` with volatility ratio 0.9560 and non-zero return frequency 0.9730.
Average L1 depth across the group: 11.47. Average trade size: 2.46.

Strategy:
- Signal: Rolling z-score of the residual for `TRANSLATOR_GRAPHITE_MIST` against its group fair-value formula using a 714-tick window.
- Entry: If residual z > +2.0, short the target leg and buy the hedge basket implied by the formula. If residual z < -2.0, buy the target leg and short the hedge basket.
- Exit: Take profit when |z| < 0.5 or when the spread crosses its rolling mean.
- Stop: Cut the trade if |z| > 3.5 or if the spread fails to mean-revert within the max-hold window.
- Holding: Use a max holding horizon of 429 ticks, roughly 3x the estimated half-life.
- Size: Theoretical regression hedge weights: TRANSLATOR_ASTRO_BLACK:+0.195, TRANSLATOR_ECLIPSE_CHARCOAL:+0.124, TRANSLATOR_GRAPHITE_MIST:+1, TRANSLATOR_SPACE_GRAY:-0.083, TRANSLATOR_VOID_BLUE:-0.047. Capped integer implementation: TRANSLATOR_ASTRO_BLACK:+2, TRANSLATOR_ECLIPSE_CHARCOAL:+1, TRANSLATOR_GRAPHITE_MIST:+10, TRANSLATOR_SPACE_GRAY:-1, TRANSLATOR_VOID_BLUE:+0.

### 3. UV

Universe: UV_VISOR_AMBER, UV_VISOR_MAGENTA, UV_VISOR_ORANGE, UV_VISOR_RED, UV_VISOR_YELLOW

Best pair spread: `UV_VISOR_AMBER - UV_VISOR_RED` | median p-value 0.0555276 | max p-value 0.531247 | half-life 861.40 ticks.
Best basket spread: `UV_VISOR_AMBER + UV_VISOR_ORANGE - 2*UV_VISOR_RED` | median p-value 0.341004 | max p-value 0.619232 | half-life 1073.94 ticks.
Best hidden formula: `UV_VISOR_AMBER = 31353.284613 + -0.969763*UV_VISOR_MAGENTA + -0.384691*UV_VISOR_ORANGE + -0.617259*UV_VISOR_RED + -0.166634*UV_VISOR_YELLOW` with median R² 0.645842 and pooled R² 0.896354.
Formula residual stationarity: median p-value 0.00127953 | max p-value 0.00242038 | half-life 240.65 ticks.
Strongest within-group lead-lag: `UV_VISOR_MAGENTA` -> `UV_VISOR_RED` at 8 ticks with correlation -0.0153.
Suppressed-volatility product: `UV_VISOR_AMBER` with volatility ratio 0.7273 and non-zero return frequency 0.9671.
Average L1 depth across the group: 18.26. Average trade size: 2.46.

Strategy:
- Signal: Rolling z-score of the residual for `UV_VISOR_AMBER` against its group fair-value formula using a 1203-tick window.
- Entry: If residual z > +2.0, short the target leg and buy the hedge basket implied by the formula. If residual z < -2.0, buy the target leg and short the hedge basket.
- Exit: Take profit when |z| < 0.5 or when the spread crosses its rolling mean.
- Stop: Cut the trade if |z| > 3.5 or if the spread fails to mean-revert within the max-hold window.
- Holding: Use a max holding horizon of 722 ticks, roughly 3x the estimated half-life.
- Size: Theoretical regression hedge weights: UV_VISOR_AMBER:+1, UV_VISOR_MAGENTA:+0.970, UV_VISOR_ORANGE:+0.385, UV_VISOR_RED:+0.617, UV_VISOR_YELLOW:+0.167. Capped integer implementation: UV_VISOR_AMBER:+10, UV_VISOR_MAGENTA:+10, UV_VISOR_ORANGE:+4, UV_VISOR_RED:+6, UV_VISOR_YELLOW:+2.
