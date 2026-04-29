# Estrategia de Trading: Enfoque Híbrido Intra-Cesta (Round 5)

Para maximizar el número de activos operados respetando el límite estricto de ±10 posiciones —y basándonos en las evidencias empíricas del análisis y el documento *Intra-Basket Reversion*—, descartamos un enfoque único para todo. Obligar a todos los activos a operarse como "pares puros" o con medias estáticas genera falsos positivos y destruye el alpha.

La estrategia maestra es un **Enfoque Híbrido Intra-Cesta** (*Intra-Basket Strategy*). Combinaremos arbitraje de identidad pasivo, pares puros dinámicos y rotaciones de 3 o 4 patas (baskets), dependiendo de la estructura matemática inherente de cada categoría. 

A continuación, se detallan los 4 pilares para implementar esta estrategia en producción.

---

## Pilar 1: Construcción Matemática del Spread

Para poder comparar el riesgo entre un par tradicional (2 activos) y una cesta compleja (4 activos), estandarizaremos la formulación del spread:

*   **Pesos Normalizados (L1-Normalization):** Cada spread se define como una combinación lineal de precios ($s_t = w^T p_t$). Aplicaremos una normalización L1 para que la suma absoluta de los pesos sea siempre 1 ($\sum |w_i| = 1$). Esto garantiza que "1 unidad de spread" consuma exactamente 1 unidad de capacidad del inventario global, permitiendo una gestión de capital uniforme.
*   **Z-Score Congelado (Formation-only Scaling):** La media ($\mu_F$) y la desviación estándar ($\sigma_F$) para normalizar el Z-Score se calculan únicamente con los datos históricos de formación (Días 2 y 3) y se dejan fijos para el Día 4. Actualizar la varianza en tiempo real enmascararía las desviaciones reales del mercado.

---

## Pilar 2: El Motor de Fair Value Dinámico

El análisis demostró que las medias estáticas fallan miserablemente debido a cambios de régimen (deriva estructural). Reemplazaremos la Media Móvil Simple por un modelo Dinámico Causal (similar a un filtro de Kalman de una dimensión):

*   **Innovación Pre-Actualización:** En cada tick, el residuo que operamos es la diferencia entre el precio real del spread hoy y la predicción del *Fair Value* calculada ayer ($u_t = s_t - m_{t|t-1}$). Esto evita introducir un sesgo predictivo (*look-ahead bias*).
*   **Ajuste por Factor $K$:** Una vez capturado el residuo, actualizamos el *Fair Value* para el siguiente tick sumándole una fracción de la desviación actual: $m_{t+1} = m_t + K(s_t - m_t)$. El parámetro $K$ dicta qué tan rápido se adapta el modelo al nuevo régimen (generalmente entre 0.001 y 0.005).

---

## Pilar 3: El Universo Operable (Asignación por Categorías)

Basados en los resultados del análisis, dividiremos el mercado de 50 activos en las siguientes sub-estrategias:

### Categoría 1: Cotización Pasiva de Identidad (Pebbles)
*   **Contexto:** La suma exacta de los 5 Pebbles (XS+S+M+L+XL) es igual a 50,000. Sin embargo, el costo de cruzar el *bid-ask spread* de las 5 patas a la vez destruye el margen de beneficio.
*   **Ejecución:** No operaremos la cesta entera. Usaremos la identidad para calcular el "precio justo" de un solo Pebble asumiendo que los otros 4 están en su valor correcto. Usaremos esto exclusivamente para poner órdenes límite pasivas (*Market Making*) esperando que el mercado nos cruce.

### Categoría 2: El Núcleo Multi-Spread (Snackpack)
*   **Contexto:** Es la categoría con mayor compresión de precios y reversión.
*   **Ejecución:** Operaremos 4 spreads superpuestos simultáneamente: Chocolate/Vanilla (dinámico), Pistachio/Strawberry (dinámico), Raspberry/Strawberry (dinámico) y una cesta de 4 patas (Choc+Van-Pist-Rasp) usando un modelo de tendencia. Al haber activos solapados, sumaremos los objetivos de posición de cada spread antes de enviar la orden final al mercado.

### Categoría 3: Cestas Secundarias Dinámicas (Microchips, Sleep Pods, Translators, Shakes)
*   **Contexto:** Tienen correlaciones de nivel, pero menos compresión de cambios de precio.
*   **Ejecución:** Usaremos el *Fair Value* dinámico ($K$) operando combinaciones específicas comprobadas. Por ejemplo, en Microchips operaremos una cesta de 4 patas (`MICRO_LVL4`) y usaremos el par Oval/Triangle solo como confirmación de señal. En Shakes, ejecutaremos una rotación de 4 patas (`OXY_ECG`) con un $K$ de 0.002.

### Categoría 4: Lista Negra y Observación (UV Visors, Panels, Robots, Galaxy)
*   **Contexto:** Mostraron falsos positivos, nula cointegración real, o dinámicas de trading desastrosas fuera de la muestra.
*   **Ejecución:** No se operan. Se excluyen del sistema de manera estricta para evitar exposición direccional injustificada.

---

## Pilar 4: Reglas de Entrada, Salida y Netting (Gestión del Límite)

Al tener cestas con múltiples patas ponderadas fraccionalmente, el enrutamiento (*routing*) de órdenes debe ser estricto para respetar el límite de ±10 unidades por activo:

*   **Disparadores (Triggers):** Entramos al mercado agresivamente (cruzando el spread) cuando el Z-Score de la innovación pre-actualización supera el umbral óptimo (por ejemplo, > ±1.5 o > ±2.0, según el spread). Salimos aplanando la posición cuando el spread revierte cerca de su media ($|Z| < 0.5$).
*   **Conversión a Enteros:** Traducimos nuestra convicción direccional (el Z-Score) a un tamaño de posición objetivo en "unidades de spread". Como el spread está L1-normalizado, multiplicamos la unidad de spread objetivo por los pesos $w_i$ de cada activo de la cesta, y redondeamos al número entero más cercano.
*   **Filtro de Límite (Limit Breach Protection):** Antes de enviar el bloque de órdenes de una cesta, el sistema suma la posición teórica requerida a la posición actual. Si cualquiera de las patas del spread excede la restricción de ±10, se recorta (*clip*) la orden de esa pata al máximo disponible, y se reescalan las órdenes de las demás patas proporcionalmente para mantener la neutralidad de la cobertura al máximo posible.