# Reto: estadísticos de los clientes + LDA robusto

**Estado:** confirmada con experimentos propios (simulador `scripts/reto_ataques.py`).

## Contexto

Escenario del reto: 100 clientes, Dirichlet α = 0,05 (cada cliente tiene ~22 clases muy
desequilibradas), 20 rondas con 10 elegidos y ~20 % de abandonos, 160 KiB de subida por cliente y
ronda, y hasta un 20 % de clientes maliciosos (etiquetas manipuladas; envíos invertidos,
exagerados o con ruido). Los datos son embeddings de 384 dimensiones de una red preentrenada, 100
clases. La clasificación (21/11/2026): FedAvg 38,87 %; «FedAvg robusto» de referencia 67,46 %.

Dos hechos condicionan todo:

1. **Con 20 rondas solo se llega a ver a ~80 de los 100 clientes** (semillas 0–5: 77–86), y
   cada uno casi siempre una o dos veces. Cualquier método está limitado a los datos de esos
   clientes.
2. **Las reglas permiten subir estadísticos de un cliente** («Los estadísticos de un cliente sí»),
   y los datos ya son features preentrenadas.

## Decisión

- **Cliente** (`fl/clients/estadisticos.py`): no entrena con SGD. Sube las cuentas por clase, la
  suma de los embeddings de cada clase con al menos 5 ejemplos y su dispersión intra-clase
  normalizada (384 × 384), más un `control = 1`.
- **Codec** (`fl/codecs/estadisticos.py`): dispersión simétrica → triángulo superior en int8 con
  escala por fila; sumas en float16; diagonal en float32 como comprobación. Peor caso
  154.196 B < 163.840 B.
- **Agregador** (`fl/aggregators/lda_robusto.py`): acumula lo recibido por `cid` en todas las
  rondas y calcula en forma cerrada el discriminante lineal de Fisher (LDA) con covarianza
  común regularizada (shrinkage 0,01). Antes filtra a los clientes en dos pasos:
  1. **Integridad.** `control` revela un factor k (exagerado; k = −1 si está invertido) y se
     deshace. Si hay ruido falla alguna comprobación: cuentas enteras que suman `n_samples`,
     diagonal en float32 igual a la del int8, dispersión semidefinida positiva y traza parecida a
     la del resto. El cliente se descarta para siempre (los atacantes lo son toda la simulación).
  2. **Coherencia de etiquetas.** Con la métrica de Mahalanobis, la media de cada clase de un
     cliente debe quedar (a) más cerca de la media de esa clase en el resto de clientes que de
     cualquier otra y (b) a la distancia esperable por azar, d·(1/n + 1/N) (z < 3). Si falla en
     más de la mitad de sus ejemplos se descarta el cliente; si falla en alguna clase suelta, solo
     esa clase. Se itera 3 veces porque las referencias iniciales incluyen a los atacantes.

## Justificación

- **Inmune al no-IID.** Un clasificador en forma cerrada calculado con estadísticos suficientes da
  lo mismo con cualquier reparto. Es la idea de Fed3R (ridge regression), FedNCM (medias de clase)
  y AFL (mínimos cuadrados). Se elige LDA en vez de ridge porque, con los mismos estadísticos, da
  más accuracy (centralizado 77,3 % frente a 76,7 %; con los clientes vistos 77,3 % frente a
  75,1 %).
- **El orden y las repeticiones no importan.** Se recalcula con todo lo acumulado; un cliente que
  llega en la ronda 19 cuenta igual que uno de la ronda 1.
- **Robustez por diseño, no solo por agregación robusta.** Las reglas tipo mediana, Krum o media
  recortada fallan con datos heterogéneos aunque no haya atacantes (Karimireddy et al., 2022).
  Aquí la mayor parte de la protección viene de la estructura de lo que se sube: la dispersión no
  depende de los nombres de las etiquetas (una permutación no la cambia) y las cuentas son enteros
  conocidos, lo que hace detectable cualquier manipulación numérica.
- **Rápido.** Sin SGD: ~6–9 s por semilla en un portátil (el límite es un minuto).

## Resultados (validación de `reto.py`, 3–5 semillas)

| Escenario | FedAvg de la práctica | LDA robusto |
|---|---|---|
| Sin atacantes | 70,0 % | **77,2 %** |
| 20 % atacantes, ataques mezclados | 45,4 % | **76,8 %** |
| Solo invertir / exagerar ×10 | 58,1 % / 39,6 % | **77,2 %** (se recuperan exactos) |
| Etiquetas permutadas | 63,5 % | **76,9 %** |
| 30 % atacantes, ataques también sobre enteros | — | **76,8 %** |
| 20 % «todas las etiquetas a la clase 0» | — | **76,8 %** |

Ningún cliente honrado se ha marcado como sospechoso en ninguna prueba. Lo que se pierde frente al
caso sin ataques son los datos de los propios atacantes.

## Alternativas descartadas

- **FedAvg + agregación robusta (mediana, Krum, recorte de normas).** Es la referencia de la
  clasificación (67,46 %) y queda por debajo incluso sin atacantes (70 %).
- **Afinar con SGD federado tras el LDA (como Fed3R+FT).** Probado: con los clientes que se ven
  dos veces no mejora (±0,2 puntos en 10 rondas simuladas).
- **Regresión logística.** Centralizada llega al ~81 %, pero en federado necesita muchas rondas;
  el gradiente completo a partir del LDA tampoco mejora en pocas iteraciones.
- **Normalizar los embeddings (centrar + L2).** +0,3 puntos en centralizado; exige un modelo
  distinto y conocer el centro antes de la primera ronda. No compensa.

## Riesgos

- No sabemos cómo implementa el profesor los ataques. Se han probado variantes (sobre todos los
  tensores o solo los float, escalas de 1,5 a 1000, ruido, permutación, desplazamiento, etiquetas
  aleatorias, colapso a una clase, solo una clase cambiada).
- Se usa `n_samples` del `ClientUpdate` para comprobar las cuentas. FedAvg ya lo usa como peso,
  así que forma parte de lo que recibe el servidor; no se usa `metrics` ni ningún otro canal.
- Clases cuyos únicos dueños nunca son elegidos: no se pueden aprender con ningún método.

## Fuentes

- Fanì, Camoriano, Caputo y Ciccone. *Accelerating Heterogeneous Federated Learning with
  Closed-form Classifiers* (Fed3R). ICML 2024. https://arxiv.org/abs/2406.01116
- Legate, Bernier, Caccia, Oyallon y Belilovsky. *Guiding The Last Layer in Federated Learning with
  Pre-Trained Models* (FedNCM). NeurIPS 2023. https://arxiv.org/abs/2306.03937
- He et al. *AFL: A Single-Round Analytic Approach for Federated Learning with Pre-trained Models*.
  CVPR 2025. https://arxiv.org/abs/2405.16240
- Karimireddy, He y Jaggi. *Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing*.
  ICLR 2022. https://arxiv.org/abs/2006.09365
- Fisher (1936), *The use of multiple measurements in taxonomic problems*; Hastie, Tibshirani y
  Friedman, *The Elements of Statistical Learning*, §4.3 (LDA y LDA regularizado).

---

# Versión 2: RDA (análisis discriminante regularizado)

**Estado:** confirmada con el simulador. Es la configuración actual de `configs/reto.yaml`.

## Contexto

El LDA supone que **todas las clases tienen la misma forma** (una única covarianza). Es la causa
principal de la distancia con la regresión logística centralizada (~82 %). Las tres opciones:

| Modelo | Forma de cada clase | Centralizado (clientes vistos) | En el reto |
|---|---|---|---|
| LDA | la misma para todas | 77,3 % | versión 1 |
| QDA | una propia por clase | 60–76 % (pocos datos para estimarla) | no cabe: ~75 KB por clase |
| RDA | mezcla: (1 − α)·común + α·propia | **82,8 %** (α = 0,1–0,25) | versión 2, en 99 direcciones |

## Decisión

- **Direcciones:** cuando hay `fijar_con = 30` clientes fiables, el servidor fija las 99
  direcciones en las que más se separan las medias de las clases (las del propio LDA, en espacio
  blanqueado). Se mandan dentro del modelo global (buffer `proyeccion`) y no cambian más.
- **Cliente:** con el modelo `rda`, manda lo del LDA solo la primera vez. Si el modelo ya trae
  direcciones, añade la forma de sus clases (≥ 10 ejemplos) proyectada en ellas, las que quepan en
  el presupuesto, empezando por las que más ejemplos tienen.
- **Codec:** cada forma igual que la dispersión (triángulo en int8 con escala por fila). ~5,3 KB
  por clase.
- **Agregador (`RDARobusto`, hereda de `LDARobusto`):** integridad y coherencia como en la v1, más
  comprobaciones de las formas (control, cuentas, clases enteras y válidas, autovalores ≥ 0). Junta
  las formas de cada clase igual que la dispersión del LDA y las mezcla con la común (α = 0,3).
- **Modelo (`ModeloRDA`):** la parte lineal es el LDA y se le resta una corrección cuadrática por
  clase en las 99 direcciones. Es exactamente el clasificador gaussiano con la covarianza de cada
  clase = la común fuera de las direcciones y la mezclada dentro (identidad de Woodbury).

## Resultados (validación de `reto.py`)

| Escenario | v1 (LDA) | v2 (RDA) |
|---|---|---|
| Sin atacantes, semillas 0–2 | 77,2 % | **80,6 %** |
| Sin atacantes, semillas 3–5 (no usadas para ajustar) | 76,9 % | **80,2 %** |
| 20 % atacantes, `mezcla` | 76,9 % | **80,4 %** |
| 20 % atacantes, `mezcla` sobre enteros también | 76,8 % | **80,1 %** |
| 30 % atacantes | 76,8 % | **79,3 %** |
| «Todo a la clase 0» | 76,8 % | **80,2 %** |

Ningún cliente honrado marcado en ninguna prueba.

## Ajustes probados

- `fijar_con`: 5 → 78,4 %; 10 → 79,1 %; 30 → **80,6 %**; 50 → 80,5 %; 60 → 80,1 %. Fijar las
  direcciones pronto deja que más clientes manden formas, pero con pocas clases vistas las
  direcciones son peores; pesa más lo segundo.
- `mezcla` (con `fijar_con = 30`): 0,2 → 80,4 %; **0,3 → 80,6 %**; 0,4 → 80,4 %; 0,5 → 80,1 %.
- Solo la diagonal de cada forma (sin direcciones): no mejora (77,3 %).

---

# Versión 3: más direcciones (160)

**Estado:** confirmada con el simulador.

Con 99 direcciones (las que separan las medias) la forma de cada clase solo se mira en una parte
del espacio. En centralizado, añadir direcciones al azar (perpendiculares a esas) mejora casi tanto
como elegirlas mirando las formas: 99 → 81,5 %; 99 + 29 al azar → 82,0 %; las 384 → 82,3 %. Cada
forma ocupa más (160 direcciones: ~13,5 KB por clase en vez de ~5,3 KB), así que cada cliente manda
menos formas por subida; el equilibrio sale a favor.

| Direcciones (semillas 0–2) | 99 | 128 | 160 | 200 | 256 |
|---|---|---|---|---|---|
| Accuracy | 80,7 % | 80,9 % | 81,1 % | 81,1 % | 80,3 % |

Con 160 direcciones se reajusta `mezcla`: 0,2 → 81,5 %; **0,25 → 81,4 %**; 0,3 → 81,1 %;
0,4 → 80,7 % (semillas 0–2). Se elige 0,25 y se confirma con semillas no usadas para ajustar.

| Escenario | v2 (99, α = 0,3) | **v3 (160, α = 0,25)** |
|---|---|---|
| Sin atacantes, semillas 3–8 | 80,2 % | **81,2 %** |
| 20 % atacantes, `mezcla` | 80,4 % | **81,0 %** |
| 20 % atacantes, ruido | 80,1 % | **80,6 %** |
| 20 % atacantes, permutar | 80,1 % | **80,6 %** |
| 30 % atacantes (también sobre enteros) | 79,3 % | 79,3 % |
| «Todo a la clase 0» | 80,2 % | **80,5 %** |

Probado y descartado en esta versión: mezcla adaptativa según los ejemplos de cada clase (no
mejora: el 85 % de los ejemplos ya llega con forma y todas las clases tienen alguna) y normalizar
los embeddings (centrar + L2: +0,02 puntos con RDA).

---

# Versión 4: RDA de bajo rango (la forma de cada clase en sus propias direcciones)

**Estado:** confirmada con el simulador. Es la configuración actual de `configs/reto.yaml`.

## Diagnóstico

Con la v3, ni el presupuesto ni la cuantización limitaban: sin límite de subida y sin int8 sale
lo mismo (81,4 %). Lo que limitaba era mirar la forma de las clases en solo 160 direcciones
**comunes**. Con la forma completa (384) y sin límite de subida, el federado llega a **82,4 %**
(α = 0,1), al nivel de la regresión logística centralizada (~82,3 %).

La forma completa no cabe (~74 KB por clase), pero no hace falta: en centralizado, quedarse con
las r direcciones en las que **cada clase** más se aparta de la forma común da casi lo mismo
(r = 8: 82,1 %; 16: 82,3 %; 32: 82,7 %; completa: 82,7 %).

## Decisión

- **Blanqueo:** cuando hay 30 clientes fiables, el servidor fija la matriz que convierte la forma
  común en una esfera (Σ^-1/2) y la manda dentro del modelo (buffer `blanqueo`).
- **Cliente:** para cada clase (≥ 10 ejemplos), calcula su forma en el espacio blanqueado y manda
  las **16 direcciones** en las que más se aparta de la esfera (autovectores) y cuánto
  (autovalores). ~6,3 KB por clase, frente a 13,5 KB de la v3.
- **Codec:** vectores en int8 con una escala por vector; valores en float32.
- **Agregador (`RDABajoRango`):** integridad y coherencia como en la v1, más comprobaciones de lo
  nuevo (los vectores miden 1, ningún valor deja una varianza negativa). Reconstruye la forma de
  cada clase de cada cliente (esfera + sus 16 direcciones), las junta como la dispersión del LDA,
  mezcla con la común (α = 0,1) y se queda con las **128 direcciones** principales de la
  corrección de cada clase. Es lo caro, así que se calcula en la última ronda (y cada 5).
- **Modelo (`ModeloRDABajoRango`):** LDA − ½ · Σ valor · (proyección de x − proyección del centro)²
  en las 128 direcciones propias de cada clase.

## Ajustes (semillas 0–2)

| | Accuracy |
|---|---|
| r = 8 / **16** / 32 (cliente) | 81,9 / **81,9** / 81,6 % (con 48 direcciones en el modelo) |
| 48 / 96 / **128** / 160 / 384 direcciones en el modelo | 81,9 / 82,2 / **82,4** / 82,4 / 82,4 % |
| α = 0,07 / **0,1** / 0,12 / 0,15 | 81,9 / **82,4** / 82,4 / 81,7 % |

## Resultados

| Escenario | v3 | **v4** |
|---|---|---|
| Sin atacantes, semillas 3–8 (no usadas para ajustar) | 81,2 % | **82,1 %** |
| 20 % atacantes, `mezcla` | 81,0 % | **82,1 %** |
| 20 % atacantes, ruido / permutar | 80,6 % | **81,8 %** |
| 20 % atacantes, también sobre enteros | 80,8 % | **82,0 %** |
| 30 % atacantes | 79,3 % | **80,8 %** |
| «Todo a la clase 0» | 80,5 % | **81,9 %** |

Ningún cliente honrado marcado en ninguna prueba. ~20 s por semilla en el contenedor de pruebas.
