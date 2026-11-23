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
