| Escenario | Configuración | Semillas | Accuracy final | Oscilación | Salto medio | MB |
|---|---|---|---|---|---|---|
| IID | CNN sin GroupNorm | 3 | 0,8830 ± 0,0018 | 0,0044 ± 0,0006 | 0,0038 ± 0,0008 | 100,2 |
| IID | CNN 3×3 sin GroupNorm | 3 | 0,8803 ± 0,0018 | 0,0036 ± 0,0003 | 0,0030 ± 0,0014 | 60,2 |
| IID | CNN | 3 | 0,9028 ± 0,0033 | 0,0045 ± 0,0015 | 0,0045 ± 0,0005 | 100,4 |
| Dirichlet α=0,1 | CNN sin GroupNorm | 3 | 0,7816 ± 0,0463 | 0,0239 ± 0,0057 | 0,0313 ± 0,0123 | 100,2 |
| Dirichlet α=0,1 | CNN 3×3 sin GroupNorm | 3 | 0,7749 ± 0,0371 | 0,0268 ± 0,0056 | 0,0352 ± 0,0070 | 60,2 |
| Dirichlet α=0,1 | CNN | 3 | 0,8332 ± 0,0330 | 0,0200 ± 0,0050 | 0,0264 ± 0,0092 | 100,4 |
