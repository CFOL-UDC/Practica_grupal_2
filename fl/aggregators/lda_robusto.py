"""Agregador del reto: calcula un LDA con los estadísticos de los clientes y descarta a los atacantes."""

import torch

from fl.aggregators.base import Aggregator
from fl.clients import ClientUpdate
from fl.models import StateDict


class LDARobusto(Aggregator):
    '''Guarda los estadisticos de cada cliente (por cid) y cada ronda recalcula el LDA con todos'''

    def __init__(self, contraccion=0.01, umbral=0.5, minimo=5):
        self.contraccion = contraccion  # regularizacion de la covarianza
        self.umbral = umbral  # fraccion minima de clases coherentes para fiarnos de un cliente
        self.minimo = minimo  # el mismo que en los clientes
        self.max_z = 3.0  # distancia maxima (normalizada) a la media de su clase; sin ataques no pasa de 2.5

        self.global_state = {}
        self.estadisticos = {}  # cid -> estadisticos ya comprobados
        self.validas = {}  # cid -> clases de ese cliente que pasan la coherencia
        self.z = {}
        self.descartados = set()  # envios manipulados (para siempre)
        self.sospechosos = set()  # etiquetas incoherentes (se recalcula cada ronda)


    def comprobar(self, update: ClientUpdate):
        '''Devuelve los estadisticos corregidos o None si el envio esta manipulado'''

        try:
            e = {k: v.double() for k, v in update.state_dict.items()}

            for v in e.values():
                if not torch.isfinite(v).all():
                    return None

            # control era 1: si vale k es que han multiplicado todo por k (k = -1 si esta invertido)
            k = e["control"].flatten()[0]
            if k.abs() < 1e-6:
                return None

            # Las cuentas tienen que ser enteros positivos que sumen n_samples
            cuentas = e["cuentas"] / k
            if (cuentas - cuentas.round()).abs().max() > 1e-3 or cuentas.min() < -1e-3:
                return None
            cuentas = cuentas.round()
            if int(cuentas.sum()) != update.n_samples:
                return None

            presentes = cuentas >= self.minimo
            if e["sumas"].shape[0] != int(presentes.sum()):
                return None

            # La diagonal del int8 tiene que coincidir con la que viaja en float32
            dispersion = e["dispersion"] / k
            diagonal = e["diagonal"] / k
            error_cuantizacion = (e["escala"] / k).abs() * 0.51
            if (torch.diagonal(dispersion) - diagonal).abs().gt(error_cuantizacion + 1e-6).any():
                return None

            # Una dispersion no puede tener autovalores negativos
            autovalores = torch.linalg.eigvalsh(dispersion)
            if autovalores.min() < -0.01 * autovalores.max().clamp(min=1e-12):
                return None

            sumas = torch.zeros(len(cuentas), e["sumas"].shape[1], dtype=torch.float64)
            sumas[presentes] = e["sumas"] / k

        except (KeyError, RuntimeError, IndexError, ValueError):
            return None

        return {"cuentas": cuentas * presentes, "sumas": sumas, "dispersion": dispersion, "n": update.n_samples}


    # Lo que aporta cada cliente quitando las clases que no han pasado la coherencia
    def cuentas(self, cid):
        return self.estadisticos[cid]["cuentas"] * self.validas[cid]

    def sumas(self, cid):
        return self.estadisticos[cid]["sumas"] * self.validas[cid].unsqueeze(1)


    def coherencia(self, fiables, precision):
        '''Para cada cliente, fraccion de sus ejemplos cuya clase cuadra con la del resto de clientes'''

        sumas = sum(self.sumas(c) for c in fiables)
        cuentas = sum(self.cuentas(c) for c in fiables)

        # Con blanqueo la distancia normal es la de Mahalanobis
        blanqueo = torch.linalg.cholesky(precision)
        puntos = {}

        for cid, e in self.estadisticos.items():

            # Medias de referencia: las del resto de clientes (le quitamos lo suyo)
            if cid in fiables:
                ref_sumas = sumas - self.sumas(cid)
                ref_cuentas = cuentas - self.cuentas(cid)
            else:
                ref_sumas = sumas
                ref_cuentas = cuentas

            hay_ref = ref_cuentas >= self.minimo
            clases = torch.nonzero((e["cuentas"] > 0) & hay_ref).flatten()

            # Si no hay con que comparar, nos fiamos
            if len(clases) == 0:
                puntos[cid] = 1.0
                continue

            ref = ref_sumas[hay_ref] / ref_cuentas[hay_ref].unsqueeze(1)
            suyas = e["sumas"][clases] / e["cuentas"][clases].unsqueeze(1)
            dist = torch.cdist(suyas @ blanqueo, ref @ blanqueo) ** 2

            # 1. Cada media suya tiene que estar mas cerca de su clase que de cualquier otra
            indice_ref = torch.nonzero(hay_ref).flatten()
            mas_cercana = indice_ref[dist.argmin(1)] == clases

            # 2. Y a la distancia normal por azar: unas d * (1/n + 1/N)
            propia = dist[torch.arange(len(clases)), torch.searchsorted(indice_ref, clases)]
            esperada = precision.shape[0] * (1 / e["cuentas"][clases] + 1 / ref_cuentas[clases])
            self.z[cid] = propia / esperada

            aciertos = mas_cercana & (self.z[cid] < self.max_z)

            # Las clases que fallan no se usan aunque el cliente sea fiable
            self.validas[cid] = torch.ones(len(e["cuentas"]), dtype=torch.bool)
            self.validas[cid][clases] = aciertos

            pesos = e["cuentas"][clases]
            puntos[cid] = float((aciertos.double() * pesos).sum() / pesos.sum())

        return puntos


    def lda(self, fiables):
        '''Calcula el LDA con los clientes fiables: pesos, sesgo y la inversa de la covarianza'''

        sumas = sum(self.sumas(c) for c in fiables)
        cuentas = sum(self.cuentas(c) for c in fiables)
        medias = sumas / cuentas.clamp(min=1).unsqueeze(1)

        # Covarianza comun = dispersion dentro de cada cliente + lo que se separan
        # las medias de cada cliente de la media global de la clase
        n = sum(self.estadisticos[c]["n"] for c in fiables)
        dispersion = sum(self.estadisticos[c]["n"] * self.estadisticos[c]["dispersion"] for c in fiables)
        for c in fiables:
            dif = self.sumas(c) / self.cuentas(c).clamp(min=1).unsqueeze(1) - medias
            dispersion = dispersion + (dif * self.cuentas(c).unsqueeze(1)).T @ dif

        covarianza = dispersion / n

        # Regularizamos (shrinkage)
        d = covarianza.shape[0]
        covarianza = covarianza + self.contraccion * torch.trace(covarianza) / d * torch.eye(d, dtype=torch.float64)
        precision = torch.linalg.inv(covarianza)

        pesos = medias @ precision
        sesgo = -0.5 * (pesos * medias).sum(1)

        # Las clases que no ha mandado nadie no se predicen nunca
        sesgo[cuentas == 0] = -1e9

        return pesos, sesgo, precision


    def aggregate(self, updates: list[ClientUpdate]) -> StateDict:

        # 1. Comprobamos lo que llega y lo guardamos
        for u in updates:
            if u.cid in self.descartados:
                continue

            e = self.comprobar(u)
            if e is None:
                self.descartados.add(u.cid)
                self.estadisticos.pop(u.cid, None)
            else:
                self.estadisticos[u.cid] = e
                self.validas[u.cid] = torch.ones(len(e["cuentas"]), dtype=torch.bool)

        # La traza de la dispersion es parecida en todos los clientes: si se aleja mucho
        # de la mediana es que la han tocado
        trazas = {c: float(torch.trace(e["dispersion"])) for c, e in self.estadisticos.items()}
        if len(trazas) >= 5:
            mediana = torch.tensor(list(trazas.values())).median()
            for c, t in trazas.items():
                if t < mediana / 3 or t > mediana * 3:
                    self.descartados.add(c)
                    self.estadisticos.pop(c)

        fiables = list(self.estadisticos)
        if not fiables:
            return self.global_state

        # 2. Quitamos a los que tienen etiquetas incoherentes. Repetimos porque al
        # principio las medias de referencia tambien incluyen a los atacantes
        for _ in range(3):
            _, _, precision = self.lda(fiables)
            puntos = self.coherencia(fiables, precision)
            nuevos = [c for c in self.estadisticos if puntos[c] >= self.umbral]
            if not nuevos or set(nuevos) == set(fiables):
                break
            fiables = nuevos

        self.sospechosos = set(self.estadisticos) - set(fiables)

        # 3. LDA con los fiables y lo metemos en el modelo lineal
        pesos, sesgo, _ = self.lda(fiables)
        if not (torch.isfinite(pesos).all() and torch.isfinite(sesgo).all()):
            return self.global_state

        nuevo = {k: v.clone() for k, v in self.global_state.items()}
        nuevo["weight"] = pesos.to(nuevo["weight"].dtype)
        nuevo["bias"] = sesgo.to(nuevo["bias"].dtype)

        return nuevo


    def resumen(self):
        return f"| descartados {sorted(self.descartados)} | sospechosos {sorted(self.sospechosos)}"
