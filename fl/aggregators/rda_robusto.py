"""Agregador del reto, versión 2: el LDA de LDARobusto más la forma propia de cada clase (RDA)."""

import torch

from fl.aggregators.lda_robusto import LDARobusto
from fl.clients import ClientUpdate
from fl.models import StateDict


class RDARobusto(LDARobusto):
    '''Igual que LDARobusto (integridad, coherencia y LDA), pero ademas mezcla la forma comun
    con la forma de cada clase en unas pocas direcciones (analisis discriminante regularizado)'''

    def __init__(self, contraccion=0.01, umbral=0.5, minimo=5, mezcla=0.3, clientes_para_fijar=30):
        super().__init__(contraccion, umbral, minimo)
        self.mezcla = mezcla  # cuanto de la forma propia de cada clase (0 = LDA)
        self.clientes_para_fijar = clientes_para_fijar  # con cuantos clientes fijamos las direcciones
        self.proyeccion = None  # las direcciones: se fijan una vez y ya no cambian
        self.formas = {}  # cid -> {clase: forma de esa clase en las direcciones}


    def comprobar_formas(self, update: ClientUpdate):
        '''Devuelve {clase: forma} corregidas o None si el envio esta manipulado'''

        try:
            e = {k: v.double() for k, v in update.state_dict.items()}

            for v in e.values():
                if not torch.isfinite(v).all():
                    return None

            # Mismas comprobaciones que en LDARobusto: control y cuentas enteras que suman n_samples
            k = e["control"].flatten()[0]
            if k.abs() < 1e-6:
                return None
            cuentas = e["cuentas"] / k
            if (cuentas - cuentas.round()).abs().max() > 1e-3 or int(cuentas.round().sum()) != update.n_samples:
                return None

            if "formas" not in e:
                return {}

            # Las clases tienen que ser enteros, distintos y con ejemplos de verdad en el cliente
            clases = e["formas.clases"] / k
            if (clases - clases.round()).abs().max() > 1e-3:
                return None
            clases = clases.round().long()
            if clases.min() < 0 or clases.max() >= len(cuentas) or len(set(clases.tolist())) != len(clases):
                return None
            if (cuentas.round()[clases] < self.minimo).any():
                return None

            # Cada forma es una dispersion: no puede tener autovalores negativos
            formas = e["formas"] / k
            autovalores = torch.linalg.eigvalsh(formas)
            if (autovalores.min(dim=1).values < -0.01 * autovalores.max(dim=1).values.clamp(min=1e-12)).any():
                return None

        except (KeyError, RuntimeError, IndexError, ValueError):
            return None

        return {c: f for c, f in zip(clases.tolist(), formas)}


    def fijar_direcciones(self, fiables, covarianza):
        '''Las direcciones en las que mas se separan las medias de las clases (las del propio LDA)'''

        cuentas = sum(self.cuentas(c) for c in fiables)
        sumas = sum(self.sumas(c) for c in fiables)
        vistas = cuentas > 0
        medias = sumas[vistas] / cuentas[vistas].unsqueeze(1)

        # Blanqueamos (la forma comun pasa a ser una esfera) y centramos las medias
        valores, vectores = torch.linalg.eigh(covarianza)
        blanqueo = vectores @ torch.diag(valores ** -0.5) @ vectores.T
        medias = medias @ blanqueo
        medias = medias - medias.mean(0)

        # Las direcciones principales de esas medias
        _, _, vt = torch.linalg.svd(medias, full_matrices=False)
        k = min(self.global_state["proyeccion"].shape[0], int(vistas.sum()) - 1)
        self.proyeccion = vt[:k] @ blanqueo


    def aggregate(self, updates: list[ClientUpdate]) -> StateDict:

        # 1. Lo del LDA, exactamente igual que en la version 1
        nuevo = super().aggregate([u for u in updates if "dispersion" in u.state_dict])

        # 2. Comprobamos y guardamos las formas
        descartado = False
        for u in updates:
            if u.cid in self.descartados or u.cid not in self.estadisticos:
                continue
            formas = self.comprobar_formas(u)
            if formas is None:
                self.descartados.add(u.cid)
                self.estadisticos.pop(u.cid)
                self.formas.pop(u.cid, None)
                descartado = True
            else:
                self.formas.setdefault(u.cid, {}).update(formas)

        if descartado:
            nuevo = super().aggregate([])  # recalculamos el LDA sin los que acabamos de quitar

        fiables = [c for c in self.estadisticos if c not in self.sospechosos]
        if len(fiables) == 0 or "proyeccion" not in nuevo or nuevo is self.global_state:
            return nuevo

        _, _, precision = self.lda(fiables)
        covarianza = torch.linalg.inv(precision)

        # 3. Fijamos las direcciones cuando ya hay suficientes clientes
        if self.proyeccion is None and len(fiables) >= self.clientes_para_fijar:
            self.fijar_direcciones(fiables, covarianza)
        if self.proyeccion is None:
            return nuevo

        # 4. La correccion de cada clase
        return self.corregir(nuevo, fiables, covarianza)


    def corregir(self, nuevo, fiables, covarianza):
        '''Mezcla la forma comun con la forma de cada clase y lo mete en el modelo'''

        p = self.proyeccion
        k = p.shape[0]
        cuentas = sum(self.cuentas(c) for c in fiables)
        sumas = sum(self.sumas(c) for c in fiables)
        centros = (sumas / cuentas.clamp(min=1).unsqueeze(1)) @ p.T

        # La forma comun vista en las direcciones
        comun = p @ covarianza @ p.T
        comun_inv = torch.linalg.inv(comun)
        _, logdet_comun = torch.linalg.slogdet(comun)

        curvatura = torch.zeros(len(cuentas), k, k, dtype=torch.float64)
        ajuste = torch.zeros(len(cuentas), dtype=torch.float64)

        for clase in range(len(cuentas)):
            # Juntamos la forma de esta clase de todos los clientes (como la dispersion del LDA)
            forma, n = torch.zeros(k, k, dtype=torch.float64), 0.0
            for cid in fiables:
                n_c = float(self.cuentas(cid)[clase])
                if n_c == 0 or clase not in self.formas.get(cid, {}):
                    continue
                media = (self.estadisticos[cid]["sumas"][clase] / n_c) @ p.T
                dif = (media - centros[clase]).unsqueeze(1)
                forma += n_c * (self.formas[cid][clase][:k, :k] + dif @ dif.T)
                n += n_c
            if n == 0:
                continue  # nadie ha mandado su forma: se queda con la comun (LDA)

            mezclada = (1 - self.mezcla) * comun + self.mezcla * forma / n
            curvatura[clase] = torch.linalg.inv(mezclada) - comun_inv
            ajuste[clase] = -0.5 * (torch.linalg.slogdet(mezclada)[1] - logdet_comun)

        if not (torch.isfinite(curvatura).all() and torch.isfinite(ajuste).all()):
            return nuevo

        # Lo metemos en el modelo (las direcciones que sobran se quedan a 0)
        nuevo["proyeccion"] = torch.zeros_like(nuevo["proyeccion"])
        nuevo["proyeccion"][:k] = p.float()
        nuevo["centros"] = torch.zeros_like(nuevo["centros"])
        nuevo["centros"][:, :k] = centros.float()
        nuevo["curvatura"] = torch.zeros_like(nuevo["curvatura"])
        nuevo["curvatura"][:, :k, :k] = curvatura.float()
        nuevo["bias"] = nuevo["bias"] + ajuste.float()

        return nuevo
