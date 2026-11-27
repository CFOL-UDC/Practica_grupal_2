"""Agregador del reto, versión 4: RDA con la forma de cada clase en sus propias direcciones."""

import torch

from fl.aggregators.lda_robusto import LDARobusto
from fl.clients import ClientUpdate
from fl.models import StateDict


class RDABajoRango(LDARobusto):
    '''Como RDARobusto, pero cada clase manda las direcciones en las que su forma mas se aleja
    de la comun (de bajo rango), en vez de usar unas direcciones fijas para todas'''

    def __init__(self, contraccion=0.01, umbral=0.5, minimo=5, mezcla=0.1, clientes_para_fijar=30, rondas=None):
        super().__init__(contraccion, umbral, minimo)
        self.rondas = rondas  # la correccion es cara: si sabemos cuantas rondas hay, solo se calcula en la ultima
        self.ronda = 0
        self.correccion = None  # lo ultimo que se calculo: centros, vectores, valores y el ajuste del sesgo
        self.mezcla = mezcla
        self.clientes_para_fijar = clientes_para_fijar
        self.blanqueo = None  # se fija una vez: los clientes miden su forma con el
        self.formas = {}  # cid -> {clase: (vectores, valores)}


    def comprobar_formas(self, update: ClientUpdate):
        '''Devuelve {clase: (vectores, valores)} corregidos o None si el envio esta manipulado'''

        try:
            e = {k: v.double() for k, v in update.state_dict.items()}

            for v in e.values():
                if not torch.isfinite(v).all():
                    return None

            # Control y cuentas, como siempre
            k = e["control"].flatten()[0]
            if k.abs() < 1e-6:
                return None
            cuentas = e["cuentas"] / k
            if (cuentas - cuentas.round()).abs().max() > 1e-3 or int(cuentas.round().sum()) != update.n_samples:
                return None

            if "vectores" not in e:
                return {}

            clases = e["formas.clases"] / k
            if (clases - clases.round()).abs().max() > 1e-3:
                return None
            clases = clases.round().long()
            if clases.min() < 0 or clases.max() >= len(cuentas) or len(set(clases.tolist())) != len(clases):
                return None
            if (cuentas.round()[clases] < self.minimo).any():
                return None

            # Los vectores tienen que medir 1 (salen de una descomposicion en autovectores)
            vectores = e["vectores"] / k
            if (vectores.norm(dim=2) - 1).abs().max() > 0.05:
                return None

            # Una forma no puede tener varianza negativa: 1 + valor >= 0
            valores = e["valores"] / k
            if valores.min() < -1.01:
                return None

        except (KeyError, RuntimeError, IndexError, ValueError):
            return None

        return {c: (v, l) for c, v, l in zip(clases.tolist(), vectores, valores)}


    def aggregate(self, updates: list[ClientUpdate]) -> StateDict:
        self.ronda += 1

        # 1. Lo del LDA, igual que en la version 1
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
            nuevo = super().aggregate([])

        fiables = [c for c in self.estadisticos if c not in self.sospechosos]
        if len(fiables) == 0 or "blanqueo" not in nuevo or nuevo is self.global_state:
            return nuevo

        _, _, precision = self.lda(fiables)
        covarianza = torch.linalg.inv(precision)

        # 3. Fijamos el blanqueo cuando ya hay suficientes clientes
        if self.blanqueo is None and len(fiables) >= self.clientes_para_fijar:
            valores, vectores = torch.linalg.eigh(covarianza)
            self.blanqueo = vectores @ torch.diag(valores ** -0.5) @ vectores.T
        if self.blanqueo is None:
            return nuevo

        nuevo["blanqueo"] = self.blanqueo.float()
        # La correccion es cara: se calcula en la ultima ronda (y cada 5, por si acaso) y entre medias
        # se reutiliza la ultima calculada
        if self.rondas is None or self.ronda >= self.rondas or self.ronda % 5 == 0:
            self.correccion = self.corregir(nuevo, fiables, covarianza)
        if self.correccion is None:
            return nuevo
        for clave in ["centros", "vectores", "valores"]:
            nuevo[clave] = self.correccion[clave]
        nuevo["bias"] = nuevo["bias"] + self.correccion["ajuste"]
        return nuevo


    def corregir(self, nuevo, fiables, covarianza):
        '''Junta la forma de cada clase, la mezcla con la comun y devuelve lo que hay que cambiar en el modelo'''

        w = self.blanqueo
        d = w.shape[0]
        identidad = torch.eye(d, dtype=torch.float64)
        cuentas = sum(self.cuentas(c) for c in fiables)
        sumas = sum(self.sumas(c) for c in fiables)
        medias = sumas / cuentas.clamp(min=1).unsqueeze(1)

        # La forma comun en el espacio blanqueado (casi la identidad)
        comun = w @ covarianza @ w
        comun_inv = torch.linalg.inv(comun)
        logdet_comun = torch.linalg.slogdet(comun)[1]

        # Juntamos la forma de cada clase de todos los clientes (como la dispersion del LDA)
        formas = torch.zeros(len(cuentas), d, d, dtype=torch.float64)
        n = torch.zeros(len(cuentas), dtype=torch.float64)
        for cid in fiables:
            n_cliente = self.cuentas(cid)
            for clase, (vectores, valores) in self.formas.get(cid, {}).items():
                n_c = float(n_cliente[clase])
                if n_c == 0:
                    continue
                dif = ((self.estadisticos[cid]["sumas"][clase] / n_c - medias[clase]) @ w).unsqueeze(1)
                forma = identidad + vectores.T @ torch.diag(valores) @ vectores + dif @ dif.T
                formas[clase] += n_c * forma
                n[clase] += n_c

        con_forma = torch.nonzero(n > 0).flatten()
        if len(con_forma) == 0:
            return None

        # Mezclamos (como en la v2) y calculamos la correccion de todas las clases a la vez
        mezcladas = (1 - self.mezcla) * comun + self.mezcla * formas[con_forma] / n[con_forma].view(-1, 1, 1)
        correccion = torch.linalg.inv(mezcladas) - comun_inv
        ajuste = -0.5 * (torch.linalg.slogdet(mezcladas)[1] - logdet_comun)

        # La pasamos al espacio original y nos quedamos con sus direcciones principales
        correccion = w @ correccion @ w
        valores, vectores = torch.linalg.eigh(correccion)
        rango = nuevo["vectores"].shape[1]
        mejores = torch.argsort(valores.abs(), dim=1, descending=True)[:, :rango]
        valores = torch.gather(valores, 1, mejores)
        vectores = torch.gather(vectores, 2, mejores.unsqueeze(1).expand(-1, d, -1)).transpose(1, 2)

        if not (torch.isfinite(valores).all() and torch.isfinite(vectores).all() and torch.isfinite(ajuste).all()):
            return None

        correccion = {
            "centros": medias.float(),
            "vectores": torch.zeros_like(nuevo["vectores"]),
            "valores": torch.zeros_like(nuevo["valores"]),
            "ajuste": torch.zeros_like(nuevo["bias"]),  # se suma al sesgo del LDA
        }
        correccion["vectores"][con_forma] = vectores.float()
        correccion["valores"][con_forma] = valores.float()
        correccion["ajuste"][con_forma] = ajuste.float()

        return correccion
