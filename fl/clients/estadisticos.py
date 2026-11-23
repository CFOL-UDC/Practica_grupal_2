"""Cliente del reto: en vez de entrenar, sube estadísticos de sus datos para calcular un LDA."""

import torch
from torch import nn

from fl.clients.base import Client, ClientUpdate


class ClienteEstadisticos(Client):
    '''No entrena con SGD: calcula las cuentas, sumas y dispersión por clase de sus datos'''

    def __init__(self, cid, dataset, *, minimo=5, **kwargs):
        super().__init__(cid, dataset, **kwargs)

        # Solo mandamos la suma de una clase si tiene al menos este nº de ejemplos
        # (asi nunca se sube algo calculado con un ejemplo suelto)
        self.minimo = minimo

        # Para RDA: el cliente se acuerda de lo que ya ha mandado (vive toda la simulacion)
        self.enviado = False
        self.formas_enviadas = set()
        self.minimo_forma = 10  # la forma de una clase necesita mas ejemplos que su media
        self.presupuesto = 160 * 1024  # bytes que se pueden subir por ronda (lo fija el reto)

    def local_train(self, global_model: nn.Module) -> ClientUpdate:
        '''Devuelve los estadisticos del cliente en vez de pesos entrenados'''

        # El nº de clases lo sacamos del sesgo de la ultima capa
        n_clases = list(global_model.parameters())[-1].shape[0]

        # Cogemos todos los datos del cliente de una vez (son pocos)
        x = torch.stack([self.dataset[i][0] for i in range(len(self.dataset))]).double()
        y = torch.tensor([int(self.dataset[i][1]) for i in range(len(self.dataset))])

        # Cuantos ejemplos hay de cada clase y la suma de sus embeddings
        cuentas = torch.bincount(y, minlength=n_clases).double()
        sumas = torch.zeros(n_clases, x.shape[1], dtype=torch.float64)
        sumas.index_add_(0, y, x)

        # Dispersion dentro de cada clase: restamos a cada ejemplo la media de su clase
        medias = sumas / cuentas.clamp(min=1).unsqueeze(1)
        centrados = x - medias[y]
        dispersion = centrados.T @ centrados / len(x)

        # Solo las sumas de las clases con suficientes ejemplos
        presentes = cuentas >= self.minimo

        estadisticos = {
            "cuentas": cuentas.float(),
            "control": torch.ones(1),  # siempre vale 1, el servidor lo usa para ver si lo han tocado
        }
        ocupado = 4 * n_clases + 4

        # Con el modelo RDA, lo del LDA solo se manda una vez (los datos no cambian) y se deja
        # sitio para las formas. Con el modelo lineal se manda siempre, como en la version 1
        proyeccion = global_model.state_dict().get("proyeccion")
        if proyeccion is None or not self.enviado:
            estadisticos["sumas"] = sumas[presentes].float()
            estadisticos["dispersion"] = dispersion.float()
            d = x.shape[1]
            ocupado += 2 * int(presentes.sum()) * d + d * (d + 1) // 2 + 8 * d
            self.enviado = True

        # RDA: si el modelo global ya trae las direcciones, mandamos la forma de nuestras clases
        if proyeccion is not None and proyeccion.abs().sum() > 0:
            formas, clases = self.formas(x, y, cuentas, proyeccion.double(), ocupado)
            if len(clases) > 0:
                estadisticos["formas"] = formas.float()
                estadisticos["formas.clases"] = clases.float()

        return ClientUpdate(self.cid, estadisticos, self.n_samples, {})

    def formas(self, x, y, cuentas, proyeccion, ocupado):
        '''Dispersion de cada clase en las direcciones del servidor, las que quepan en el presupuesto'''

        k = proyeccion.shape[0]
        por_clase = k * (k + 1) // 2 + 4 * k + 4  # bytes de una forma ya comprimida (ver el codec)
        caben = max(0, (self.presupuesto - ocupado - 1024) // por_clase)

        # Primero las clases con mas ejemplos, que dan la forma mas fiable
        pendientes = [c for c in torch.argsort(cuentas, descending=True).tolist()
                      if cuentas[c] >= self.minimo_forma and c not in self.formas_enviadas]
        clases = torch.tensor(pendientes[:caben], dtype=torch.long)

        u = x @ proyeccion.T
        formas = []
        for c in clases.tolist():
            uc = u[y == c]
            centrados = uc - uc.mean(0)
            formas.append(centrados.T @ centrados / len(uc))
            self.formas_enviadas.add(c)

        if formas:
            return torch.stack(formas), clases
        return torch.zeros(0, k, k), clases
