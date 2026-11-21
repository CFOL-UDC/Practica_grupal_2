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
            "sumas": sumas[presentes].float(),
            "dispersion": dispersion.float(),
            "control": torch.ones(1),  # siempre vale 1, el servidor lo usa para ver si lo han tocado
        }

        return ClientUpdate(self.cid, estadisticos, self.n_samples, {})
