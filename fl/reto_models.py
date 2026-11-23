"""Modelos del reto: reciben un embedding de 384 números y devuelven 100 logits.

Aquí se añaden los modelos, que se eligen con ``model.name`` en ``configs/reto.yaml`` (igual que
``build_model`` en ``fl/models.py``). El evaluador reconstruye el modelo final con esta función.
"""

from __future__ import annotations

import torch
from torch import nn

from fl.utils import Config

IN_FEATURES, N_CLASSES = 384, 100


def build_reto_model(cfg: Config) -> nn.Module:
    """Crea el modelo indicado en la sección ``model`` de la configuración."""
    if cfg.name == "linear":
        return nn.Linear(IN_FEATURES, N_CLASSES)
    if cfg.name == "mlp":
        # ~97 000 parámetros: en float32 no cabe en el presupuesto de subida (hace falta comprimir)
        return nn.Sequential(nn.Linear(IN_FEATURES, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, N_CLASSES))
    if cfg.name == "rda":
        return ModeloRDA(IN_FEATURES, N_CLASSES, cfg.get("direcciones", 99))
    raise ValueError(f"Modelo desconocido: {cfg.name!r}")


class ModeloRDA(nn.Module):
    '''LDA (parte lineal) + una correccion cuadratica por clase en unas pocas direcciones'''

    def __init__(self, dim, n_clases, direcciones):
        super().__init__()

        # La parte lineal: es el LDA de siempre (mismos nombres que nn.Linear)
        self.weight = nn.Parameter(torch.zeros(n_clases, dim))
        self.bias = nn.Parameter(torch.zeros(n_clases))

        # La correccion: la calcula el servidor, no se entrena (por eso son buffers)
        self.register_buffer("proyeccion", torch.zeros(direcciones, dim))  # las direcciones
        self.register_buffer("centros", torch.zeros(n_clases, direcciones))  # media de cada clase en ellas
        self.register_buffer("curvatura", torch.zeros(n_clases, direcciones, direcciones))  # forma de cada clase

    def forward(self, x):
        lineal = x @ self.weight.T + self.bias

        # Distancia de x al centro de cada clase en las direcciones, con la forma de esa clase
        u = x @ self.proyeccion.T
        d = u.unsqueeze(1) - self.centros.unsqueeze(0)
        cuadratica = torch.einsum("ncj,cjk,nck->nc", d, self.curvatura, d)

        return lineal - 0.5 * cuadratica
