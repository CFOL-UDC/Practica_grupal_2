"""Agregadores: un archivo por método. Para añadir uno, registradlo en ``build_aggregator``."""

from fl.aggregators.base import Aggregator
from fl.aggregators.fedavg import FedAvg
from fl.aggregators.lda_robusto import LDARobusto
from fl.utils import Config


__all__ = ["Aggregator", "FedAvg", "LDARobusto", "build_aggregator"]


def build_aggregator(cfg: Config) -> Aggregator:
    """Crea el agregador indicado en ``cfg.aggregator`` (sección ``server`` de la configuración)."""
    if cfg.aggregator == "fedavg":
        return FedAvg()
    if cfg.aggregator == "lda_robusto":
        return LDARobusto(cfg.get("contraccion", 0.01), cfg.get("umbral_coherencia", 0.5), cfg.get("minimo_clase", 5))
    raise ValueError(f"Agregador desconocido: {cfg.aggregator!r}")
