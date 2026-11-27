"""Agregadores: un archivo por método. Para añadir uno, registradlo en ``build_aggregator``."""

from fl.aggregators.base import Aggregator
from fl.aggregators.fedavg import FedAvg
from fl.aggregators.lda_robusto import LDARobusto
from fl.aggregators.rda_bajo_rango import RDABajoRango
from fl.aggregators.rda_robusto import RDARobusto
from fl.utils import Config


__all__ = ["Aggregator", "FedAvg", "LDARobusto", "RDABajoRango", "RDARobusto", "build_aggregator"]


def build_aggregator(cfg: Config) -> Aggregator:
    """Crea el agregador indicado en ``cfg.aggregator`` (sección ``server`` de la configuración)."""
    if cfg.aggregator == "fedavg":
        return FedAvg()
    if cfg.aggregator == "lda_robusto":
        return LDARobusto(cfg.get("contraccion", 0.01), cfg.get("umbral_coherencia", 0.5), cfg.get("minimo_clase", 5))
    if cfg.aggregator == "rda_robusto":
        return RDARobusto(cfg.get("contraccion", 0.01), cfg.get("umbral_coherencia", 0.5), cfg.get("minimo_clase", 5),
                          cfg.get("mezcla", 0.25), cfg.get("fijar_con", 30))
    if cfg.aggregator == "rda_bajo_rango":
        return RDABajoRango(cfg.get("contraccion", 0.01), cfg.get("umbral_coherencia", 0.5), cfg.get("minimo_clase", 5),
                            cfg.get("mezcla", 0.1), cfg.get("fijar_con", 30), cfg.rounds)
    raise ValueError(f"Agregador desconocido: {cfg.aggregator!r}")
