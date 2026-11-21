"""Clientes: un archivo por tipo. Para añadir uno, cread su archivo y elegidlo en ``build_clients``."""

from torch.utils.data import Dataset, Subset

from fl.clients.base import Client, ClientUpdate
from fl.clients.estadisticos import ClienteEstadisticos
from fl.clients.fedprox import FedProxClient
from fl.data import Partition
from fl.utils import Config


__all__ = ["Client", "ClientUpdate", "ClienteEstadisticos", "FedProxClient", "build_clients"]


def build_clients(dataset: Dataset, partition: Partition, cfg: Config, device: str) -> list[Client]:
    """Crea un cliente por trozo del particionado: ``FedProxClient`` si ``cfg.client.fedprox_mu > 0``
    y ``Client`` (FedAvg) si no."""
    c = cfg.client
    train_args = dict(local_epochs=c.local_epochs, batch_size=c.batch_size, lr=c.lr, device=device)
    clients = []
    for cid, indices in enumerate(partition):
        local_data = Subset(dataset, indices)
        if c.get("estadisticos", False):  # reto (con .get: las configuraciones de la práctica no la tienen)
            minimo = cfg.get("server", {}).get("minimo_clase", 5)
            clients.append(ClienteEstadisticos(cid, local_data, minimo=minimo, **train_args))
        elif c.fedprox_mu > 0:
            clients.append(FedProxClient(cid, local_data, mu=c.fedprox_mu, **train_args))
        else:
            clients.append(Client(cid, local_data, **train_args))
    return clients
