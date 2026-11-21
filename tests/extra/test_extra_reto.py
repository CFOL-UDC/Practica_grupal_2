"""Pruebas de la estrategia del reto: ClienteEstadisticos + codec estadisticos + LDARobusto.

Usan datos sintéticos (nubes gaussianas), así que no necesitan data/reto.
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset

from fl.aggregators.lda_robusto import LDARobusto
from fl.clients.estadisticos import ClienteEstadisticos
from fl.codecs.estadisticos import estadisticosCodeDecode

D, C = 16, 10


def nubes(n_por_clase=60, semilla=0):
    centros = 4 * torch.randn(C, D, generator=torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(semilla)
    y = torch.arange(C).repeat_interleave(n_por_clase)
    x = centros[y] + torch.randn(len(y), D, generator=g)
    return x, y


def clientes(x, y, grupos, etiquetas=None):
    """Un cliente por grupo de clases (los ejemplos de cada clase se reparten entre los clientes
    que la tienen); ``etiquetas`` permite cambiar las de algún cliente."""
    dueno = torch.full((len(y),), -1)
    for c in range(C):
        tienen = [cid for cid, clases in enumerate(grupos) if c in clases]
        indices = torch.nonzero(y == c).flatten()
        for i, idx in enumerate(indices):
            dueno[idx] = tienen[i % len(tienen)]
    out = []
    for cid in range(len(grupos)):
        m = dueno == cid
        yc = y[m] if etiquetas is None or cid not in etiquetas else etiquetas[cid](y[m])
        out.append(ClienteEstadisticos(cid, TensorDataset(x[m], yc), minimo=5))
    return out


def ronda(cls, agregador, ataque=None):
    modelo = nn.Linear(D, C)
    codec = estadisticosCodeDecode()
    referencia = modelo.state_dict()
    updates = []
    for c in cls:
        u = c.local_train(modelo)
        payload = codec.encode(u.state_dict, referencia)
        if ataque and c.cid in ataque:
            payload = ataque[c.cid](payload)
        u.state_dict = codec.decode(payload, referencia)
        updates.append(u)
    agregador.global_state = {k: v.clone() for k, v in referencia.items()}
    return agregador.aggregate(updates)


def accuracy(estado, x, y):
    modelo = nn.Linear(D, C)
    modelo.load_state_dict(estado)
    return (modelo(x).argmax(1) == y).float().mean().item()


GRUPOS = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0], [1, 3, 5], [7, 9, 2], [0, 4, 8]]


def test_cliente_sube_estadisticos_correctos():
    x, y = nubes()
    m = y < 3
    estado = ClienteEstadisticos(0, TensorDataset(x[m], y[m]), minimo=5).local_train(nn.Linear(D, C)).state_dict
    assert estado["cuentas"].tolist() == [60, 60, 60] + [0] * 7
    assert torch.allclose(estado["sumas"], torch.stack([x[y == c].sum(0) for c in range(3)]), atol=1e-3)
    assert torch.allclose(estado["dispersion"], estado["dispersion"].T)


def test_no_sube_sumas_de_clases_con_pocos_ejemplos():
    x, y = nubes()
    m = (y == 0) | ((y == 1) & (torch.arange(len(y)) % 60 == 0))  # clase 1: un solo ejemplo
    estado = ClienteEstadisticos(0, TensorDataset(x[m], y[m]), minimo=5).local_train(nn.Linear(D, C)).state_dict
    assert estado["cuentas"][1] == 1
    assert estado["sumas"].shape[0] == 1  # solo la clase 0


def test_payload_cabe_en_el_presupuesto_con_todas_las_clases():
    estado = {
        "cuentas": torch.full((100,), 10.0),
        "sumas": torch.randn(100, 384),
        "dispersion": torch.eye(384),
        "control": torch.ones(1),
    }
    payload = estadisticosCodeDecode().encode(estado, {})
    assert sum(t.numel() * t.element_size() for t in payload.values()) <= 160 * 1024


def test_codec_reconstruye_la_dispersion():
    a = torch.randn(D, 3 * D)
    dispersion = a @ a.T / (3 * D)
    estado = {"cuentas": torch.ones(C), "sumas": torch.zeros(0, D), "dispersion": dispersion, "control": torch.ones(1)}
    codec = estadisticosCodeDecode()
    dec = codec.decode(codec.encode(estado, {}), {})
    assert torch.allclose(dec["dispersion"], dispersion, atol=dispersion.abs().max().item() / 100)


def test_lda_no_depende_del_reparto():
    x, y = nubes()
    a = ronda(clientes(x, y, GRUPOS), LDARobusto())
    b = ronda(clientes(x, y, [list(range(C))]), LDARobusto())  # todos los datos en un cliente
    xt, yt = nubes(semilla=1)
    assert abs(accuracy(a, xt, yt) - accuracy(b, xt, yt)) < 0.02
    assert accuracy(a, xt, yt) > 0.95


def test_envio_exagerado_o_invertido_se_corrige():
    x, y = nubes()
    limpio = ronda(clientes(x, y, GRUPOS), LDARobusto())
    ataque = {0: lambda p: {k: v * 10 if v.is_floating_point() else v for k, v in p.items()},
              1: lambda p: {k: -v if v.is_floating_point() else v for k, v in p.items()}}
    agregador = LDARobusto()
    atacado = ronda(clientes(x, y, GRUPOS), agregador, ataque)
    assert not agregador.descartados
    assert torch.allclose(limpio["weight"], atacado["weight"], rtol=1e-3, atol=1e-3)


def test_envio_con_ruido_se_descarta():
    x, y = nubes()
    g = torch.Generator().manual_seed(0)
    ruido = {2: lambda p: {k: v + 0.1 * v.abs().max() * torch.randn(v.shape, generator=g) if v.is_floating_point() else v
                           for k, v in p.items()}}
    agregador = LDARobusto()
    ronda(clientes(x, y, GRUPOS), agregador, ruido)
    assert agregador.descartados == {2}


def test_etiquetas_cambiadas_se_detectan():
    x, y = nubes()
    agregador = LDARobusto()
    estado = ronda(clientes(x, y, GRUPOS, {3: lambda yc: (yc + 1) % C}), agregador)
    assert 3 in agregador.sospechosos
    assert not (agregador.sospechosos - {3})
    xt, yt = nubes(semilla=1)
    assert accuracy(estado, xt, yt) > 0.95
