"""Codec del reto: comprime los estadísticos de ClienteEstadisticos para que quepan en 160 KiB."""

import torch

from fl.codecs.base import Codec
from fl.models import StateDict


def comprimir_simetrica(matriz):
    '''Triangulo superior de una matriz simetrica en int8, con una escala por fila'''

    d = matriz.shape[0]

    # Como es simetrica, solo mandamos el triangulo superior
    filas, columnas = torch.triu_indices(d, d)
    valores = matriz[filas, columnas]

    # Una escala por fila (como en int8, pero por filas para no perder precision)
    maximo = torch.triu(matriz).abs().amax(dim=1)
    escala = (maximo / 127).clamp(min=1e-12)

    # Cuantizamos
    estirado = valores / escala[filas]
    recortado = torch.clamp(torch.round(estirado), min=-127, max=127)

    return recortado.to(torch.int8), escala.float()


def descomprimir_simetrica(v_int8, escala):
    '''Lo contrario: deshace la cuantizacion y reconstruye la matriz entera'''

    d = escala.shape[0]
    filas, columnas = torch.triu_indices(d, d)

    # Deshacemos la cuantizacion
    valores = v_int8.float().flatten() * escala[filas]

    # Reconstruimos la matriz entera (simetrica)
    matriz = torch.zeros(d, d)
    matriz[filas, columnas] = valores
    matriz[columnas, filas] = valores

    return matriz


class estadisticosCodeDecode(Codec):
    '''La dispersion va en int8 (solo el triangulo superior, es simetrica) y las sumas en float16'''

    def encode(self, state:StateDict, reference:StateDict):
        '''El cliente comprime para enviar'''

        comprimido = {
            "cuentas": state["cuentas"].float(),
            "control": state["control"].float(),
        }

        # Lo del LDA (con el modelo RDA solo viene la primera vez)
        if "dispersion" in state:
            dispersion = state["dispersion"]
            v_int8, escala = comprimir_simetrica(dispersion)

            comprimido["sumas"] = state["sumas"].half()
            comprimido["dispersion"] = v_int8
            comprimido["dispersion.escala"] = escala
            # La diagonal tambien en float32: el servidor la compara con la del int8 para ver si la han tocado
            comprimido["dispersion.diagonal"] = torch.diagonal(dispersion).float()

        # Las formas de cada clase (RDA): cada una igual que la dispersion
        if "formas" in state:
            partes = [comprimir_simetrica(forma) for forma in state["formas"]]
            comprimido["formas"] = torch.stack([p[0] for p in partes])
            comprimido["formas.escala"] = torch.stack([p[1] for p in partes])
            comprimido["formas.clases"] = state["formas.clases"].float()

        # RDA de bajo rango: vectores en int8 con una escala por vector, valores en float32
        if "vectores" in state:
            vectores = state["vectores"]
            escala = (vectores.abs().amax(dim=2) / 127).clamp(min=1e-12)
            cuantizado = torch.clamp(torch.round(vectores / escala.unsqueeze(2)), min=-127, max=127)
            comprimido["vectores"] = cuantizado.to(torch.int8)
            comprimido["vectores.escala"] = escala.float()
            comprimido["valores"] = state["valores"].float()
            comprimido["formas.clases"] = state["formas.clases"].float()

        return comprimido


    def decode(self, comprimido:StateDict, reference:StateDict):
        '''El servidor descomprime lo que comprimio el encode'''

        # Si llega algo mal formado devolvemos un control NaN y el agregador lo descarta
        try:
            estadisticos = {
                "cuentas": comprimido["cuentas"].float(),
                "control": comprimido["control"].float(),
            }

            if "dispersion" in comprimido:
                escala = comprimido["dispersion.escala"].float()
                estadisticos["sumas"] = comprimido["sumas"].float()
                estadisticos["dispersion"] = descomprimir_simetrica(comprimido["dispersion"], escala)
                estadisticos["escala"] = escala
                estadisticos["diagonal"] = comprimido["dispersion.diagonal"].float()

            if "formas" in comprimido:
                escalas = comprimido["formas.escala"].float()
                formas = [descomprimir_simetrica(v, e) for v, e in zip(comprimido["formas"], escalas)]
                estadisticos["formas"] = torch.stack(formas)
                estadisticos["formas.clases"] = comprimido["formas.clases"].float()

            if "vectores" in comprimido:
                escala = comprimido["vectores.escala"].float()
                estadisticos["vectores"] = comprimido["vectores"].float() * escala.unsqueeze(2)
                estadisticos["valores"] = comprimido["valores"].float()
                estadisticos["formas.clases"] = comprimido["formas.clases"].float()

        except (KeyError, RuntimeError, IndexError, ValueError):
            estadisticos = {"control": torch.tensor([float("nan")])}

        return estadisticos
