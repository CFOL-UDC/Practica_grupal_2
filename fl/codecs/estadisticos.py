"""Codec del reto: comprime los estadísticos de ClienteEstadisticos para que quepan en 160 KiB."""

import torch

from fl.codecs.base import Codec
from fl.models import StateDict


class estadisticosCodeDecode(Codec):
    '''La dispersion va en int8 (solo el triangulo superior, es simetrica) y las sumas en float16'''

    def encode(self, state:StateDict, reference:StateDict):
        '''El cliente comprime para enviar'''

        dispersion = state["dispersion"]
        d = dispersion.shape[0]

        # Como es simetrica, solo mandamos el triangulo superior
        filas, columnas = torch.triu_indices(d, d)
        valores = dispersion[filas, columnas]

        # Una escala por fila (como en int8, pero por filas para no perder precision)
        maximo = torch.triu(dispersion).abs().amax(dim=1)
        escala = (maximo / 127).clamp(min=1e-12)

        # Cuantizamos
        estirado = valores / escala[filas]
        recortado = torch.clamp(torch.round(estirado), min=-127, max=127)
        v_int8 = recortado.to(torch.int8)

        comprimido = {
            "cuentas": state["cuentas"].float(),
            "sumas": state["sumas"].half(),
            "dispersion": v_int8,
            "dispersion.escala": escala.float(),
            # La diagonal tambien en float32: el servidor la compara con la del int8 para ver si la han tocado
            "dispersion.diagonal": torch.diagonal(dispersion).float(),
            "control": state["control"].float(),
        }

        return comprimido


    def decode(self, comprimido:StateDict, reference:StateDict):
        '''El servidor descomprime lo que comprimio el encode'''

        # Si llega algo mal formado devolvemos un control NaN y el agregador lo descarta
        try:
            escala = comprimido["dispersion.escala"].float()
            d = escala.shape[0]
            filas, columnas = torch.triu_indices(d, d)

            # Deshacemos la cuantizacion
            valores = comprimido["dispersion"].float().flatten() * escala[filas]

            # Reconstruimos la matriz entera (simetrica)
            dispersion = torch.zeros(d, d)
            dispersion[filas, columnas] = valores
            dispersion[columnas, filas] = valores

            estadisticos = {
                "cuentas": comprimido["cuentas"].float(),
                "sumas": comprimido["sumas"].float(),
                "dispersion": dispersion,
                "escala": escala,
                "diagonal": comprimido["dispersion.diagonal"].float(),
                "control": comprimido["control"].float(),
            }

        except (KeyError, RuntimeError, IndexError, ValueError):
            estadisticos = {"control": torch.tensor([float("nan")])}

        return estadisticos
