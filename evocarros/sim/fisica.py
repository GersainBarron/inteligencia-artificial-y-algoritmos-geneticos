"""Física del carro: modelo cinemático de bicicleta (protocolo v0.4).

Paso fijo DT = 1/30 s, determinista. Convenciones: x, y en metros; rumbo en
radianes, 0 = +X, positivo a la izquierda (antihorario); el origen del carro es
el centro de su rectángulo de choque.

Entradas de control en [-1, 1]:
- volante: +1 = todo a la izquierda (30°), -1 = todo a la derecha.
- pedal: > 0 acelera (hasta 6 m/s²), < 0 frena (hasta 12 m/s²).
Sin acelerador (pedal <= 0) se suma una desaceleración de 0.5 m/s².
La curvatura pedida se recorta para no pasar de 1.5 g laterales: el carro
subvira en lugar de derrapar. No hay reversa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DT = 1.0 / 30.0
G = 9.81


@dataclass(frozen=True)
class ParamFisica:
    v_tope: float = 50.0          # m/s
    a_acel: float = 6.0           # m/s²
    a_freno: float = 12.0         # m/s²
    a_resist: float = 0.5         # m/s² sin acelerador
    a_lat: float = 1.5 * G        # m/s²
    ejes: float = 2.6             # m
    giro_max: float = math.radians(30.0)
    largo: float = 4.0            # rectángulo de choque
    ancho: float = 1.8


FISICA = ParamFisica()


@dataclass
class EstadoCarro:
    x: float
    y: float
    rumbo: float
    v: float = 0.0


def paso(e: EstadoCarro, volante: float, pedal: float, v_max: float, f: ParamFisica = FISICA) -> None:
    """Avanza el estado un paso DT (lo modifica en el lugar)."""
    volante = min(1.0, max(-1.0, volante))
    pedal = min(1.0, max(-1.0, pedal))
    v_max = min(v_max, f.v_tope)

    if pedal > 0.0:
        a = pedal * f.a_acel
    else:
        a = pedal * f.a_freno - f.a_resist
    v1 = min(v_max, max(0.0, e.v + a * DT))
    v_med = 0.5 * (e.v + v1)

    kappa = math.tan(volante * f.giro_max) / f.ejes
    if v_med > 0.1:
        k_lim = f.a_lat / (v_med * v_med)
        kappa = max(-k_lim, min(k_lim, kappa))
    dth = v_med * kappa * DT
    th_med = e.rumbo + 0.5 * dth
    e.x += v_med * math.cos(th_med) * DT
    e.y += v_med * math.sin(th_med) * DT
    e.rumbo = (e.rumbo + dth + math.pi) % (2 * math.pi) - math.pi
    e.v = v1


def esquinas(e: EstadoCarro, f: ParamFisica = FISICA) -> np.ndarray:
    """Las 4 esquinas del rectángulo de choque, (4, 2), en orden."""
    c, s = math.cos(e.rumbo), math.sin(e.rumbo)
    hl, hw = f.largo / 2, f.ancho / 2
    loc = ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    return np.array([(e.x + c * a - s * b, e.y + s * a + c * b) for a, b in loc])
