"""Piloto de referencia escrito a mano (no evolucionado).

Sigue un punto de la línea central adelante del carro y frena según el radio más
cerrado de los próximos ~90 m. No usa los rayos: conoce la pista completa, así que
NO es una condición experimental. Sirve para comprobar que la física y la pista
permiten completar la vuelta, como piloto de demostración del visor y como cota
de referencia razonable (no óptima) del tiempo de vuelta.
"""
from __future__ import annotations

import math

import numpy as np

from .fisica import FISICA
from .pista import Pista

CODIGO_DEMO = '''def manejar(estado, pista):
    s = pista.progreso(estado)
    objetivo = pista.punto_central(s + 6 + 0.35 * estado.v)
    volante = angulo_hacia(objetivo, estado) / GIRO_MAX
    radio = pista.radio_minimo(s, s + 90)
    v_segura = min(45, 0.8 * (A_LAT * radio) ** 0.5)
    if estado.v < v_segura - 1:
        pedal = 1.0
    elif estado.v > v_segura + 2:
        pedal = -1.0
    else:
        pedal = 0.2
    return volante, pedal
'''


class PilotoCentro:
    def __init__(self, pista: Pista):
        self.p = pista
        self.estado = None          # episodio.correr lo llena con el estado del carro
        self.i = 0

    def __call__(self, entradas):
        e, p = self.estado, self.p
        s, self.i = p.progreso_local((e.x, e.y), self.i)
        n = len(p.centro)
        j = int(np.searchsorted(p.s, (s + 6 + e.v * 0.35) % p.longitud) - 1) % n
        obj = p.centro[j]
        ang = math.atan2(obj[1] - e.y, obj[0] - e.x) - e.rumbo
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        volante = max(-1.0, min(1.0, ang / FISICA.giro_max))
        rmin = 1e9
        for d in range(0, 90, 6):
            k = int(np.searchsorted(p.s, (s + d) % p.longitud) - 1) % n
            a, b, c = p.centro[k - 1], p.centro[k], p.centro[(k + 1) % n]
            cr = abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])
            if cr > 1e-9:
                rmin = min(rmin, np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(c - a) / (2 * cr))
        v_obj = min(45.0, 0.8 * math.sqrt(FISICA.a_lat * rmin))
        pedal = 1.0 if e.v < v_obj - 1 else (-1.0 if e.v > v_obj + 2 else 0.2)
        return np.array([volante, pedal])
