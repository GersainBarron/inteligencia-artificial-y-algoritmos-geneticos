"""Un episodio: un carro, una pista, hasta que choca, se estanca, va en sentido
contrario, completa la vuelta o se acaba el tiempo (protocolo v0.4).

Aptitud:  F = N_cp + d_avance / d_tramo + b * max(0, (T_max - t_vuelta) / T_max)
- N_cp + fracción se calculan sobre el avance MÁXIMO alcanzado (morir conserva
  el avance logrado).
- El bono b = 10 solo aplica si completa la vuelta.

Fin del episodio (motivo):
- "choque": una esquina del rectángulo toca un borde.
- "estancamiento": en 3 s su avance máximo no creció 10 m.
- "sentido_contrario": su avance actual cae 20 m por debajo del máximo
  (equivale a la "marcha atrás sostenida" del protocolo; el carro no tiene reversa,
  así que solo puede retroceder dando la vuelta).
- "vuelta": completó una vuelta.
- "tiempo": alcanzó el tiempo límite de esta generación.

Entradas de la red (7): 5 rayos normalizados a [0, 1] (1 = nada dentro del
alcance), velocidad / v_tope, y ángulo al centro del siguiente checkpoint / pi.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .fisica import DT, FISICA, EstadoCarro, ParamFisica, esquinas, paso
from .genoma import Cuerpo
from .pista import Pista

BONO_B = 10.0
VENTANA_ESTANCAMIENTO_S = 3.0
AVANCE_MIN_M = 10.0
RETROCESO_MAX_M = 20.0


@dataclass
class Resultado:
    aptitud: float
    avance_m: float               # avance máximo alcanzado
    fraccion_vuelta: float
    checkpoints: int
    vuelta_completa: bool
    t_vuelta: float | None
    t_final: float
    motivo: str
    pasos: int
    registro: dict | None = field(default=None, repr=False)


def posicion_salida(pista: Pista, s0: float | None = None, lateral: float = 0.0,
                    d_rumbo: float = 0.0) -> EstadoCarro:
    """Carro parado sobre la línea central (centro 2.5 m después de la salida).

    `lateral` (m, + a la izquierda) y `d_rumbo` (rad) perturban la salida para la
    prueba de robustez del protocolo (20 salidas ligeramente aleatorias)."""
    if s0 is None:
        s0 = FISICA.largo / 2 + 0.5
    i = int(np.searchsorted(pista.s, s0) - 1) % len(pista.centro)
    a = pista.centro[i]; b = pista.centro[(i + 1) % len(pista.centro)]
    t = (s0 - pista.s[i]) / max(np.linalg.norm(b - a), 1e-9)
    p = a + (b - a) * t
    rumbo = float(math.atan2(*(b - a)[::-1]))
    p = p + lateral * np.array([-math.sin(rumbo), math.cos(rumbo)])
    return EstadoCarro(x=float(p[0]), y=float(p[1]), rumbo=rumbo + d_rumbo)


def aptitud(pista: Pista, avance: float, t_vuelta: float | None, t_max: float) -> tuple[float, int]:
    """N_cp + fracción del tramo actual (+ bono si completó la vuelta)."""
    k = len(pista.cp_s)
    vueltas, s = divmod(max(avance, 0.0), pista.longitud)
    cps = np.append(pista.cp_s, pista.longitud)
    j = int(np.searchsorted(cps, s, side="right") - 1)
    frac = (s - cps[j]) / max(cps[j + 1] - cps[j], 1e-9)
    n_cp = int(vueltas) * k + j
    f = n_cp + frac
    if t_vuelta is not None:
        f += BONO_B * max(0.0, (t_max - t_vuelta) / t_max)
    return float(f), n_cp


def correr(pista: Pista, controlador: Callable[[np.ndarray], np.ndarray], cuerpo: Cuerpo,
           t_limite: float, t_max: float, registrar: bool = False,
           f: ParamFisica = FISICA, salida: EstadoCarro | None = None) -> Resultado:
    e = posicion_salida(pista) if salida is None else EstadoCarro(salida.x, salida.y, salida.rumbo, salida.v)
    i_tramo = pista.indice_inicial((e.x, e.y))
    s_ant, i_tramo = pista.progreso_local((e.x, e.y), i_tramo)
    vueltas = 0
    avance = avance_max = s_ant
    s_ini = s_ant
    hist = deque(maxlen=int(round(VENTANA_ESTANCAMIENTO_S / DT)) + 1)
    entradas = np.zeros(7)
    reg = {k: [] for k in ("x", "y", "rumbo", "v", "entradas", "salidas", "oculta")} if registrar else None
    n_pasos = int(math.ceil(t_limite / DT))
    motivo, t_vuelta = "tiempo", None
    paso_i = 0
    if hasattr(controlador, "estado"):      # pilotos de referencia escritos a mano (no la red)
        controlador.estado = e
    for paso_i in range(1, n_pasos + 1):
        # --- sentidos
        d = pista.rayos((e.x, e.y), e.rumbo, cuerpo.angulos, cuerpo.alcances_m)
        entradas[:5] = d / cuerpo.alcances_m
        entradas[5] = e.v / f.v_tope
        s_local = avance - vueltas * pista.longitud
        j = int(np.searchsorted(pista.cp_s, s_local % pista.longitud, side="right")) % len(pista.cp_s)
        m = 0.5 * (pista.cp_izq[j] + pista.cp_der[j])
        ang = math.atan2(m[1] - e.y, m[0] - e.x) - e.rumbo
        entradas[6] = ((ang + math.pi) % (2 * math.pi) - math.pi) / math.pi
        # --- decisión y movimiento
        salidas = np.asarray(controlador(entradas), float)
        if registrar:
            reg["x"].append(e.x); reg["y"].append(e.y); reg["rumbo"].append(e.rumbo); reg["v"].append(e.v)
            reg["entradas"].append(entradas.copy()); reg["salidas"].append(salidas.copy())
            oc = getattr(controlador, "oculta", None)
            reg["oculta"].append(None if oc is None else np.array(oc))
        paso(e, float(salidas[0]), float(salidas[1]), cuerpo.v_max, f)
        t = paso_i * DT
        # --- avance (desenrollado por vueltas)
        s, i_tramo = pista.progreso_local((e.x, e.y), i_tramo)
        if s - s_ant < -0.5 * pista.longitud:
            vueltas += 1
        elif s - s_ant > 0.5 * pista.longitud:
            vueltas -= 1
        s_ant = s
        avance = s + vueltas * pista.longitud
        avance_max = max(avance_max, avance)
        hist.append(avance_max)
        # --- fin del episodio
        if pista.toca_borde(esquinas(e, f)):
            motivo = "choque"; break
        if avance_max - s_ini >= pista.longitud:
            motivo, t_vuelta = "vuelta", t; break
        if avance < avance_max - RETROCESO_MAX_M:
            motivo = "sentido_contrario"; break
        if len(hist) == hist.maxlen and hist[-1] - hist[0] < AVANCE_MIN_M:
            motivo = "estancamiento"; break
    avance_rel = avance_max - s_ini
    apt, n_cp = aptitud(pista, avance_rel, t_vuelta, t_max)
    return Resultado(aptitud=apt, avance_m=float(avance_rel), fraccion_vuelta=float(avance_rel / pista.longitud),
                     checkpoints=n_cp, vuelta_completa=t_vuelta is not None, t_vuelta=t_vuelta,
                     t_final=paso_i * DT, motivo=motivo, pasos=paso_i, registro=reg)
