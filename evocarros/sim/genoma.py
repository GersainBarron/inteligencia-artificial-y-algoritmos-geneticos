"""Genoma del piloto y red neuronal 7:8:2 (protocolo v0.4).

Disposición del vector (93 números):
  [0:82]   pesos de la red: W1 (8x7), b1 (8), W2 (2x8), b2 (2), en ese orden
  [82:87]  ángulos de los 5 rayos (radianes, respecto al frente)
  [87:92]  alcances de los 5 rayos (en unidades de W de la pista de entrenamiento)
  [92]     velocidad máxima (m/s)

En las condiciones A los genes de cuerpo se ignoran y se usan los valores fijos;
en A y B la velocidad máxima es fija (0.7 v_tope). El algoritmo evolutivo decide
qué genes muta; este módulo solo decodifica y recorta a sus rangos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

N_ENT, N_OCU, N_SAL = 7, 8, 2
N_PESOS = (N_ENT + 1) * N_OCU + (N_OCU + 1) * N_SAL      # 82
N_RAYOS = 5
N_GENOMA = N_PESOS + 2 * N_RAYOS + 1                      # 93

W_ENTRENAMIENTO = 12.249   # m, ancho medio del Hermanos Rodríguez (pista.json)
ANGULOS_FIJOS = np.radians([-60.0, -30.0, 0.0, 30.0, 60.0])
ALCANCE_FIJO_W = 4.0
ALCANCE_MIN_W, ALCANCE_MAX_W = 0.5, 8.0
V_MIN = 15.0                # m/s, límite inferior del gen de velocidad (supuesto)


@dataclass
class Cuerpo:
    angulos: np.ndarray        # rad
    alcances_m: np.ndarray     # m
    v_max: float               # m/s


def decodificar_cuerpo(g: np.ndarray, condicion: str, v_tope: float = 50.0,
                       w: float = W_ENTRENAMIENTO) -> Cuerpo:
    """Cuerpo del carro según la condición (A: fijo; B: rayos evolutivos; C: B + velocidad)."""
    if condicion.upper().startswith("A") or condicion.upper().startswith("D"):
        ang, alc = ANGULOS_FIJOS.copy(), np.full(N_RAYOS, ALCANCE_FIJO_W)
    else:
        ang = (g[N_PESOS:N_PESOS + N_RAYOS] + math.pi) % (2 * math.pi) - math.pi
        alc = np.clip(g[N_PESOS + N_RAYOS:N_PESOS + 2 * N_RAYOS], ALCANCE_MIN_W, ALCANCE_MAX_W)
    if condicion.upper().startswith("C"):
        vmax = float(np.clip(g[N_GENOMA - 1], V_MIN, v_tope))
    else:
        vmax = 0.7 * v_tope
    return Cuerpo(angulos=np.asarray(ang, float), alcances_m=np.asarray(alc, float) * w, v_max=vmax)


def genoma_inicial(rng: np.random.Generator, condicion: str, v_tope: float = 50.0) -> np.ndarray:
    g = np.empty(N_GENOMA)
    g[:N_PESOS] = rng.normal(0.0, 0.5, N_PESOS)
    g[N_PESOS:N_PESOS + N_RAYOS] = ANGULOS_FIJOS
    g[N_PESOS + N_RAYOS:N_PESOS + 2 * N_RAYOS] = ALCANCE_FIJO_W
    g[N_GENOMA - 1] = 0.7 * v_tope
    return g


class Red:
    """Perceptrón 7:8:2 con tanh. Guarda la última activación para el visor."""

    def __init__(self, pesos: np.ndarray):
        p = np.asarray(pesos, float)
        i = 0
        self.W1 = p[i:i + N_OCU * N_ENT].reshape(N_OCU, N_ENT); i += N_OCU * N_ENT
        self.b1 = p[i:i + N_OCU]; i += N_OCU
        self.W2 = p[i:i + N_SAL * N_OCU].reshape(N_SAL, N_OCU); i += N_SAL * N_OCU
        self.b2 = p[i:i + N_SAL]
        self.oculta = np.zeros(N_OCU)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.oculta = np.tanh(self.W1 @ x + self.b1)
        return np.tanh(self.W2 @ self.oculta + self.b2)
