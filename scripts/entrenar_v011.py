"""Entrenador para el protocolo v0.11: envuelve a evocarros.neuro.entrenar SIN modificarlo.

Uso (igual que el entrenador):
    python scripts/entrenar_v011.py --condicion R-GA --semilla 3
    python scripts/entrenar_v011.py --condicion B-GA --semilla 14

Qué agrega:
- Condición de control R-GA (protocolo v0.11): rayos fijos sorteados al azar, una
  configuración por semilla s, en el rango en que B puede evolucionarlos: 5 ángulos
  uniformes en (-180°, 180°) y 5 alcances uniformes en (0.5 W, 8 W), con el generador
  numpy.random.default_rng(110000 + s) (primero ángulos, luego alcances). Ese generador
  es distinto del de la corrida: los pesos iniciales son idénticos a los de A, B y C con
  la misma semilla (diseño pareado). Los 50 individuos comparten la configuración y
  nunca muta (el entrenador solo muta rayos en B y C); velocidad máxima fija (35 m/s);
  mismo genético, presupuesto y convergencia que A-GA.
- Las corridas lanzadas con este script registran protocolo "v0.11" en config.json.

No toca el simulador ni el entrenador: solo sustituye, en este proceso, la función que
crea el genoma inicial y la lista de condiciones admitidas. Los procesos trabajadores
usan el código original (decodificar_cuerpo lee los rayos del genoma para toda condición
que no empiece con A o D).
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from evocarros.neuro import entrenar as E  # noqa: E402
from evocarros.sim.genoma import ALCANCE_MAX_W, ALCANCE_MIN_W, N_PESOS, N_RAYOS, W_ENTRENAMIENTO  # noqa: E402

SEMILLA_BASE_CONTROL = 110000


def rayos_control(semilla: int):
    """Configuración de rayos del control R para una semilla: (ángulos en rad, alcances en W)."""
    rng = np.random.default_rng(SEMILLA_BASE_CONTROL + semilla)
    ang = np.radians(rng.uniform(-180.0, 180.0, N_RAYOS))
    alc = rng.uniform(ALCANCE_MIN_W, ALCANCE_MAX_W, N_RAYOS)
    return ang, alc


def instalar(semilla: int):
    original = E.genoma_inicial

    def genoma_inicial(rng, condicion, v_tope=50.0):
        g = original(rng, condicion, v_tope)          # mismos pesos que A, B y C con esta semilla
        if condicion.upper().startswith("R"):
            ang, alc = rayos_control(semilla)
            g[N_PESOS:N_PESOS + N_RAYOS] = ang
            g[N_PESOS + N_RAYOS:N_PESOS + 2 * N_RAYOS] = alc
        return g

    class Parser(argparse.ArgumentParser):
        def add_argument(self, *a, **k):
            if a and a[0] == "--condicion":
                k["choices"] = list(k["choices"]) + ["R-GA"]
            return super().add_argument(*a, **k)

    E.genoma_inicial = genoma_inicial
    E.argparse = types.SimpleNamespace(ArgumentParser=Parser)
    E.PROTOCOLO = "v0.11"


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--condicion", default="A-GA")
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--corrida", default=None)
    ap.add_argument("--runs", type=Path, default=E.runs_por_defecto())
    a, _ = ap.parse_known_args()
    instalar(a.semilla)
    if a.condicion.startswith("R"):
        d = a.runs / (a.corrida or f"{a.condicion}_s{a.semilla:02d}")
        d.mkdir(parents=True, exist_ok=True)
        ang, alc = rayos_control(a.semilla)
        (d / "rayos_control.json").write_text(json.dumps({
            "semilla": a.semilla, "generador": f"numpy.random.default_rng({SEMILLA_BASE_CONTROL} + semilla)",
            "angulos_grados": np.degrees(ang).round(3).tolist(), "alcances_W": alc.round(4).tolist(),
            "alcances_m": (alc * W_ENTRENAMIENTO).round(3).tolist()}, indent=2), encoding="utf-8")
    E.main()


if __name__ == "__main__":
    main()
