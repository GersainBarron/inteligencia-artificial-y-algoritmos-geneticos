"""Formato fijo de repeticiones: esquema `evocarros.repeticion/1`.

Un archivo = una generación de una corrida, comprimido con gzip:
`<corrida>/replays/gen_XXXX.json.gz`. El formato queda fijo desde la fase 2 para
poder ver cualquier corrida antigua; si cambia, sube la versión del esquema y el
visor debe seguir leyendo las anteriores.

{
  "esquema": "evocarros.repeticion/1",
  "pista": "Autódromo Hermanos Rodríguez", "pista_archivo": "pistas/mexico/pista.json",
  "dt": 0.0333333, "corrida": "A-GA_s01", "condicion": "A-GA", "semilla": 1, "generacion": 120,
  "carros": [
    {"id": "g0120_i007", "padre": "g0119_i003", "rango": 1,
     "cuerpo": {"angulos_rad": [5], "alcances_m": [5], "v_max": 35.0},
     "controlador": {"tipo": "red", "arquitectura": [7, 8, 2], "activacion": "tanh",
                     "orden_pesos": "W1(8x7) b1(8) W2(2x8) b2(2)", "pesos": [82]},
         o bien {"tipo": "codigo", "codigo": "def manejar(...): ..."} en la condición D,
     "resultado": {"aptitud": .., "avance_m": .., "fraccion_vuelta": .., "checkpoints": ..,
                   "vuelta_completa": .., "t_vuelta": .., "t_final": .., "motivo": ".."},
     "pasos": {"x": [..], "y": [..], "rumbo": [..], "v": [..],
               "entradas": [[7], ..], "salidas": [[2], ..],
               "lineas": [[..], ..]   # solo D: líneas de código ejecutadas en cada paso
     }}
  ]
}

Precisión guardada: posición 1 cm, rumbo 1e-4 rad, velocidad 1 cm/s, entradas y
salidas 1e-3. Las activaciones de la capa oculta no se guardan: el visor las
recalcula con los pesos y las entradas de cada paso.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .episodio import Resultado
from .fisica import DT
from .genoma import Cuerpo

ESQUEMA = "evocarros.repeticion/1"
N_GUARDADOS = 20


def _r(a, nd):
    return np.round(np.asarray(a, float), nd).tolist()


def carro(id_: str, cuerpo: Cuerpo, res: Resultado, pesos=None, codigo: str | None = None,
          padre: str | None = None, rango: int | None = None, lineas=None) -> dict:
    if res.registro is None:
        raise ValueError("el episodio se corrió sin registrar=True")
    reg = res.registro
    if codigo is None:
        ctrl = {"tipo": "red", "arquitectura": [7, 8, 2], "activacion": "tanh",
                "orden_pesos": "W1(8x7) b1(8) W2(2x8) b2(2)", "pesos": _r(pesos, 6)}
    else:
        ctrl = {"tipo": "codigo", "codigo": codigo}
    pasos = {"x": _r(reg["x"], 2), "y": _r(reg["y"], 2), "rumbo": _r(reg["rumbo"], 4), "v": _r(reg["v"], 2),
             "entradas": _r(reg["entradas"], 3), "salidas": _r(reg["salidas"], 3)}
    if lineas is not None:
        pasos["lineas"] = lineas
    return {"id": id_, "padre": padre, "rango": rango,
            "cuerpo": {"angulos_rad": _r(cuerpo.angulos, 5), "alcances_m": _r(cuerpo.alcances_m, 3),
                       "v_max": round(float(cuerpo.v_max), 3)},
            "controlador": ctrl,
            "resultado": {"aptitud": round(res.aptitud, 5), "avance_m": round(res.avance_m, 2),
                          "fraccion_vuelta": round(res.fraccion_vuelta, 5), "checkpoints": res.checkpoints,
                          "vuelta_completa": res.vuelta_completa,
                          "t_vuelta": None if res.t_vuelta is None else round(res.t_vuelta, 3),
                          "t_final": round(res.t_final, 3), "motivo": res.motivo},
            "pasos": pasos}


def guardar(ruta: str | Path, carros: list[dict], *, pista: str, pista_archivo: str, corrida: str,
            condicion: str, semilla: int, generacion: int) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    doc = {"esquema": ESQUEMA, "pista": pista, "pista_archivo": pista_archivo, "dt": DT,
           "corrida": corrida, "condicion": condicion, "semilla": semilla, "generacion": generacion,
           "carros": carros}
    with gzip.open(ruta, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    return ruta


def cargar(ruta: str | Path) -> dict:
    ruta = Path(ruta)
    abrir = gzip.open if ruta.suffix == ".gz" else open
    with abrir(ruta, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("esquema") != ESQUEMA:
        raise ValueError(f"esquema desconocido: {doc.get('esquema')}")
    return doc
