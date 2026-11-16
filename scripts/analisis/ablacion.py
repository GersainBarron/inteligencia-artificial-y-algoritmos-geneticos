"""Ablación de rayos (protocolo v0.11, definición operativa). Solo lee las corridas.

En el campeón exacto de cada corrida (pob[0] de checkpoint.pkl) se fija en 1 la lectura
de un rayo a la vez y se evalúa en México desde 21 salidas (la estándar y las 20
perturbadas de la evaluación final: ±1.5 m, ±3°, semilla 12345), con T_max = 250 s.
Métrica: caída de la aptitud media y del porcentaje de vueltas respecto al campeón
intacto en las mismas 21 salidas.

Cómo se fija la lectura en 1 sin tocar el simulador: la entrada de la red es
distancia / alcance, y vale 1 cuando el rayo no toca ningún borde dentro de su alcance.
Dar al rayo un alcance de 1e-6 m (el carro nunca está a esa distancia de un borde sin
haber chocado antes) deja su lectura constante en 1. El script lo comprueba con el
registro de entradas de un episodio por campeón.

El protocolo pide B y C; A (rayos fijos de diseño) y R (fijos al azar) se incluyen como
referencia exploratoria.

Uso:  python scripts/analisis/ablacion.py
Escribe resultados/ablacion.csv (una fila por campeón y rayo; rayo = -1 es el intacto).
"""
from __future__ import annotations

import csv
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from evocarros.neuro.entrenar import runs_por_defecto  # noqa: E402
from evocarros.sim import episodio as ep  # noqa: E402
from evocarros.sim import rapido  # noqa: E402
from evocarros.sim.genoma import N_PESOS, Cuerpo, decodificar_cuerpo  # noqa: E402
from evocarros.sim.pista import Pista  # noqa: E402

T_MAX = 250.0
ALCANCE_APAGADO_M = 1e-6


def main():
    assert rapido.disponible, "se necesita el motor Numba (el mismo de las corridas)"
    pista = Pista.desde_json(RAIZ / "pistas/mexico/pista.json")
    rng = np.random.default_rng(12345)          # mismas salidas que scripts/evaluar_campeon.py
    pert = [(0.0, 0.0)] + [(float(rng.uniform(-1.5, 1.5)), float(np.radians(rng.uniform(-3, 3)))) for _ in range(20)]
    salidas = [ep.posicion_salida(pista, lateral=lat, d_rumbo=dr) for lat, dr in pert]
    filas, comprobados = [], 0
    for d in sorted(runs_por_defecto().iterdir()):
        if not d.is_dir() or not re.fullmatch(r"[ABCR]-(GA|ES)_s\d\d", d.name):
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        with open(d / "checkpoint.pkl", "rb") as f:
            g = np.asarray(pickle.load(f)["pob"][0], float)
        cuerpo = decodificar_cuerpo(g, cfg["condicion"])

        def evaluar(c: Cuerpo):
            res = [rapido.correr_red(pista, g[:N_PESOS], c, t_limite=T_MAX, t_max=T_MAX, salida=s) for s in salidas]
            return (float(np.mean([r.aptitud for r in res])), 100.0 * float(np.mean([r.vuelta_completa for r in res])),
                    float(np.mean([min(r.fraccion_vuelta, 1.0) for r in res])))

        base = evaluar(cuerpo)
        filas.append({"corrida": d.name, "condicion": cfg["condicion"], "semilla": cfg["semilla"], "rayo": -1,
                      "angulo_grados": "", "alcance_m": "", "aptitud_media": base[0], "pct_vueltas": base[1],
                      "fraccion_media": base[2], "caida_aptitud": 0.0, "caida_pct_vueltas": 0.0})
        for k in range(5):
            alc = cuerpo.alcances_m.copy(); alc[k] = ALCANCE_APAGADO_M
            c = Cuerpo(angulos=cuerpo.angulos.copy(), alcances_m=alc, v_max=cuerpo.v_max)
            if k == 0:      # comprobación: la entrada del rayo apagado vale 1 en todos los pasos
                r = rapido.correr_red(pista, g[:N_PESOS], c, t_limite=T_MAX, t_max=T_MAX, registrar=True, salida=salidas[0])
                ent = np.asarray(r.registro["entradas"] if isinstance(r.registro, dict) and "entradas" in r.registro else r.registro)
                col = ent[:r.pasos, 0] if ent.ndim == 2 and ent.shape[1] == 7 else ent[:r.pasos, 4]
                assert np.all(col == 1.0), f"{d.name}: el rayo apagado no lee 1"
                comprobados += 1
            m = evaluar(c)
            filas.append({"corrida": d.name, "condicion": cfg["condicion"], "semilla": cfg["semilla"], "rayo": k,
                          "angulo_grados": round(math.degrees(cuerpo.angulos[k]), 3), "alcance_m": round(float(cuerpo.alcances_m[k]), 3),
                          "aptitud_media": m[0], "pct_vueltas": m[1], "fraccion_media": m[2],
                          "caida_aptitud": base[0] - m[0], "caida_pct_vueltas": base[1] - m[1]})
        print(d.name, flush=True)
    out = RAIZ / "resultados" / "ablacion.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    print(f"{len(filas)} filas -> {out}; lectura fija en 1 comprobada en {comprobados} campeones")


if __name__ == "__main__":
    main()
