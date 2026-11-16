"""Evaluación final de una corrida: generalización y robustez del campeón (protocolo v0.6).

Uso:  python scripts/evaluar_campeon.py "D:\\redes neuronales\\runs\\A-GA_s01" [--salidas 20]

Toma el campeón final (el mejor genoma de checkpoint.pkl, con precisión completa; si no
hay checkpoint, el archivo más reciente de <corrida>/champions/) y lo evalúa, sin reentrenar:
- en la pista de entrenamiento y en las 3 de prueba (Spa, Monza, Hungaroring);
- desde la salida estándar y desde 20 salidas ligeramente aleatorias (desplazamiento
  lateral uniforme de ±1.5 m y giro de ±3°; semilla fija 12345, iguales para todas
  las corridas, para que las comparaciones sean pareadas).
Cada pista usa su propio T_max = 2 × vuelta ideal (scripts/perfil_velocidad.py) como
tiempo límite. Los alcances de los rayos se conservan en metros (no se reescalan al
ancho de cada pista).
Escribe <corrida>/eval_final.csv (una fila por pista y salida) y eval_final.json (resumen).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
from evocarros.sim import episodio as ep  # noqa: E402
from evocarros.sim import rapido  # noqa: E402
from evocarros.sim.genoma import N_PESOS, Red, decodificar_cuerpo  # noqa: E402
from evocarros.sim.pista import Pista  # noqa: E402
from perfil_velocidad import perfil  # noqa: E402

PISTAS = {"mexico": "entrenamiento", "spa": "prueba", "monza": "prueba", "hungaroring": "prueba"}


def evaluar(corrida: Path, n_salidas: int = 20) -> dict:
    cfg = json.loads((corrida / "config.json").read_text(encoding="utf-8"))
    camp = sorted((corrida / "champions").glob("gen_*.json"))[-1]
    c = json.loads(camp.read_text(encoding="utf-8"))
    g = np.array(c["genoma"])
    ckpt = corrida / "checkpoint.pkl"
    origen = "champions (json)"
    if ckpt.exists():
        import pickle
        with open(ckpt, "rb") as f:
            estado = pickle.load(f)
        g = np.asarray(estado["pob"][0], dtype=float)   # el mejor de la población final, exacto
        c["generacion"] = estado["gen"]
        origen = "checkpoint.pkl (exacto)"
    cuerpo = decodificar_cuerpo(g, cfg["condicion"])
    rng = np.random.default_rng(12345)
    pert = [(0.0, 0.0)] + [(float(rng.uniform(-1.5, 1.5)), float(np.radians(rng.uniform(-3, 3))))
                           for _ in range(n_salidas)]
    filas, resumen = [], {"corrida": corrida.name, "condicion": cfg["condicion"], "campeon": camp.name,
                          "generacion": c["generacion"], "origen_genoma": origen, "pistas": {}}
    for nombre, uso in PISTAS.items():
        ruta = RAIZ / "pistas" / nombre / "pista.json"
        pista = Pista.desde_json(ruta)
        t_max = round(2 * perfil(str(ruta))["t_vuelta_ideal_s"], 1)
        res = []
        for k, (lat, dr) in enumerate(pert):
            salida = ep.posicion_salida(pista, lateral=lat, d_rumbo=dr)
            if rapido.disponible:
                r = rapido.correr_red(pista, g[:N_PESOS], cuerpo, t_limite=t_max, t_max=t_max, salida=salida)
            else:
                r = ep.correr(pista, Red(g[:N_PESOS]), cuerpo, t_limite=t_max, t_max=t_max, salida=salida)
            res.append(r)
            filas.append({"pista": nombre, "uso": uso, "salida": k, "lateral_m": round(lat, 3),
                          "giro_grados": round(math.degrees(dr), 3), "completa": r.vuelta_completa,
                          "t_vuelta": None if r.t_vuelta is None else round(r.t_vuelta, 3),
                          "fraccion_vuelta": round(min(r.fraccion_vuelta, 1.0), 4), "motivo": r.motivo})
        pert_res = res[1:]
        tv = [r.t_vuelta for r in pert_res if r.vuelta_completa]
        resumen["pistas"][nombre] = {
            "uso": uso, "t_max": t_max,
            "salida_estandar": {"completa": res[0].vuelta_completa, "t_vuelta": res[0].t_vuelta,
                                "fraccion_vuelta": round(min(res[0].fraccion_vuelta, 1.0), 4), "motivo": res[0].motivo},
            "pct_completa_perturbadas": round(100 * len(tv) / len(pert_res), 1),
            "t_vuelta_mediana_perturbadas": None if not tv else round(float(np.median(tv)), 3),
            "fraccion_media_perturbadas": round(float(np.mean([min(r.fraccion_vuelta, 1.0) for r in pert_res])), 4)}
    with open(corrida / "eval_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    (corrida / "eval_final.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corrida", type=Path)
    ap.add_argument("--salidas", type=int, default=20)
    a = ap.parse_args()
    r = evaluar(a.corrida, a.salidas)
    for nombre, d in r["pistas"].items():
        e = d["salida_estandar"]
        print(f"{nombre:12s} ({d['uso']:13s}) estándar: {'vuelta ' + format(e['t_vuelta'], '.1f') + ' s' if e['completa'] else e['motivo'] + ' al ' + format(100 * e['fraccion_vuelta'], '.0f') + '%'}"
              f" | perturbadas: {d['pct_completa_perturbadas']}% completan, avance medio {100 * d['fraccion_media_perturbadas']:.0f}%")
