"""Evaluación final del campeón de una corrida de la condición D (mismo formato que las redes).

Uso:  python scripts/evaluar_campeon_d.py "D:\\redes neuronales\\runs\\D-LLM_s01" [--sandbox docker]

Igual que scripts/evaluar_campeon.py (4 pistas, salida estándar + 20 perturbadas con la misma
semilla 12345, T_max propio de cada pista), pero el código del campeón corre dentro del
contenedor aislado. Escribe eval_final.csv y eval_final.json en la corrida.
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
from evaluar_campeon import PISTAS  # noqa: E402
from evocarros.llm_evo.base import Base  # noqa: E402
from evocarros.llm_evo.evaluador import evaluar  # noqa: E402
from perfil_velocidad import perfil  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corrida", type=Path)
    ap.add_argument("--salidas", type=int, default=20)
    ap.add_argument("--sandbox", default="docker", choices=["docker", "local"])
    a = ap.parse_args()
    m = Base(a.corrida / "programas.sqlite").mejor()
    rng = np.random.default_rng(12345)
    pert = [(0.0, 0.0)] + [(float(rng.uniform(-1.5, 1.5)), float(np.radians(rng.uniform(-3, 3))))
                           for _ in range(a.salidas)]
    pistas = {}
    for nombre in PISTAS:
        ruta_host = RAIZ / "pistas" / nombre / "pista.json"
        ruta = f"/app/pistas/{nombre}/pista.json" if a.sandbox == "docker" else str(ruta_host)
        pistas[nombre] = {"ruta": ruta, "t_max": round(2 * perfil(str(ruta_host))["t_vuelta_ideal_s"], 1)}
    r = evaluar(m["codigo"], a.corrida / "tmp", modo=a.sandbox, todas_las_pistas=True, limite_s=3600,
                peticion_extra={"evaluacion_final": True, "pistas": pistas, "perturbaciones": pert})
    if not r.get("ok"):
        raise SystemExit(f"falló la evaluación: {r.get('error')}")
    filas, resumen = [], {"corrida": a.corrida.name, "condicion": "D-LLM", "campeon": f"prog_{m['id']:05d}",
                          "origen_genoma": "programas.sqlite (código del mejor programa)", "pistas": {}}
    for nombre, res in r["pistas"].items():
        for f, (lat, dr) in zip(res, pert):
            filas.append({"pista": nombre, "uso": PISTAS[nombre], "salida": f["salida"], "lateral_m": round(lat, 3),
                          "giro_grados": round(math.degrees(dr), 3), "completa": f["completa"],
                          "t_vuelta": f["t_vuelta"], "fraccion_vuelta": round(f["fraccion_vuelta"], 4), "motivo": f["motivo"]})
        p = res[1:]
        tv = [f["t_vuelta"] for f in p if f["completa"]]
        resumen["pistas"][nombre] = {
            "uso": PISTAS[nombre], "t_max": pistas[nombre]["t_max"],
            "salida_estandar": {k: res[0][k] for k in ("completa", "t_vuelta", "fraccion_vuelta", "motivo")},
            "pct_completa_perturbadas": round(100 * len(tv) / len(p), 1),
            "t_vuelta_mediana_perturbadas": None if not tv else round(float(np.median(tv)), 3),
            "fraccion_media_perturbadas": round(float(np.mean([f["fraccion_vuelta"] for f in p])), 4)}
    with open(a.corrida / "eval_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    (a.corrida / "eval_final.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    for nombre, d in resumen["pistas"].items():
        e = d["salida_estandar"]
        print(f"{nombre:12s} estándar: {'vuelta ' + format(e['t_vuelta'], '.1f') + ' s' if e['completa'] else e['motivo'] + ' al ' + format(100 * e['fraccion_vuelta'], '.0f') + '%'}"
              f" | perturbadas: {d['pct_completa_perturbadas']}% completan")


if __name__ == "__main__":
    main()
