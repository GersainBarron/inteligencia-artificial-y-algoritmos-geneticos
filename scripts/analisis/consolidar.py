"""Consolida las corridas de red del protocolo v0.11/v0.12 (fase 8). Solo lee las corridas.

Uso:  python scripts/analisis/consolidar.py
Escribe resultados/resumen_redes_v011.csv: una fila por corrida de A, B, C y R con GA
(semillas 1-20) y de A, B, C con ES (semillas 1-10). Además de las columnas de
scripts/consolidar_resultados.py agrega los genes de rayos del campeón exacto
(pob[0] de checkpoint.pkl): ángulos crudos en radianes (sin envolver) y alcances en W.
"""
from __future__ import annotations

import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from evocarros.neuro.entrenar import runs_por_defecto  # noqa: E402
from evocarros.sim.genoma import N_GENOMA, N_PESOS, N_RAYOS  # noqa: E402

PISTAS = ["mexico", "spa", "monza", "hungaroring"]


def leer(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    runs = runs_por_defecto()
    filas = []
    for d in sorted(runs.iterdir()):
        if not d.is_dir() or not re.fullmatch(r"[ABCR]-(GA|ES)_s\d\d", d.name):
            continue
        cfg, est, ev = leer(d / "config.json"), leer(d / "estado.json"), leer(d / "eval_final.json")
        assert est.get("estado") == "terminada" and ev, f"{d.name}: sin terminar o sin evaluación"
        with open(d / "fitness.csv", encoding="utf-8") as f:
            fit = list(csv.DictReader(f))
        primera = next((r for r in fit if float(r["pct_vuelta"] or 0) > 0), None)
        todos = next((r for r in fit if float(r["pct_vuelta"] or 0) >= 95), None)
        with open(d / "checkpoint.pkl", "rb") as f:
            ck = pickle.load(f)
        g = np.asarray(ck["pob"][0], float)
        r0 = ck["res"][0]
        cond = cfg["condicion"]
        fila = {"corrida": d.name, "condicion": cond, "cuerpo": cond[0], "algoritmo": cond[2:],
                "semilla": cfg["semilla"], "protocolo": cfg.get("protocolo"), "motor": cfg.get("motor", "python"),
                "fin": est.get("fin"), "generaciones": est.get("generacion"), "episodios": est.get("episodios"),
                "mejor_aptitud": est.get("mejor_aptitud"), "aptitud_campeon": r0.aptitud,
                "pct_vuelta_final": est.get("pct_vuelta"), "mejor_t_vuelta": est.get("mejor_t_vuelta"),
                "t_vuelta_campeon": r0.t_vuelta,
                "episodios_primera_vuelta": primera and int(primera["episodios"]),
                "episodios_95pct_vuelta": todos and int(todos["episodios"]),
                "v_max_gen": float(g[N_GENOMA - 1]),
                "genes_ang_rad": json.dumps([float(x) for x in g[N_PESOS:N_PESOS + N_RAYOS]]),
                "genes_alc_W": json.dumps([float(x) for x in g[N_PESOS + N_RAYOS:N_PESOS + 2 * N_RAYOS]])}
        for p in PISTAS:
            e = (ev.get("pistas") or {}).get(p, {})
            s = e.get("salida_estandar", {})
            fila[f"{p}_completa"] = s.get("completa")
            fila[f"{p}_t_vuelta"] = s.get("t_vuelta")
            fila[f"{p}_fraccion"] = s.get("fraccion_vuelta")
            fila[f"{p}_pct_perturbadas"] = e.get("pct_completa_perturbadas")
            fila[f"{p}_fraccion_media_pert"] = e.get("fraccion_media_perturbadas")
        filas.append(fila)
    out = RAIZ / "resultados" / "resumen_redes_v011.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    from collections import Counter
    print(f"{len(filas)} corridas -> {out}")
    print(sorted(Counter(r["condicion"] for r in filas).items()))


if __name__ == "__main__":
    main()
