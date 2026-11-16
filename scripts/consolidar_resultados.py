"""Junta en una sola tabla los resultados de las 60 corridas de red (para la fase 8).

Uso:  python scripts/consolidar_resultados.py
Escribe resultados/resumen_redes.csv (una fila por corrida) en la carpeta del proyecto.
Solo lee; no modifica ninguna corrida. Excluye pilotos, pruebas y archivos (_*).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from evocarros.neuro.entrenar import runs_por_defecto  # noqa: E402

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
        if not d.is_dir() or d.name.startswith("_") or not (d / "estado.json").exists():
            continue
        if not re.fullmatch(r"[ABC]-(GA|ES)_s\d\d", d.name):   # excluye el piloto y las pruebas
            continue
        cfg, est, ev = leer(d / "config.json"), leer(d / "estado.json"), leer(d / "eval_final.json")
        with open(d / "fitness.csv", encoding="utf-8") as f:
            fit = list(csv.DictReader(f))
        primera = next((r for r in fit if float(r["pct_vuelta"] or 0) > 0), None)
        todos = next((r for r in fit if float(r["pct_vuelta"] or 0) >= 95), None)
        camp = sorted((d / "champions").glob("gen_*.json"))
        c = leer(camp[-1]) if camp else {}
        cond = cfg.get("condicion", d.name.split("_")[0])
        fila = {"corrida": d.name, "condicion": cond, "cuerpo": cond[0], "algoritmo": cond[2:],
                "semilla": cfg.get("semilla"), "protocolo": cfg.get("protocolo"), "motor": cfg.get("motor", "python"),
                "estado": est.get("estado"), "fin": est.get("fin"), "generaciones": est.get("generacion"),
                "episodios": est.get("episodios"), "mejor_aptitud": est.get("mejor_aptitud"),
                "pct_vuelta_final": est.get("pct_vuelta"), "mejor_t_vuelta": est.get("mejor_t_vuelta"),
                "episodios_primera_vuelta": primera and int(primera["episodios"]),
                "episodios_95pct_vuelta": todos and int(todos["episodios"]),
                "v_max_campeon": c.get("v_max"),
                "angulos_grados": json.dumps(c.get("angulos_grados")), "alcances_m": json.dumps(c.get("alcances_m"))}
        for p in PISTAS:
            e = (ev.get("pistas") or {}).get(p, {})
            s = e.get("salida_estandar", {})
            fila[f"{p}_completa"] = s.get("completa")
            fila[f"{p}_t_vuelta"] = s.get("t_vuelta")
            fila[f"{p}_fraccion"] = s.get("fraccion_vuelta")
            fila[f"{p}_pct_perturbadas"] = e.get("pct_completa_perturbadas")
            fila[f"{p}_fraccion_media_pert"] = e.get("fraccion_media_perturbadas")
        filas.append(fila)
    out = RAIZ / "resultados" / "resumen_redes.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"{len(filas)} corridas -> {out}")
    for cond in ("A-GA", "B-GA", "C-GA", "A-ES", "B-ES", "C-ES"):
        fs = [r for r in filas if r["condicion"] == cond]
        tv = sorted(r["mejor_t_vuelta"] for r in fs if r["mejor_t_vuelta"])
        print(f"{cond}: {len(fs)} corridas, {len(tv)} completan la vuelta; "
              f"mediana mejor vuelta {tv[len(tv) // 2]:.1f} s" if tv else f"{cond}: {len(fs)} corridas, ninguna completa")


if __name__ == "__main__":
    main()
