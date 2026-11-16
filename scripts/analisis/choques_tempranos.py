"""H3b (protocolo v0.11): fracción de episodios que terminan en choque en los primeros
10,000 episodios de cada corrida B y C.

Los motivos de fin de episodio no se guardaron por población, pero el simulador y los
algoritmos son deterministas: se re-ejecuta el inicio de cada corrida con la misma
semilla y el mismo código (clases del entrenador, sin modificarlo), contando el motivo
de cada episodio evaluado, y se verifica que la re-ejecución reproduce fila por fila el
fitness.csv oficial (todas las columnas menos "segundos").

Uso:  python scripts/analisis/choques_tempranos.py [--procesos 6]
Escribe resultados/choques_tempranos.csv. La re-ejecución escribe sus archivos en
resultados/_reejecucion/ (nunca en las corridas oficiales).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from evocarros.neuro import entrenar as E  # noqa: E402

N_TEMPRANOS = 10_000
CORRIDAS = [(c, s) for c in ("B-GA", "C-GA") for s in range(1, 21)] + \
           [(c, s) for c in ("B-ES", "C-ES") for s in range(1, 11)]


def reejecutar(cond: str, semilla: int, pool, salida: Path, oficiales: Path) -> dict:
    a = SimpleNamespace(condicion=cond, semilla=semilla, corrida=None, pista="pistas/mexico/pista.json",
                        t_max=250.0, episodios=100_000, procesos=pool._processes, runs=salida)
    c = E.CorridaES(a) if cond.endswith("ES") else E.Corrida(a)
    motivos, contados = Counter(), [0]
    evaluar_original = c.evaluar

    def evaluar(pool_, genomas):
        res = evaluar_original(pool_, genomas)
        for r in res:                       # en el orden en que se evalúan
            if contados[0] < N_TEMPRANOS:
                motivos[r.motivo] += 1
                contados[0] += 1
        return res

    c.evaluar = evaluar
    c.nuevo()
    if isinstance(c, E.CorridaES):
        c.inicial(pool)
    else:
        c.res = c.evaluar(pool, c.pob)
    c.registrar_generacion(0.0)
    # la corrida oficial pudo terminar (convergencia) antes de 10,000 episodios: no pasar de ahí
    with open(oficiales / c.id / "fitness.csv", encoding="utf-8") as f:
        ofi = list(csv.reader(f))
    ep_fin_oficial = int(ofi[-1][1])
    while c.episodios < min(N_TEMPRANOS, ep_fin_oficial):
        c.generacion(pool)
    with open(c.dir / "fitness.csv", encoding="utf-8") as f:
        mio = list(csv.reader(f))
    n = len(mio)
    igual = len(ofi) >= n and all(x[:9] == y[:9] for x, y in zip(mio, ofi[:n]))
    return {"corrida": c.id, "condicion": cond, "semilla": semilla, "episodios_contados": contados[0],
            "n_choque": motivos.get("choque", 0), "frac_choque": motivos.get("choque", 0) / contados[0],
            "motivos": json.dumps(dict(motivos), ensure_ascii=False), "filas_comparadas": n,
            "reproduce_oficial": igual}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=6)
    a = ap.parse_args()
    salida = RAIZ / "resultados" / "_reejecucion"
    salida.mkdir(parents=True, exist_ok=True)
    oficiales = E.runs_por_defecto()
    filas = []
    with Pool(a.procesos, initializer=E._init, initargs=(str(RAIZ / "pistas/mexico/pista.json"),)) as pool:
        for cond, s in CORRIDAS:
            f = reejecutar(cond, s, pool, salida, oficiales)
            filas.append(f)
            print(f["corrida"], "reproduce" if f["reproduce_oficial"] else "NO REPRODUCE", f["episodios_contados"], flush=True)
    out = RAIZ / "resultados" / "choques_tempranos.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    malas = [f["corrida"] for f in filas if not f["reproduce_oficial"]]
    print(f"{len(filas)} corridas -> {out}; no reproducen: {malas or 'ninguna'}")


if __name__ == "__main__":
    main()
