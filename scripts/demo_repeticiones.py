"""Genera una corrida de demostración para probar el visor y el panel (fase 2).

Uso:  python scripts/demo_repeticiones.py [pistas/mexico/pista.json] [carpeta_runs]

Crea `<runs>/_demo/` con:
- replays/gen_0000.json.gz: 19 redes aleatorias de la condición B (rayos y alcances
  al azar) más el piloto de referencia escrito a mano, guardado como "código" para
  probar la vista de la condición D.
- replays/gen_0001.json.gz: solo el piloto de referencia (para probar la carrera
  de fantasmas entre generaciones).
- estado.json y fitness.csv con el formato que escribirá el entrenador.
No es un resultado experimental: son datos inventados para probar la interfaz.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from evocarros.sim import episodio as ep  # noqa: E402
from evocarros.sim import repeticion as rp  # noqa: E402
from evocarros.sim.genoma import N_PESOS, N_RAYOS, Cuerpo, Red, decodificar_cuerpo, genoma_inicial  # noqa: E402
from evocarros.sim.pista import Pista  # noqa: E402
from evocarros.sim.referencia import CODIGO_DEMO, PilotoCentro  # noqa: E402


def carpeta_runs(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("EVOCARROS_RUNS"):
        return Path(os.environ["EVOCARROS_RUNS"])
    d = Path(r"D:\redes neuronales\runs")
    return d if d.exists() else RAIZ / "runs"


def main():
    args = sys.argv[1:]
    ruta_pista = Path(args[0]) if args else RAIZ / "pistas" / "mexico" / "pista.json"
    runs = carpeta_runs(args[1] if len(args) > 1 else None)
    pista = Pista.desde_json(ruta_pista)
    rel = ruta_pista.resolve().relative_to(RAIZ) if ruta_pista.resolve().is_relative_to(RAIZ) else ruta_pista
    dest = runs / "_demo"
    rng = np.random.default_rng(2026)

    carros = []
    for k in range(19):
        g = genoma_inicial(rng, "B")
        g[N_PESOS:N_PESOS + N_RAYOS] = np.sort(rng.uniform(-1.6, 1.6, N_RAYOS))
        g[N_PESOS + N_RAYOS:N_PESOS + 2 * N_RAYOS] = rng.uniform(1.0, 7.0, N_RAYOS)
        cuerpo = decodificar_cuerpo(g, "B")
        res = ep.correr(pista, Red(g[:N_PESOS]), cuerpo, t_limite=45.0, t_max=250.0, registrar=True)
        carros.append((res.aptitud, rp.carro(f"g0000_i{k:03d}", cuerpo, res, pesos=g[:N_PESOS])))
    base = decodificar_cuerpo(np.zeros(93), "A")
    cuerpo_ref = Cuerpo(base.angulos, base.alcances_m, 50.0)
    res_ref = ep.correr(pista, PilotoCentro(pista), cuerpo_ref, t_limite=250.0, t_max=250.0, registrar=True)
    ref = rp.carro("referencia", cuerpo_ref, res_ref, codigo=CODIGO_DEMO)
    carros.append((res_ref.aptitud, ref))
    carros.sort(key=lambda c: -c[0])
    lista = []
    for r, (_, c) in enumerate(carros, 1):
        c["rango"] = r
        lista.append(c)
    meta = dict(pista=pista.nombre, pista_archivo=str(rel).replace("\\", "/"), corrida="_demo",
                condicion="demo", semilla=2026)
    rp.guardar(dest / "replays" / "gen_0000.json.gz", lista, generacion=0, **meta)
    rp.guardar(dest / "replays" / "gen_0001.json.gz", [dict(ref, rango=1)], generacion=1, **meta)

    with open(dest / "fitness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generacion", "aptitud_max", "aptitud_media", "aptitud_mediana", "pct_vuelta",
                    "mejor_t_vuelta", "diversidad", "t_limite", "segundos"])
        for gnum in range(60):   # curva inventada solo para probar la gráfica del panel
            m = 50 * (1 - np.exp(-gnum / 15))
            w.writerow([gnum, round(m + 5 * (gnum > 40), 3), round(0.6 * m, 3), round(0.55 * m, 3),
                        round(min(100, max(0, (gnum - 35) * 4)), 1), "", round(1.5 * np.exp(-gnum / 30), 3),
                        min(250, 30 * 1.5 ** (gnum // 10)), 12.0])
    estado = {"corrida": "_demo", "condicion": "demo", "semilla": 2026, "estado": "demostracion",
              "generacion": 59, "generaciones_max": 1000, "mejor_aptitud": 55.0, "pct_vuelta": 96.0,
              "mejor_t_vuelta": res_ref.t_vuelta, "eta_s": None, "pista": meta["pista_archivo"],
              "actualizado": datetime.now().isoformat(timespec="seconds")}
    (dest / "estado.json").write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK demo en {dest}: {len(lista)} carros; referencia: {res_ref.motivo}, "
          f"{res_ref.t_vuelta and round(res_ref.t_vuelta, 1)} s")


if __name__ == "__main__":
    main()
