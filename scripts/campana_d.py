"""Cola de la condición D (mini-AlphaEvolve): semillas 1 a 5, una tras otra.

Corre en la GPU, en paralelo con la campaña de redes (que usa el CPU).
Uso: python scripts/campana_d.py [--semillas 1-5]
- Salta las corridas terminadas; reanuda las incompletas (la base SQLite es el estado).
- Al terminar cada corrida evalúa al campeón (código) con la misma evaluación final que
  las redes (scripts/evaluar_campeon_d.py).
- Si una corrida queda "detenida" desde el panel, la cola se detiene.
- Candado propio (runs/campana_d.lock) y bitácora runs/campana_d.log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
from campana import semillas, tomar_candado  # noqa: E402
from evocarros.neuro.entrenar import runs_por_defecto  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", default="1-5")
    ap.add_argument("--runs", type=Path, default=runs_por_defecto())
    a = ap.parse_args()
    a.runs.mkdir(parents=True, exist_ok=True)
    log = a.runs / "campana_d.log"

    def anotar(t):
        linea = f"{datetime.now().isoformat(timespec='seconds')} {t}"
        print(linea, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(linea + "\n")

    candado = a.runs / "campana_d.lock"
    if not tomar_candado(candado):
        print("ya hay una cola D corriendo; salgo")
        return
    try:
        for s in semillas(a.semillas):
            d = a.runs / f"D-LLM_s{s:02d}"
            est = json.loads((d / "estado.json").read_text(encoding="utf-8")) if (d / "estado.json").exists() else {}
            if est.get("estado") != "terminada":
                if (d / "control.json").exists():
                    (d / "control.json").unlink()
                anotar(f"inicia D-LLM_s{s:02d}")
                subprocess.run([sys.executable, "-m", "evocarros.llm_evo.evolucionar", "--semilla", str(s),
                                "--runs", str(a.runs)], cwd=RAIZ)
                est = json.loads((d / "estado.json").read_text(encoding="utf-8")) if (d / "estado.json").exists() else {}
                anotar(f"fin D-LLM_s{s:02d}: estado {est.get('estado')}, fin {est.get('fin')}, programas "
                       f"{est.get('programas')}, mejor aptitud {est.get('mejor_aptitud')}, mejor vuelta {est.get('mejor_t_vuelta')}")
                if est.get("estado") == "detenida":
                    anotar("cola D detenida desde el panel")
                    return
            if est.get("estado") == "terminada" and not (d / "eval_final.json").exists():
                r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "evaluar_campeon_d.py"), str(d)],
                                   cwd=RAIZ, capture_output=True, text=True)
                anotar(f"evaluación D-LLM_s{s:02d}: " + (r.stdout.strip().replace("\n", " | ") or r.stderr[-300:]))
        anotar("cola D terminada")
    finally:
        candado.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
