"""Cola del protocolo v0.11: amplía el brazo genético a 20 semillas y corre el control R-GA.

Uso:
    python scripts/campana_v011.py                      # A-GA, B-GA, C-GA y R-GA, semillas 1-20
    python scripts/campana_v011.py --semillas 11-20 --condiciones A-GA,B-GA

- Usa scripts/entrenar_v011.py (envoltorio; no modifica el entrenador ni el simulador).
- Salta las corridas "terminada" (las 30 de A, B y C con semillas 1-10 ya lo están);
  reanuda las incompletas desde su checkpoint; evalúa al campeón al terminar cada una.
- Obedece el mismo contrato que la cola original: control.json por corrida (panel y
  guardián), "detenida" detiene la cola. Candado propio: <runs>/campana_v011.lock
  (una segunda instancia sale sin hacer nada).
- Nunca entrena en paralelo con la cola original: antes de cada corrida espera a que
  campana.py suelte su candado (campana.lock).
- Bitácora: <runs>/campana_v011.log. arranque_evocarros.ps1 la relanza al iniciar sesión
  (continúa donde se quedó).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
from campana import proceso_vivo, semillas, tomar_candado  # noqa: E402
from evocarros.neuro.entrenar import runs_por_defecto  # noqa: E402

CONDICIONES = ["A-GA", "B-GA", "C-GA", "R-GA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", default="1-20")
    ap.add_argument("--condiciones", default=",".join(CONDICIONES))
    ap.add_argument("--procesos", type=int, default=12)
    ap.add_argument("--runs", type=Path, default=runs_por_defecto())
    ap.add_argument("--episodios", type=int, default=None, help="solo para pruebas cortas")
    a = ap.parse_args()
    a.runs.mkdir(parents=True, exist_ok=True)
    log = a.runs / "campana_v011.log"

    def anotar(t):
        linea = f"{datetime.now().isoformat(timespec='seconds')} {t}"
        print(linea, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(linea + "\n")

    candado = a.runs / "campana_v011.lock"
    if not tomar_candado(candado):
        print("ya hay una cola v0.11 corriendo; salgo")
        return
    try:
        esperar_cola_original(a.runs, anotar)
        cola = [(s, c) for s in semillas(a.semillas) for c in a.condiciones.split(",")]
        pendientes = []
        if pasada(a, anotar, cola, pendientes) and pendientes:
            anotar(f"reintento de {len(pendientes)} corrida(s)")
            pasada(a, anotar, pendientes, [])
        anotar("campaña v0.11 terminada")
    finally:
        candado.unlink(missing_ok=True)


def cola_original_viva(runs: Path) -> bool:
    vieja = runs / "campana.lock"
    try:
        return vieja.exists() and proceso_vivo(int(vieja.read_text().strip()))
    except (ValueError, OSError):
        return False


def esperar_cola_original(runs: Path, anotar, cada_s: float = 30.0):
    """Nunca entrena en paralelo con la cola original (campana.py): espera a que suelte su candado."""
    avisado = False
    while cola_original_viva(runs):
        if not avisado:
            anotar("la cola original (campana.py) está viva; espero a que termine")
            avisado = True
        time.sleep(cada_s)


def pasada(a, anotar, cola, pendientes) -> bool:
    anotar(f"campaña v0.11: {len(cola)} corridas en cola")
    for s, c in cola:
        cid = f"{c}_s{s:02d}"
        d = a.runs / cid
        est = json.loads((d / "estado.json").read_text(encoding="utf-8")) if (d / "estado.json").exists() else {}
        if est.get("estado") != "terminada":
            if (d / "control.json").exists():
                (d / "control.json").unlink()
            esperar_cola_original(a.runs, anotar)
            anotar(f"inicia {cid}")
            cmd = [sys.executable, str(RAIZ / "scripts" / "entrenar_v011.py"), "--condicion", c, "--semilla", str(s),
                   "--procesos", str(a.procesos), "--runs", str(a.runs)]
            if a.episodios:
                cmd += ["--episodios", str(a.episodios)]
            r = subprocess.run(cmd, cwd=RAIZ)
            est = json.loads((d / "estado.json").read_text(encoding="utf-8")) if (d / "estado.json").exists() else {}
            anotar(f"fin {cid}: código {r.returncode}, estado {est.get('estado')}, fin {est.get('fin')}, "
                   f"gen {est.get('generacion')}, episodios {est.get('episodios')}")
            if est.get("estado") == "detenida":
                anotar("campaña v0.11 detenida desde el panel")
                return False
            if est.get("estado") != "terminada":
                anotar(f"{cid} no terminó (estado {est.get('estado')}); sigo con la siguiente")
                pendientes.append((s, c))
                continue
        if not (d / "eval_final.json").exists():
            r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "evaluar_campeon.py"), str(d)], cwd=RAIZ,
                               capture_output=True, text=True)
            anotar(f"evaluación {cid}: " + ("ok" if r.returncode == 0 else r.stderr[-300:]))
    return True


if __name__ == "__main__":
    main()
