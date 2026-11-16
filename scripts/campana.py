"""Cola de corridas: las ejecuta una tras otra con todos los núcleos (fases 4, 5 y 7).

Uso:
    python scripts/campana.py                      # semilla 1 de las 6 condiciones de red
    python scripts/campana.py --semillas 1-10      # campaña completa de red (60 corridas)

- Salta las corridas con estado "terminada"; reanuda las incompletas desde su checkpoint.
- Al terminar cada corrida, evalúa al campeón (scripts/evaluar_campeon.py).
- Si una corrida queda "detenida" (botón Detener del panel), la cola se detiene.
- Bitácora: <runs>/campana.log.
- Candado <runs>/campana.lock: si ya hay una cola viva, una segunda instancia sale sin
  hacer nada (evita que dos colas entrenen la misma corrida, por ejemplo si el arranque
  automático de Windows se dispara con una cola ya corriendo).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from evocarros.neuro.entrenar import runs_por_defecto  # noqa: E402

CONDICIONES = ["A-GA", "B-GA", "C-GA", "A-ES", "B-ES", "C-ES"]


def semillas(txt: str) -> list[int]:
    out = []
    for parte in txt.split(","):
        a, _, b = parte.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def proceso_vivo(pid: int) -> bool:
    if sys.platform == "win32":   # os.kill(pid, 0) en Windows TERMINA el proceso: no usarlo
        # tras un reinicio el PID viejo puede pertenecer a otro programa: se exige que sea python
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FI", "IMAGENAME eq python*", "/NH"],
                           capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def tomar_candado(ruta: Path) -> bool:
    if ruta.exists():
        try:
            if proceso_vivo(int(ruta.read_text().strip())):
                return False
        except ValueError:
            pass
    ruta.write_text(str(os.getpid()))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", default="1")
    ap.add_argument("--condiciones", default=",".join(CONDICIONES))
    ap.add_argument("--procesos", type=int, default=12)
    ap.add_argument("--runs", type=Path, default=runs_por_defecto())
    a = ap.parse_args()
    a.runs.mkdir(parents=True, exist_ok=True)
    log = a.runs / "campana.log"

    def anotar(t):
        linea = f"{datetime.now().isoformat(timespec='seconds')} {t}"
        print(linea, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(linea + "\n")

    candado = a.runs / "campana.lock"
    if not tomar_candado(candado):
        print("ya hay una cola corriendo; salgo")
        return
    try:
        correr_cola(a, anotar)
    finally:
        candado.unlink(missing_ok=True)


def correr_cola(a, anotar):
    cola = [(s, c) for s in semillas(a.semillas) for c in a.condiciones.split(",")]
    pendientes = []
    if not _pasada(a, anotar, cola, pendientes):
        return
    if pendientes:   # segunda oportunidad: reanudan desde su checkpoint
        anotar(f"reintento de {len(pendientes)} corrida(s) que no terminaron: {', '.join(f'{c}_s{s:02d}' for s, c in pendientes)}")
        _pasada(a, anotar, pendientes, [])
    anotar("campaña terminada")


def _pasada(a, anotar, cola, pendientes) -> bool:
    """Una pasada por la cola. Devuelve False si el usuario detuvo la campaña."""
    anotar(f"campaña: {len(cola)} corridas")
    for s, c in cola:
        cid = f"{c}_s{s:02d}"
        d = a.runs / cid
        est = {}
        if (d / "estado.json").exists():
            est = json.loads((d / "estado.json").read_text(encoding="utf-8"))
        if est.get("estado") != "terminada":
            if (d / "control.json").exists():
                (d / "control.json").unlink()          # una orden vieja de "detener" no bloquea la reanudación
            anotar(f"inicia {cid}")
            r = subprocess.run([sys.executable, "-m", "evocarros.neuro.entrenar", "--condicion", c, "--semilla", str(s),
                                "--procesos", str(a.procesos), "--runs", str(a.runs)], cwd=RAIZ)
            est = json.loads((d / "estado.json").read_text(encoding="utf-8")) if (d / "estado.json").exists() else {}
            anotar(f"fin {cid}: código {r.returncode}, estado {est.get('estado')}, fin {est.get('fin')}, "
                   f"gen {est.get('generacion')}, episodios {est.get('episodios')}, mejor vuelta {est.get('mejor_t_vuelta')}")
            if est.get("estado") == "detenida":
                anotar("campaña detenida desde el panel")
                return False
            if est.get("estado") != "terminada":
                anotar(f"{cid} no terminó (estado {est.get('estado')}); sigo con la siguiente")
                pendientes.append((s, c))
                continue
        if not (d / "eval_final.json").exists():
            r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "evaluar_campeon.py"), str(d)], cwd=RAIZ,
                               capture_output=True, text=True)
            anotar(f"evaluación {cid}: " + (r.stdout.strip().replace("\n", " | ") or r.stderr[-300:]))
    return True


if __name__ == "__main__":
    main()
