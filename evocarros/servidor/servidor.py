"""Servidor local del panel y el visor: http://localhost:8080 (solo esta máquina).

Uso:  python -m evocarros.servidor.servidor [--puerto 8080] [--runs "D:\\redes neuronales\\runs"]

Rutas:
  /                         panel (visor/panel.html)
  /visor                    visor 3D (visor/index.html)
  /<archivo>                archivos estáticos de visor/ (js, css, modelos/*.glb)
  /pistas/<nombre>/pista.json
  /runs/<corrida>/replays/<gen>.json.gz   (el navegador lo descomprime)
  GET  /api/corridas        estado.json de cada corrida + últimas filas de fitness.csv
  GET  /api/corridas/<id>/fitness         fitness.csv completo como JSON
  GET  /api/corridas/<id>/replays         lista de repeticiones disponibles
  GET  /api/sistema         temperatura de GPU (nvidia-smi) y disco libre de la carpeta de runs
  POST /api/control         {"corrida": "<id>" | "todas", "orden": "correr" | "pausar" | "detener"}

Contrato con el entrenador (fase 3): cada corrida vive en <runs>/<id>/. El
entrenador escribe estado.json y fitness.csv; el panel solo escribe control.json
({"orden": ..., "fecha": ...}). El entrenador lo lee al terminar cada generación,
así que pausar o detener funciona aunque el navegador o este servidor fallen.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from evocarros.util import reemplazar
RAIZ = Path(__file__).resolve().parents[2]
VISOR = RAIZ / "visor"
PISTAS = RAIZ / "pistas"
ORDENES = {"correr", "pausar", "detener"}


def runs_por_defecto() -> Path:
    if os.environ.get("EVOCARROS_RUNS"):
        return Path(os.environ["EVOCARROS_RUNS"])
    d = Path(r"D:\redes neuronales\runs")
    return d if d.exists() else RAIZ / "runs"


def _dentro(base: Path, rel: str) -> Path | None:
    """Ruta segura bajo `base` (evita ../)."""
    p = (base / rel).resolve()
    return p if p.is_relative_to(base.resolve()) and p.is_file() else None


def leer_fitness(ruta: Path, ultimas: int | None = None) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    out = []
    for r in filas[-ultimas:] if ultimas else filas:
        out.append({k: (float(v) if v not in ("", None) else None) for k, v in r.items()})
    return out


def temp_gpu() -> float | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Manejador(SimpleHTTPRequestHandler):
    runs: Path = runs_por_defecto()

    def log_message(self, fmt, *args):  # silencioso salvo errores
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------ utilidades
    def _json(self, obj, codigo=HTTPStatus.OK):
        datos = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def _archivo(self, ruta: Path | None):
        if ruta is None:
            return self.send_error(HTTPStatus.NOT_FOUND)
        tipos = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
                 ".glb": "model/gltf-binary", ".gz": "application/octet-stream", ".png": "image/png"}
        datos = ruta.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", tipos.get(ruta.suffix, "application/octet-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def _corridas(self) -> list[dict]:
        out = []
        if not self.runs.exists():
            return out
        for d in sorted(p for p in self.runs.iterdir() if p.is_dir()):
            est = {}
            if (d / "estado.json").exists():
                try:
                    est = json.loads((d / "estado.json").read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    est = {"estado": "estado.json ilegible"}
            ctrl = {}
            if (d / "control.json").exists():
                try:
                    ctrl = json.loads((d / "control.json").read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            out.append({"id": d.name, **est, "orden": ctrl.get("orden", "correr"),
                        "fitness": leer_fitness(d / "fitness.csv", ultimas=300),
                        "n_replays": len(list((d / "replays").glob("gen_*.json*"))) if (d / "replays").exists() else 0})
        return out

    # ------------------------------------------------------------ GET
    def do_GET(self):
        ruta = unquote(urlparse(self.path).path)
        if ruta in ("/", "/panel"):
            return self._archivo(VISOR / "panel.html")
        if ruta == "/visor":
            return self._archivo(VISOR / "index.html")
        if ruta == "/api/corridas":
            return self._json(self._corridas())
        if ruta == "/api/sistema":
            libre = shutil.disk_usage(self.runs if self.runs.exists() else RAIZ).free / 1e9
            guardian = {}
            try:
                guardian = json.loads((self.runs / "guardian.json").read_text(encoding="utf-8"))
                edad = datetime.now().timestamp() - (self.runs / "guardian.json").stat().st_mtime
                guardian["activo"] = edad < 120          # sin latido en 2 min = guardián caído
            except (OSError, json.JSONDecodeError):
                guardian = {"activo": False}
            return self._json({"gpu_c": temp_gpu(), "disco_libre_gb": round(libre, 1), "runs": str(self.runs),
                               "hora": datetime.now().isoformat(timespec="seconds"), "guardian": guardian})
        partes = ruta.strip("/").split("/")
        if len(partes) == 4 and partes[:2] == ["api", "corridas"]:
            d = (self.runs / partes[2]).resolve()
            if not d.is_relative_to(self.runs.resolve()) or not d.is_dir():
                return self.send_error(HTTPStatus.NOT_FOUND)
            if partes[3] == "fitness":
                return self._json(leer_fitness(d / "fitness.csv"))
            if partes[3] == "replays":
                gens = sorted(p.name for p in (d / "replays").glob("gen_*.json*")) if (d / "replays").exists() else []
                return self._json(gens)
        if ruta.startswith("/pistas/"):
            return self._archivo(_dentro(PISTAS, ruta[len("/pistas/"):]))
        if ruta.startswith("/runs/"):
            return self._archivo(_dentro(self.runs, ruta[len("/runs/"):]))
        return self._archivo(_dentro(VISOR, ruta.lstrip("/")))

    # ------------------------------------------------------------ POST
    def do_POST(self):
        if urlparse(self.path).path != "/api/control":
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            n = int(self.headers.get("Content-Length", 0))
            pedido = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "JSON inválido"}, HTTPStatus.BAD_REQUEST)
        orden, cual = pedido.get("orden"), pedido.get("corrida")
        if orden not in ORDENES or not cual:
            return self._json({"error": f"orden debe ser una de {sorted(ORDENES)} y corrida obligatoria"},
                              HTTPStatus.BAD_REQUEST)
        destinos = [p for p in self.runs.iterdir() if p.is_dir()] if cual == "todas" else [self.runs / cual]
        hechos = []
        for d in destinos:
            d = d.resolve()
            if not d.is_relative_to(self.runs.resolve()) or not d.is_dir():
                continue
            tmp = d / "control.json.tmp"
            tmp.write_text(json.dumps({"orden": orden, "fecha": datetime.now().isoformat(timespec="seconds")}),
                           encoding="utf-8")
            reemplazar(tmp, d / "control.json")   # escritura atómica
            hechos.append(d.name)
        return self._json({"ok": True, "orden": orden, "corridas": hechos})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--puerto", type=int, default=8080)
    ap.add_argument("--runs", type=Path, default=runs_por_defecto())
    a = ap.parse_args()
    Manejador.runs = a.runs
    a.runs.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.puerto), Manejador)
    print(f"Panel: http://localhost:{a.puerto}/   Visor: http://localhost:{a.puerto}/visor   runs: {a.runs}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
