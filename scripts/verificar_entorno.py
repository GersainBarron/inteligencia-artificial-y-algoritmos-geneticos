"""Verifica que el entorno de evocarros esté listo en la AORUS.

Uso (desde la carpeta del proyecto):
    conda activate evocarros
    python scripts/verificar_entorno.py

Revisa: Python y librerías, ffmpeg, carpetas de datos en D:, espacio libre,
GPU, Ollama con los modelos y su velocidad, y el contenedor aislado de Docker.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DATOS = Path(r"D:\redes neuronales")
OLLAMA = "http://localhost:11434"
MODELOS = ["qwen2.5-coder:7b", "qwen3.5:9b"]
IMAGEN_SANDBOX = "evocarros-sandbox"

resultados: list[tuple[str, bool, str]] = []


def registrar(nombre: str, ok: bool, detalle: str = "") -> None:
    resultados.append((nombre, ok, detalle))
    marca = "OK " if ok else "FALLA"
    print(f"[{marca}] {nombre}{': ' + detalle if detalle else ''}")


def correr(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def verificar_python() -> None:
    v = sys.version_info
    registrar("Python 3.12", (v.major, v.minor) == (3, 12), sys.version.split()[0])
    for modulo in ["numpy", "scipy", "matplotlib", "pandas", "yaml", "requests", "imageio"]:
        try:
            m = __import__(modulo)
            registrar(f"librería {modulo}", True, getattr(m, "__version__", ""))
        except ImportError as e:
            registrar(f"librería {modulo}", False, str(e))


def verificar_ffmpeg() -> None:
    ruta = shutil.which("ffmpeg")
    registrar("ffmpeg", ruta is not None, ruta or "no encontrado en PATH")


def verificar_datos() -> None:
    for sub in ["runs", "ollama-models"]:
        p = DATOS / sub
        registrar(f"carpeta {p}", p.is_dir())
    if DATOS.exists():
        libre_gb = shutil.disk_usage(DATOS).free / 1e9
        registrar("espacio libre en D: (>= 20 GB)", libre_gb >= 20, f"{libre_gb:.0f} GB")


def verificar_gpu() -> None:
    try:
        r = correr(
            ["nvidia-smi", "--query-gpu=name,memory.total,temperature.gpu", "--format=csv,noheader"]
        )
        registrar("GPU NVIDIA", r.returncode == 0, r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        registrar("GPU NVIDIA", False, str(e))


def ollama_api(ruta: str, datos: dict | None = None, timeout: int = 600) -> dict:
    cuerpo = json.dumps(datos).encode() if datos is not None else None
    req = urllib.request.Request(
        OLLAMA + ruta, data=cuerpo, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def verificar_ollama() -> None:
    try:
        version = ollama_api("/api/version")["version"]
        registrar("servidor Ollama", True, version)
    except OSError as e:
        registrar("servidor Ollama", False, f"no responde en {OLLAMA}: {e}")
        return

    instalados = {m["name"] for m in ollama_api("/api/tags")["models"]}
    for modelo in MODELOS:
        registrar(f"modelo {modelo}", modelo in instalados)

    # Velocidad del modelo principal (dos llamadas: la primera calienta la GPU).
    prompt = "Escribe una funcion de Python que sume dos numeros. Solo codigo."
    for intento in (1, 2):
        t0 = time.time()
        r = ollama_api(
            "/api/generate",
            {"model": MODELOS[0], "prompt": prompt, "stream": False, "options": {"num_ctx": 8192}},
        )
        if intento == 2:
            tok_s = r["eval_count"] / (r["eval_duration"] / 1e9)
            registrar(
                f"velocidad {MODELOS[0]} (>= 30 tok/s)",
                tok_s >= 30,
                f"{tok_s:.1f} tok/s, {time.time() - t0:.1f} s total",
            )


def verificar_docker() -> None:
    if shutil.which("docker") is None:
        registrar("docker", False, "no encontrado")
        return
    r = correr(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=60)
    registrar("servidor Docker", r.returncode == 0, r.stdout.strip() or r.stderr.strip()[:200])
    if r.returncode != 0:
        return

    sandbox = RAIZ_PROYECTO / "sandbox"
    r = correr(["docker", "build", "-q", "-t", IMAGEN_SANDBOX, str(sandbox)], timeout=900)
    registrar("imagen del contenedor aislado", r.returncode == 0, r.stderr.strip()[-200:])
    if r.returncode != 0:
        return

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--cpus", "1", "--memory", "512m", "--pids-limit", "64",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "-v", f"{sandbox / 'prueba_aislamiento.py'}:/work/prueba.py:ro",
        IMAGEN_SANDBOX, "python", "-I", "/work/prueba.py",
    ]
    r = correr(cmd, timeout=120)
    try:
        datos = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        registrar("ejecución aislada", False, (r.stdout + r.stderr).strip()[-300:])
        return
    registrar("contenedor sin red", datos.get("red_bloqueada") is True)
    registrar("contenedor de solo lectura", datos.get("escritura_bloqueada") is True)
    registrar("contenedor sin root", datos.get("usuario_no_root") is True)
    registrar("numpy en el contenedor", bool(datos.get("numpy")), str(datos.get("numpy")))


def main() -> int:
    print(f"Proyecto: {RAIZ_PROYECTO}\n")
    for paso in (verificar_python, verificar_ffmpeg, verificar_datos, verificar_gpu,
                 verificar_ollama, verificar_docker):
        try:
            paso()
        except Exception as e:  # noqa: BLE001 - queremos ver cualquier fallo y seguir
            registrar(paso.__name__, False, f"{type(e).__name__}: {e}")
    fallas = [n for n, ok, _ in resultados if not ok]
    print(f"\n{len(resultados) - len(fallas)}/{len(resultados)} verificaciones correctas.")
    if fallas:
        print("Fallaron: " + ", ".join(fallas))
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
