"""Guardián: vigila la laptop en paralelo y pausa las corridas si se pasan los límites.

Uso: python scripts/guardian.py           (lo lanza arranque_evocarros.ps1; corre siempre)

Cada 15 s mide:
- GPU (nvidia-smi): pausa si pasa de 85 °C durante 60 s seguidos; reanuda bajo 75 °C.
- CPU: si hay un monitor de sensores con WMI (LibreHardwareMonitor u OpenHardwareMonitor
  abiertos), pausa sobre 95 °C sostenidos y reanuda bajo 85 °C. Sin él, Windows no deja leer la
  temperatura del CPU sin permisos de administrador y se registra como "no disponible".
- Disco donde están las corridas: pausa con menos de 20 GB libres; reanuda con más de 25 GB.
- Corridas colgadas: avisa si una corrida dice "corriendo" pero su estado.json no se
  actualiza en 20 minutos.
La histéresis (pausar a 85, reanudar a 75) evita que prenda y apague cada minuto.

Cómo pausa: escribe control.json {"orden": "pausar", "por": "guardian"} en cada corrida que
esté corriendo (redes y condición D). El entrenador lo lee al terminar su generación, guarda
y espera. Al reanudar, solo devuelve a "correr" las corridas que él mismo pausó: una pausa
manual desde el panel se respeta.

Estado en <runs>/guardian.json (lo muestra el panel) y bitácora en <runs>/guardian.log.
Candado <runs>/guardian.lock para que nunca haya dos guardianes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
from campana import tomar_candado  # noqa: E402
from evocarros.util import reemplazar  # noqa: E402
from evocarros.neuro.entrenar import runs_por_defecto  # noqa: E402

INTERVALO_S = 15
GPU_PAUSA, GPU_REANUDA, GPU_SOSTENIDO_S = 85.0, 75.0, 60
CPU_PAUSA, CPU_REANUDA = 95.0, 85.0
DISCO_PAUSA_GB, DISCO_REANUDA_GB = 20.0, 25.0
COLGADA_MIN = 20


def temp_gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def temp_cpu():
    """Temperatura máxima de los sensores de CPU vía WMI de LibreHardwareMonitor/OpenHardwareMonitor."""
    if sys.platform != "win32":
        return None
    for ns in ("root/LibreHardwareMonitor", "root/OpenHardwareMonitor"):
        cmd = (f"Get-CimInstance -Namespace {ns} -ClassName Sensor -ErrorAction Stop | "
               "Where-Object { $_.SensorType -eq 'Temperature' -and $_.Identifier -match 'cpu' } | "
               "Measure-Object -Property Value -Maximum | Select-Object -ExpandProperty Maximum")
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=20)
            v = r.stdout.strip()
            if r.returncode == 0 and v:
                return float(v.replace(",", "."))
        except Exception:
            pass
    return None


def leer_json(ruta: Path) -> dict:
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def escribir_json(ruta: Path, datos: dict):
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    reemplazar(tmp, ruta)


class Guardian:
    def __init__(self, runs: Path):
        self.runs = runs
        self.pausado_por = None          # motivo activo, o None
        self.gpu_alta_desde = None
        self.log = runs / "guardian.log"

    def anotar(self, t: str):
        linea = f"{datetime.now().isoformat(timespec='seconds')} {t}"
        print(linea, flush=True)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(linea + "\n")

    def corridas(self):
        for d in self.runs.iterdir():
            if d.is_dir() and not d.name.startswith("_") and (d / "estado.json").exists():
                yield d, leer_json(d / "estado.json")

    def pausar_todas(self, motivo: str):
        n = 0
        for d, est in self.corridas():
            ctrl = leer_json(d / "control.json")
            if est.get("estado") == "corriendo" and ctrl.get("orden", "correr") == "correr":
                escribir_json(d / "control.json", {"orden": "pausar", "por": "guardian", "motivo": motivo,
                                                   "fecha": datetime.now().isoformat(timespec="seconds")})
                n += 1
        self.anotar(f"PAUSA ({motivo}): {n} corrida(s)")

    def reanudar_las_mias(self):
        n = 0
        for d, _ in self.corridas():
            ctrl = leer_json(d / "control.json")
            if ctrl.get("orden") == "pausar" and ctrl.get("por") == "guardian":
                escribir_json(d / "control.json", {"orden": "correr", "por": "guardian",
                                                   "fecha": datetime.now().isoformat(timespec="seconds")})
                n += 1
        self.anotar(f"REANUDA: {n} corrida(s)")

    def ciclo(self):
        ahora = time.time()
        gpu, cpu = temp_gpu(), temp_cpu()
        libre = shutil.disk_usage(self.runs).free / 1e9
        motivo = None
        if gpu is not None and gpu > GPU_PAUSA:
            self.gpu_alta_desde = self.gpu_alta_desde or ahora
            if ahora - self.gpu_alta_desde >= GPU_SOSTENIDO_S:
                motivo = f"GPU a {gpu:.0f} °C"
        else:
            self.gpu_alta_desde = None
        if cpu is not None and cpu > CPU_PAUSA:
            motivo = motivo or f"CPU a {cpu:.0f} °C"
        if libre < DISCO_PAUSA_GB:
            motivo = motivo or f"solo {libre:.1f} GB libres"
        if motivo and not self.pausado_por:
            self.pausado_por = motivo
            self.pausar_todas(motivo)
        elif self.pausado_por and not motivo:
            fresco = ((gpu is None or gpu < GPU_REANUDA) and (cpu is None or cpu < CPU_REANUDA)
                      and libre > DISCO_REANUDA_GB)
            if fresco:
                self.pausado_por = None
                self.reanudar_las_mias()
        colgadas = []
        for d, est in self.corridas():
            if est.get("estado") == "corriendo" and (ahora - (d / "estado.json").stat().st_mtime) > COLGADA_MIN * 60:
                colgadas.append(d.name)
        escribir_json(self.runs / "guardian.json", {
            "activo": True, "actualizado": datetime.now().isoformat(timespec="seconds"),
            "gpu_c": gpu, "cpu_c": cpu, "cpu_disponible": cpu is not None, "disco_libre_gb": round(libre, 1),
            "pausado_por": self.pausado_por, "colgadas": colgadas,
            "limites": {"gpu_pausa": GPU_PAUSA, "gpu_reanuda": GPU_REANUDA, "cpu_pausa": CPU_PAUSA,
                        "cpu_reanuda": CPU_REANUDA, "disco_pausa_gb": DISCO_PAUSA_GB}})
        return colgadas


def main():
    runs = runs_por_defecto()
    runs.mkdir(parents=True, exist_ok=True)
    if not tomar_candado(runs / "guardian.lock"):
        print("ya hay un guardián corriendo; salgo")
        return
    g = Guardian(runs)
    g.anotar(f"guardián iniciado (GPU pausa > {GPU_PAUSA} °C sostenidos {GPU_SOSTENIDO_S} s, reanuda < {GPU_REANUDA}; "
             f"CPU {'medible' if temp_cpu() is not None else 'no disponible sin monitor de sensores'}; "
             f"disco pausa < {DISCO_PAUSA_GB} GB)")
    avisadas = set()
    try:
        while True:
            try:
                colg = set(g.ciclo())
                for c in colg - avisadas:
                    g.anotar(f"AVISO: {c} no actualiza su estado desde hace más de {COLGADA_MIN} min")
                avisadas = colg
            except Exception as e:  # el guardián nunca debe morir por un error de lectura
                g.anotar(f"error en el ciclo: {e}")
            time.sleep(INTERVALO_S)
    finally:
        (runs / "guardian.lock").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
