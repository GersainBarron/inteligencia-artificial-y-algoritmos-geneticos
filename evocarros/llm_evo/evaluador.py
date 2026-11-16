"""Evalúa programas piloto en el contenedor aislado (condición D).

Cada evaluación corre en un `docker run` nuevo con: sin red, sistema de archivos de solo
lectura, 1 CPU, 512 MB, 64 procesos, sin capacidades, sin escalar privilegios y usuario sin
privilegios. Solo se montan (de solo lectura) el código del simulador, el script de
evaluación y la pista de entrenamiento, más una carpeta temporal de trabajo por programa.
El código generado nunca se ejecuta fuera del contenedor (`modo="local"` existe solo para
probar el sistema con programas escritos a mano, nunca con código del modelo).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
IMAGEN = "evocarros-sandbox"
LIMITE_S = 120


def evaluar(codigo: str, carpeta_tmp: Path, registrar: bool = False, modo: str = "docker",
            peticion_extra: dict | None = None, todas_las_pistas: bool = False, limite_s: int = LIMITE_S) -> dict:
    """Corre el programa en el contenedor. `todas_las_pistas` monta pistas/ completa (solo para
    la evaluación final del campeón, después de terminar la corrida)."""
    carpeta_tmp.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="prog_", dir=carpeta_tmp))
    try:
        (d / "programa.py").write_text(codigo, encoding="utf-8")
        if modo == "docker":
            pista = "/app/pistas/mexico/pista.json"
        else:
            pista = str(RAIZ / "pistas" / "mexico" / "pista.json")
        pet = {"registrar": registrar, "pista": pista}
        pet.update(peticion_extra or {})
        (d / "peticion.json").write_text(json.dumps(pet), encoding="utf-8")
        if modo == "docker":
            nombre = f"evocarros-prog-{uuid.uuid4().hex[:10]}"
            cmd = ["docker", "run", "--rm", "--name", nombre, "--network", "none", "--read-only",
                   "--cpus", "1", "--memory", "512m", "--pids-limit", "64", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--user", "1000:1000",
                   "--tmpfs", "/tmp:size=16m", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-v", f"{RAIZ / 'evocarros'}:/app/evocarros:ro",
                   "-v", f"{RAIZ / 'sandbox'}:/app/sandbox:ro",
                   "-v", (f"{RAIZ / 'pistas'}:/app/pistas:ro" if todas_las_pistas
                          else f"{RAIZ / 'pistas' / 'mexico'}:/app/pistas/mexico:ro"),
                   "-v", f"{d}:/trabajo",
                   IMAGEN, "python", "/app/sandbox/evaluar_programa.py",
                   "/trabajo/programa.py", "/trabajo/peticion.json", "/trabajo/resultado.json"]
        else:
            nombre = None
            cmd = [sys.executable, str(RAIZ / "sandbox" / "evaluar_programa.py"),
                   str(d / "programa.py"), str(d / "peticion.json"), str(d / "resultado.json")]
        env = dict(os.environ, PYTHONPATH=str(RAIZ), PYTHONDONTWRITEBYTECODE="1")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=limite_s, env=env)
        except subprocess.TimeoutExpired:
            if nombre:
                subprocess.run(["docker", "kill", nombre], capture_output=True)
            return {"ok": False, "etapa": 0, "episodios": 0, "error": "tiempo agotado en el contenedor"}
        res = d / "resultado.json"
        if not res.exists():
            return {"ok": False, "etapa": 0, "episodios": 0,
                    "error": ("el contenedor no produjo resultado: " + (r.stderr or r.stdout)[-400:]).strip()}
        return json.loads(res.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
