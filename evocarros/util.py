"""Utilidades compartidas."""
from __future__ import annotations

import os
import time
from pathlib import Path


def reemplazar(tmp: Path, destino: Path, intentos: int = 30) -> None:
    """os.replace con reintentos: en Windows falla con "Acceso denegado" (WinError 5) si otro
    proceso (el panel o el guardián) está leyendo el destino justo en ese instante."""
    for k in range(intentos):
        try:
            os.replace(tmp, destino)
            return
        except PermissionError:
            if k == intentos - 1:
                raise
            time.sleep(0.05 * (k + 1))
