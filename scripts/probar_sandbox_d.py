"""Prueba el evaluador aislado de la condición D (fase 6): resultado y aislamiento.

Uso: python scripts/probar_sandbox_d.py
Evalúa el programa inicial (escrito a mano) y tres programas hostiles escritos a mano que
intentan usar la red, escribir en el código montado y ver carpetas personales; los tres deben
fallar dentro del contenedor.
"""
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from evocarros.llm_evo import llm  # noqa: E402
from evocarros.llm_evo.evaluador import evaluar  # noqa: E402

TMP = Path(r"D:\redes neuronales\runs\_prueba_d\tmp") if Path(r"D:\redes neuronales").exists() else RAIZ / "runs" / "_prueba_d" / "tmp"
PIE = "\ndef manejar(r, v, a):\n    return 0.0, 1.0\n"
HOSTILES = {
    "red": "import socket\nsocket.create_connection(('8.8.8.8', 53), timeout=3)" + PIE,
    "escritura en el código": "open('/app/evocarros/intruso.txt', 'w').write('x')" + PIE,
    "carpetas personales": ("import os\nvisto = str(os.listdir('/')) + str(os.listdir('/app'))\n"
                            "assert 'Users' not in visto and 'gersa' not in visto, 've carpetas personales: ' + visto\n"
                            "raise RuntimeError('solo ve: ' + visto)") + PIE,
}

t = time.time()
r = evaluar(llm.PROGRAMA_INICIAL, TMP)
print(f"programa inicial ({time.time() - t:.1f} s): ok={r.get('ok')} etapa={r.get('etapa')} "
      f"aptitud={r.get('aptitud')} motivo={r.get('motivo')} error={r.get('error')}")
print("  retro:", r.get("texto"))
for nombre, cod in HOSTILES.items():
    r = evaluar(cod, TMP)
    print(f"{nombre}: ok={r.get('ok')} -> {r.get('error')}")
