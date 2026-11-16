"""Genera una pista de prueba (Spa, Monza, Hungaroring) sin abrir la ventana de Blender.

Uso (desde la carpeta del proyecto):

    blender -b --factory-startup --python blender/generar_pista_prueba.py -- Spa.csv spa "Spa-Francorchamps"

Construye la pista con `construir_pista.construir` (mismos supuestos visuales y
la misma validación que el Hermanos Rodríguez), guarda `blender/<carpeta>.blend`
y exporta `pistas/<carpeta>/pista.json` con `exportar_pista.exportar`.
Este script solo crea la pista. El entorno (gradas, edificios, árboles, anuncios)
se agrega después con `reconstruir_entorno.py`, que no toca `pista.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
import construir_pista  # noqa: E402
import exportar_pista  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(args) != 3:
    raise SystemExit("Uso: ... -- <archivo.csv> <carpeta> <nombre>")
csv, carpeta, nombre = args

informe_c = construir_pista.construir(csv, nombre=nombre)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / f"{carpeta}.blend"))
informe_e = exportar_pista.exportar(f"pistas/{carpeta}/pista.json")
print("INFORME_CONSTRUIR " + json.dumps(informe_c, default=str, ensure_ascii=False))
print("INFORME_EXPORTAR " + json.dumps(informe_e, default=str, ensure_ascii=False))
