"""Reconstruye el entorno de un circuito sin abrir la ventana de Blender.

    blender -b blender/hermanos_rodriguez.blend --python blender/reconstruir_entorno.py -- [Cam_A Cam_B ...]
    blender -b blender/spa.blend --python blender/reconstruir_entorno.py -- Spa.csv spa [Cam_A ...]

Vuelve a crear las colecciones `Entorno_*` (no toca `Pista_Sim` ni `pista.json`),
renderiza las cámaras pedidas (todas si no se indica ninguna) y guarda el .blend.
Con un circuito distinto de México, las vistas llevan su nombre como prefijo.
Después, para el visor: `exportar_gltf_pista.py` (ver su encabezado).
"""
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import construir_entorno  # noqa: E402
import render_vistas  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
csv, carpeta = "MexicoCity.csv", "mexico"
if args and args[0].lower().endswith(".csv"):
    csv, carpeta, args = args[0], args[1], args[2:]
inf = construir_entorno.construir(csv, f"pistas/{carpeta}/entorno.json")
bpy.context.view_layer.update()
hechas = render_vistas.render(args or None, prefijo="" if carpeta == "mexico" else f"{carpeta}_")
bpy.ops.wm.save_mainfile()
resumen = {k: v for k, v in inf.items() if k != "tribunas"}
resumen["tribunas"] = len(inf.get("tribunas", []))
resumen["caras"] = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
resumen["vistas"] = [Path(h).name for h in hechas]
print("INFORME_ENTORNO " + json.dumps(resumen, ensure_ascii=False, default=str))
