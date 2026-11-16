"""Exporta la parte visual de una pista a glTF binario para el visor Three.js.

Uso (sin ventana):

    blender -b blender/hermanos_rodriguez.blend --python blender/exportar_gltf_pista.py -- visor/modelos/hermanos_rodriguez.glb

Se exportan solo las mallas de `Pista_Visual` y de las colecciones `Entorno_*`.
`Pista_Sim` (bordes y checkpoints) no se exporta: el visor dibuja esa capa desde
`pista.json`, que es la fuente de verdad del simulador. Tampoco cámaras ni luces.
"""
import json
import sys
from pathlib import Path

import bpy

RAIZ = Path(__file__).resolve().parent.parent
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
destino = Path(args[0]) if args else RAIZ / "visor" / "modelos" / "pista.glb"
if not destino.is_absolute():
    destino = RAIZ / destino
destino.parent.mkdir(parents=True, exist_ok=True)

cols = [c for c in bpy.data.collections if c.name == "Pista_Visual" or c.name.startswith("Entorno_")]
obs = {o for c in cols for o in c.all_objects if o.type == "MESH"}
bpy.ops.object.select_all(action="DESELECT")
for o in obs:
    o.hide_set(False)
    o.select_set(True)
bpy.context.view_layer.update()
bpy.ops.export_scene.gltf(filepath=str(destino), export_format="GLB", use_selection=True,
                          export_apply=True, export_cameras=False, export_lights=False)
caras = sum(len(o.data.polygons) for o in obs)
info = {"archivo": str(destino), "colecciones": sorted(c.name for c in cols),
        "objetos": len(obs), "caras": caras, "bytes": destino.stat().st_size}
print("INFORME_GLTF " + json.dumps(info, ensure_ascii=False))
