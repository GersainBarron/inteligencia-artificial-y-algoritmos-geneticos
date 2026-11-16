"""Renderiza las vistas de presentación del modelo (se ejecuta DENTRO de Blender).

    import render_vistas; render_vistas.render()            # todas las cámaras Cam_*
    render_vistas.render(["Cam_Recta"], motor="BLENDER_WORKBENCH")

Las imágenes quedan en `blender/vistas/` con el nombre de la cámara (con `prefijo`
delante, para no mezclar circuitos). Las cámaras
cenitales (ortográficas) se renderizan con Workbench (colores planos, para
control); las demás con EEVEE.
"""
from pathlib import Path

import bpy

VISTAS = Path(__file__).resolve().parent / "vistas"


def render(camaras=None, ancho=1600, alto=900, motor=None, prefijo="") -> list[str]:
    sc = bpy.context.scene
    VISTAS.mkdir(exist_ok=True)
    hechas = []
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        if ob.type != "CAMERA" or not ob.name.startswith("Cam_") or (camaras and ob.name not in camaras):
            continue
        orto = ob.data.type == "ORTHO"
        sc.render.engine = motor or ("BLENDER_WORKBENCH" if orto else "BLENDER_EEVEE")
        if sc.render.engine == "BLENDER_WORKBENCH":
            sc.display.shading.light = "FLAT"
            sc.display.shading.color_type = "MATERIAL"
        sc.render.resolution_x, sc.render.resolution_y = (1800, 1250) if orto else (ancho, alto)
        sc.render.image_settings.file_format = "PNG"
        sc.camera = ob
        sc.render.filepath = str(VISTAS / f"{prefijo}{ob.name[4:].lower()}.png")
        bpy.ops.render.render(write_still=True)
        hechas.append(sc.render.filepath)
    sc.render.filepath = "//vistas/"
    return hechas
