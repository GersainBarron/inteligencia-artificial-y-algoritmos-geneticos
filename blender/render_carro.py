"""Render de control del carro (Workbench, sin ventana). Uso: blender -b blender/carro.blend --python blender/render_carro.py"""
from pathlib import Path
import math
import bpy

AQUI = Path(__file__).resolve().parent
esc = bpy.context.scene
esc.render.engine = "BLENDER_WORKBENCH"
esc.display.shading.light = "STUDIO"
esc.display.shading.color_type = "MATERIAL"
esc.render.resolution_x, esc.render.resolution_y = 900, 600
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
esc.collection.objects.link(cam)
esc.camera = cam
cam.location = (4.2, -4.2, 2.6)
cam.rotation_euler = (math.radians(68), 0, math.radians(45))
esc.render.filepath = str(AQUI / "vistas" / "carro.png")
bpy.ops.render.render(write_still=True)
print("RENDER_OK", esc.render.filepath)
