"""Carro low-poly de monoplaza genérico para el visor Three.js (no es réplica de ningún equipo).

Uso (sin ventana, desde la carpeta del proyecto):

    blender -b --factory-startup --python blender/construir_carro.py

Escribe `blender/carro.blend` y `visor/modelos/carro.glb`.

Convenciones (las mismas que usará el simulador):
- Unidades en metros; el frente del carro apunta a +X, la izquierda a +Y, arriba +Z
  (en glTF, Y arriba: el exportador convierte).
- Origen en el centro del rectángulo de choque, a nivel del suelo.
- Rectángulo de choque = LARGO x ANCHO (ver abajo). El simulador usa estas medidas,
  el modelo visual cabe dentro de él.
- El material `Carroceria` es blanco a propósito: el visor lo tiñe por carro
  (color por condición o por linaje).
"""
from __future__ import annotations

import json
from pathlib import Path

import bpy

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

# Supuesto de diseño: monoplaza "acotado", más chico que un F1 real (5.6 x 2.0 m).
LARGO, ANCHO = 4.0, 1.8
R_RUEDA, A_RUEDA = 0.33, 0.30

COLORES = {
    "Carroceria": (0.95, 0.95, 0.95, 1.0),
    "Llanta": (0.03, 0.03, 0.035, 1.0),
    "Detalle": (0.08, 0.08, 0.09, 1.0),
    "Casco": (0.95, 0.75, 0.10, 1.0),
}


def material(nombre):
    m = bpy.data.materials.get(nombre) or bpy.data.materials.new(nombre)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = COLORES[nombre]
    bsdf.inputs["Roughness"].default_value = 0.5
    return m


def caja(nombre, centro, medidas, mat, padre):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=centro)
    ob = bpy.context.active_object
    ob.name = nombre
    ob.scale = medidas
    bpy.ops.object.transform_apply(scale=True)
    ob.data.materials.append(material(mat))
    ob.parent = padre
    return ob


def rueda(nombre, x, y, padre):
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=R_RUEDA, depth=A_RUEDA,
                                        location=(x, y, R_RUEDA), rotation=(1.5708, 0, 0))
    ob = bpy.context.active_object
    ob.name = nombre
    ob.data.materials.append(material("Llanta"))
    ob.parent = padre
    return ob


def construir():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)

    raiz = bpy.data.objects.new("Carro", None)
    bpy.context.scene.collection.objects.link(raiz)

    xr, xd = -LARGO / 2 + 0.55, LARGO / 2 - 0.85      # ejes trasero y delantero
    yr = ANCHO / 2 - A_RUEDA / 2                         # ruedas al borde del rectángulo
    # chasis: nariz, monocasco, pontones y motor
    caja("Nariz", (LARGO / 2 - 0.45, 0, 0.30), (0.9, 0.28, 0.18), "Carroceria", raiz)
    caja("Monocasco", (0.15, 0, 0.36), (1.9, 0.62, 0.34), "Carroceria", raiz)
    caja("Pontones", (-0.2, 0, 0.30), (1.1, 1.10, 0.26), "Carroceria", raiz)
    caja("Motor", (-1.05, 0, 0.42), (0.9, 0.50, 0.38), "Carroceria", raiz)
    caja("Cockpit", (0.25, 0, 0.55), (0.55, 0.40, 0.06), "Detalle", raiz)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=6, radius=0.15, location=(0.15, 0, 0.66))
    casco = bpy.context.active_object
    casco.name = "Casco"; casco.data.materials.append(material("Casco")); casco.parent = raiz
    # alerones
    caja("AleronDelantero", (LARGO / 2 - 0.15, 0, 0.12), (0.30, ANCHO - 0.05, 0.04), "Carroceria", raiz)
    caja("AleronTrasero", (-LARGO / 2 + 0.12, 0, 0.85), (0.24, 1.10, 0.05), "Carroceria", raiz)
    caja("SoporteAleron", (-LARGO / 2 + 0.25, 0, 0.62), (0.10, 0.06, 0.42), "Detalle", raiz)
    for i, (x, y) in enumerate([(xd, yr), (xd, -yr), (xr, yr), (xr, -yr)]):
        rueda(f"Rueda_{i}", x, y, raiz)

    bpy.context.view_layer.update()
    caras = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
    # verificación: todo el modelo cabe en el rectángulo de choque
    xs, ys = [], []
    for o in bpy.data.objects:
        if o.type == "MESH":
            for v in o.data.vertices:
                w = o.matrix_world @ v.co
                xs.append(w.x); ys.append(w.y)
    tol = 1e-4  # 0.1 mm: Blender guarda vértices en float32
    ajusta = (min(xs) >= -LARGO / 2 - tol and max(xs) <= LARGO / 2 + tol
              and min(ys) >= -ANCHO / 2 - tol and max(ys) <= ANCHO / 2 + tol)
    if not ajusta:
        raise RuntimeError(f"El modelo se sale del rectángulo de choque: x {min(xs):.3f}..{max(xs):.3f}, y {min(ys):.3f}..{max(ys):.3f}")

    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "carro.blend"))
    destino = RAIZ / "visor" / "modelos" / "carro.glb"
    destino.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(destino), export_format="GLB", export_apply=True)
    meta = {"largo_m": LARGO, "ancho_m": ANCHO, "frente": "+X en Blender y en glTF (Y de Blender pasa a -Z en glTF; Z arriba pasa a Y arriba)",
            "origen": "centro del rectángulo de choque, a nivel del suelo",
            "caras": caras, "bytes": destino.stat().st_size}
    (RAIZ / "visor" / "modelos" / "carro.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


if __name__ == "__main__":
    print("INFORME_CARRO " + json.dumps(construir(), ensure_ascii=False))
