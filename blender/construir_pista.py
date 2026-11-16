"""Construye en Blender una pista a partir de un trazado de TUM racetrack-database.

Se ejecuta DENTRO de Blender (usa `bpy`). Crea dos colecciones:

  Pista_Sim     Geometría que usa el simulador (fuente de verdad de `pista.json`):
                curvas `LineaCentral`, `BordeIzquierdo`, `BordeDerecho` y los
                objetos `CP_000`, `CP_001`, ... (un segmento cada uno, extremo
                izquierdo primero). `exportar_pista.py` lee estos objetos.
  Pista_Visual  Lo que solo se ve: asfalto, líneas de límite de pista, pianos,
                escapatorias, pasto y línea de salida. No influye en el simulador.

Uso desde el editor de texto o la consola de Blender:

    import sys; sys.path.insert(0, r"<carpeta del proyecto>\\blender")
    import construir_pista
    construir_pista.construir("MexicoCity.csv", nombre="Autódromo Hermanos Rodríguez")

Decisiones (ver ESTADO.md):
- Los bordes del simulador son el límite de pista de TUM (línea central ± ancho
  medido). Pianos y escapatorias son decoración: su posición y ancho son
  supuestos visuales, no medidas.
- Checkpoints equiespaciados en longitud de arco: cada uno vale la misma
  distancia, así la aptitud (checkpoints + fracción del tramo) es proporcional
  al avance real y no depende de dónde haya más o menos líneas.
- La pista es plana (z = 0); el simulador es 2D.
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))
import pista_geom as geo  # noqa: E402

COL_SIM, COL_VISUAL = "Pista_Sim", "Pista_Visual"

# Supuestos visuales (no medidos); se pueden ajustar sin tocar el simulador.
VISUAL = {
    "radio_curva_m": 150.0,       # un tramo cuenta como curva si su radio es menor
    "ancho_piano_m": 1.2,
    "largo_bloque_piano_m": 2.5,  # alternancia rojo/blanco (medio segmento de TUM)
    "ancho_linea_m": 0.15,        # línea blanca de límite de pista
    "ancho_escapatoria_m": 18.0,  # máximo, en el exterior de las curvas
    "margen_pasto_m": 32.0,
    "ancho_franja_verde_m": 1.0,  # franja pintada de verde detrás de los pianos
    # tramos (s_ini, s_fin, factor) con escapatoria más amplia, por archivo de TUM
    "escapatorias_amplias": {"MexicoCity.csv": [(880.0, 1060.0, 2.2)]},   # curva 1
}

COLORES = {
    "Asfalto": (0.045, 0.047, 0.052, 1.0),
    "LineaBlanca": (0.90, 0.90, 0.88, 1.0),
    "PianoRojo": (0.72, 0.05, 0.04, 1.0),
    "PianoBlanco": (0.92, 0.92, 0.90, 1.0),
    "Escapatoria": (0.20, 0.20, 0.21, 1.0),
    "Pasto": (0.13, 0.30, 0.09, 1.0),
    "VerdePista": (0.04, 0.40, 0.22, 1.0),
}


# ------------------------------------------------------------ utilidades
def _material(nombre: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(nombre) or bpy.data.materials.new(nombre)
    color = COLORES[nombre]
    mat.diffuse_color = color
    try:
        mat.use_nodes = True
    except Exception:
        pass
    if mat.node_tree:
        # se busca por tipo: el nombre del nodo cambia con el idioma de Blender
        bsdf = next((nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED"), None)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Roughness"].default_value = 0.9 if nombre in ("Pasto", "Escapatoria") else 0.7
    return mat


def _coleccion(nombre: str) -> bpy.types.Collection:
    """Colección vacía con ese nombre (borra la anterior y sus objetos)."""
    vieja = bpy.data.collections.get(nombre)
    if vieja:
        for ob in list(vieja.objects):
            datos = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            if datos is not None and datos.users == 0:
                bpy.data.batch_remove([datos])
        bpy.data.collections.remove(vieja)
    col = bpy.data.collections.new(nombre)
    bpy.context.scene.collection.children.link(col)
    return col


def _objeto(col, nombre, datos):
    ob = bpy.data.objects.new(nombre, datos)
    col.objects.link(ob)
    return ob


def _curva(col, nombre, puntos, z):
    cu = bpy.data.curves.new(nombre, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(puntos) - 1)
    co = np.column_stack([puntos, np.full(len(puntos), z), np.ones(len(puntos))])
    sp.points.foreach_set("co", co.ravel())
    sp.use_cyclic_u = True
    return _objeto(col, nombre, cu)


def _malla(col, nombre, verts, caras, materiales, indices_mat=None):
    """Malla de cuadriláteros; orienta cada cara con la normal hacia +Z."""
    me = bpy.data.meshes.new(nombre)
    v = np.asarray(verts, dtype=float)
    caras_ok = []
    for cara in caras:
        p = v[list(cara), :2]
        x, y = p.T
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        caras_ok.append(tuple(cara) if area >= 0 else tuple(reversed(cara)))
    me.from_pydata(v.tolist(), [], caras_ok)
    for nombre_mat in materiales:
        me.materials.append(_material(nombre_mat))
    if indices_mat is not None:
        me.polygons.foreach_set("material_index", np.asarray(indices_mat, dtype=np.int32))
    me.update()
    return _objeto(col, nombre, me)


def _tira(a, b, z, mascara=None, cerrada=True):
    """Vértices y caras de la tira entre las polilíneas homólogas a y b.
    `mascara[i]` indica si se crea la cara del segmento i -> i+1."""
    n = len(a)
    verts = np.vstack([np.column_stack([a, np.full(n, z)]), np.column_stack([b, np.full(n, z)])])
    caras = []
    for i in range(n if cerrada else n - 1):
        if mascara is not None and not mascara[i]:
            continue
        j = (i + 1) % n
        caras.append((i, j, n + j, n + i))
    return verts, caras


def _dilatar(m: np.ndarray, k: int) -> np.ndarray:
    salida = m.copy()
    for d in range(1, k + 1):
        salida |= np.roll(m, d) | np.roll(m, -d)
    return salida


def _suavizar(x: np.ndarray, k: int) -> np.ndarray:
    m = k // 2
    return np.convolve(np.concatenate([x[-m:], x, x[:m]]), np.ones(k) / k, "valid")


def zonas_de_curva(centro) -> tuple[np.ndarray, np.ndarray]:
    """Máscara de puntos en curva (con ~10 m de margen) y curvatura con signo."""
    k = geo.curvatura(centro)
    return _dilatar(np.abs(k) > 1.0 / VISUAL["radio_curva_m"], 2), k


def ancho_escapatoria(centro, csv: str, izquierda: bool) -> np.ndarray:
    """Ancho de la escapatoria en cada punto, del lado pedido. Va en el exterior
    de cada curva, con rampa suave de entrada y salida."""
    en_curva, k = zonas_de_curva(centro)
    lado = (k < 0) if izquierda else (k >= 0)    # curva a la derecha: el exterior es el lado izquierdo
    peso = _suavizar((en_curva & lado).astype(float), 11)
    ancho = VISUAL["ancho_escapatoria_m"] * peso * peso * (3 - 2 * peso)
    s, _ = geo.abscisa(centro)
    factor = np.ones(len(centro))
    for s0, s1, f in VISUAL["escapatorias_amplias"].get(csv, []):
        factor[(s >= s0) & (s <= s1)] = f
    return ancho * _suavizar(factor, 15)


# ---------------------------------------------------------- construcción
def construir(csv: str = "MexicoCity.csv", nombre: str = "Autódromo Hermanos Rodríguez",
              n_checkpoints: int = 50, limpiar_escena_inicial: bool = True) -> dict:
    ruta_csv = RAIZ / "pistas" / "fuente_tum" / csv
    centro, w_der, w_izq = geo.leer_tum(ruta_csv)
    n = len(centro)
    nor = geo.normales_izq(centro)
    b_izq, b_der = geo.bordes(centro, w_der, w_izq)
    cps = geo.checkpoints(centro, b_izq, b_der, n_checkpoints)
    informe = geo.validar(centro, b_izq, b_der, cps)
    if informe["fallas"]:
        raise RuntimeError(f"El trazado no pasa la validación: {informe['fallas']}")

    if limpiar_escena_inicial:  # objetos de la escena por defecto de Blender
        for nom in ("Cube", "Camera", "Light"):
            ob = bpy.data.objects.get(nom)
            if ob:
                bpy.data.objects.remove(ob, do_unlink=True)

    def desplazada(d):  # línea central desplazada d metros hacia la izquierda
        return centro + nor * np.asarray(d)[:, None]

    # ---- capa del simulador
    sim = _coleccion(COL_SIM)
    _curva(sim, "LineaCentral", centro, 0.03)
    _curva(sim, "BordeIzquierdo", b_izq, 0.03)
    _curva(sim, "BordeDerecho", b_der, 0.03)
    for cp in cps:
        me = bpy.data.meshes.new(f"CP_{cp['indice']:03d}")
        izq, der = cp["izq"] - cp["centro"], cp["der"] - cp["centro"]
        me.from_pydata([(*izq, 0.0), (*der, 0.0)], [(0, 1)], [])
        ob = _objeto(sim, me.name, me)
        ob.location = (*cp["centro"], 0.05)
        ob["s_m"] = float(cp["s_m"])
        ob.show_in_front = True
    sim["nombre"] = nombre
    sim["fuente_csv"] = f"pistas/fuente_tum/{csv}"
    sim["fuente_sha256"] = geo.sha256(ruta_csv)
    sim["n_checkpoints"] = n_checkpoints

    # ---- capa visual
    vis = _coleccion(COL_VISUAL)
    v, c = _tira(b_izq, b_der, 0.0)
    _malla(vis, "Asfalto", v, c, ["Asfalto"])

    a = VISUAL["ancho_linea_m"]
    v1, c1 = _tira(b_izq, desplazada(w_izq - a), 0.004)
    v2, c2 = _tira(desplazada(-w_der + a), b_der, 0.004)
    c2 = [tuple(i + len(v1) for i in cara) for cara in c2]
    _malla(vis, "LineasLimite", np.vstack([v1, v2]), c1 + c2, ["LineaBlanca"])

    en_curva, k = zonas_de_curva(centro)

    # pianos: en ambos lados de cada curva, en bloques rojo/blanco
    p = VISUAL["ancho_piano_m"]
    verts, caras, mats = [], [], []
    for d_int, d_ext in ((w_izq, w_izq + p), (-w_der - p, -w_der)):
        a_, b_ = desplazada(d_int), desplazada(d_ext)
        # se intercala el punto medio de cada segmento: bloques de ~2.5 m
        a2 = np.empty((2 * n, 2)); b2 = np.empty((2 * n, 2))
        a2[0::2], a2[1::2] = a_, 0.5 * (a_ + np.roll(a_, -1, axis=0))
        b2[0::2], b2[1::2] = b_, 0.5 * (b_ + np.roll(b_, -1, axis=0))
        masc = np.repeat(en_curva & np.roll(en_curva, -1), 2)
        vt, ct = _tira(a2, b2, 0.012, masc)
        base = sum(len(x) for x in verts)
        verts.append(vt)
        for cara in ct:
            caras.append(tuple(i + base for i in cara))
            mats.append(cara[0] % 2)
    _malla(vis, "Pianos", np.vstack(verts), caras, ["PianoRojo", "PianoBlanco"], mats)

    # franja verde pintada detrás de cada piano
    fv = VISUAL["ancho_franja_verde_m"]
    verts, caras = [], []
    for d0, d1 in ((w_izq + p, w_izq + p + fv), (-w_der - p - fv, -w_der - p)):
        vt, ct = _tira(desplazada(d0), desplazada(d1), 0.011, en_curva & np.roll(en_curva, -1))
        base = sum(len(x) for x in verts)
        verts.append(vt)
        caras += [tuple(i + base for i in cara) for cara in ct]
    _malla(vis, "FranjaVerde", np.vstack(verts), caras, ["VerdePista"])

    # escapatorias: asfalto claro en el exterior de cada curva
    verts, caras = [], []
    for izquierda, d0, signo in ((True, w_izq, 1.0), (False, -w_der, -1.0)):
        ancho = ancho_escapatoria(centro, csv, izquierda)
        masc = (ancho > 0.3) & (np.roll(ancho, -1) > 0.3)
        vt, ct = _tira(desplazada(d0), desplazada(d0 + signo * ancho), -0.02, masc)
        base = sum(len(x) for x in verts)
        verts.append(vt)
        caras += [tuple(i + base for i in cara) for cara in ct]
    _malla(vis, "Escapatorias", np.vstack(verts), caras, ["Escapatoria"])

    # pasto: franja a ambos lados de la pista (el resto del suelo lo pone construir_entorno.py)
    m = VISUAL["margen_pasto_m"]
    v, c = _tira(desplazada(w_izq + m), desplazada(-w_der - m), -0.05)
    _malla(vis, "Pasto", v, c, ["Pasto"])

    # línea de salida/meta sobre el checkpoint 0
    t0 = geo.tangentes(centro)[0] * 0.3
    izq, der = cps[0]["izq"], cps[0]["der"]
    _malla(vis, "LineaSalida", [(*(izq - t0), 0.008), (*(der - t0), 0.008), (*(der + t0), 0.008), (*(izq + t0), 0.008)],
           [(0, 1, 2, 3)], ["LineaBlanca"])

    informe["n_puntos"] = n
    informe["tramos_curvos"] = int(np.count_nonzero(en_curva & ~np.roll(en_curva, 1)))
    return informe


if __name__ == "__main__":
    print(construir())
