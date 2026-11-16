"""Construye en Blender el entorno de un circuito (solo visual).

Se ejecuta DENTRO de Blender, después de `construir_pista.py`. Lee
`pistas/<circuito>/entorno.json` (OpenStreetMap alineado con la pista) y el
trazado de TUM, y crea las colecciones `Entorno_*`. Nació para el Autódromo
Hermanos Rodríguez, que conserva sus piezas propias (estadio, Palacio de los
Deportes, zonas de boletos); para Spa, Monza y Hungaroring aplica reglas
generales, descritas en `construir`:

  Entorno_Tribunas    gradas escalonadas con asientos por zona, techos y palcos
  Entorno_Pits        edificio de pits (33 fosos, palcos del Paddock Club, torre
                      de control), pit lane, muro de pits, parrilla y pórtico de salida
  Entorno_Estadios    Estadio GNP Seguros (podio), Alfredo Harp Helú (Tridente),
                      Palacio de los Deportes, velódromos, Palillo
  Entorno_Ciudad      edificios cercanos como volúmenes
  Entorno_Vegetacion  árboles low-poly
  Entorno_Suelo       suelo urbano, áreas verdes, canchas, estacionamientos, agua
  Entorno_Barreras    muros y mallas de contención
  Entorno_Detalles    torres de luz, casetas de comisarios, puentes publicitarios, carpas
  Entorno_Gente       grupos de aficionados con pancartas detrás del muro y mecánicos en los pits
  Entorno_Aire        dirigible y helicóptero (el visor los hace volar), globos y nubes
  Entorno_Luz         sol y cámaras de presentación

Nada de esto entra al simulador: `pista.json` sale solo de `Pista_Sim`.

Qué es dato y qué es supuesto (ver ESTADO.md):
- DATO: la huella en planta de cada tribuna, edificio, estadio, cancha y la
  posición de cada árbol (OpenStreetMap, error de alineación RMS 2.8 m).
- SUPUESTO: todo lo vertical. OSM casi no trae alturas, así que las gradas
  suben con una pendiente típica (0.9 m cada 1.7 m), y las alturas de techos,
  palcos, torre de control, cúpula y Tridente son estimaciones visuales.
  Los colores de asientos siguen las zonas del mapa oficial de tribunas
  (REFERENCIAS_BLENDER.md); no se copiaron fotos ni texturas.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))
import pista_geom as geo  # noqa: E402
from construir_pista import _coleccion, _objeto  # noqa: E402

# nombre: (color RGB, rugosidad, metálico, alfa)
MATERIALES = {
    "Concreto": ((0.50, 0.50, 0.48), 0.9, 0.0, 1.0),
    "ConcretoOscuro": ((0.30, 0.30, 0.30), 0.9, 0.0, 1.0),
    "Acero": ((0.62, 0.64, 0.67), 0.45, 0.6, 1.0),
    "TechoBlanco": ((0.88, 0.88, 0.86), 0.6, 0.0, 1.0),
    "Vidrio": ((0.16, 0.30, 0.40), 0.08, 0.3, 1.0),
    "Porton": ((0.07, 0.07, 0.08), 0.6, 0.0, 1.0),
    "MuroPits": ((0.78, 0.78, 0.76), 0.7, 0.0, 1.0),
    "Malla": ((0.62, 0.64, 0.67), 0.6, 0.3, 0.22),
    "AsfaltoPits": ((0.085, 0.087, 0.092), 0.8, 0.0, 1.0),
    "LineaBlanca": ((0.90, 0.90, 0.88), 0.7, 0.0, 1.0),
    "AsientoVerde": ((0.05, 0.42, 0.16), 0.7, 0.0, 1.0),
    "AsientoNaranja": ((0.85, 0.36, 0.04), 0.7, 0.0, 1.0),
    "AsientoAzul": ((0.05, 0.20, 0.60), 0.7, 0.0, 1.0),
    "AsientoAmarillo": ((0.85, 0.66, 0.05), 0.7, 0.0, 1.0),
    "AsientoGris": ((0.28, 0.34, 0.44), 0.7, 0.0, 1.0),
    "AsientoCafe": ((0.36, 0.20, 0.09), 0.7, 0.0, 1.0),
    "AsientoRojo": ((0.60, 0.05, 0.06), 0.7, 0.0, 1.0),
    "AsientoNeutro": ((0.30, 0.38, 0.48), 0.7, 0.0, 1.0),
    "Cobre": ((0.42, 0.25, 0.13), 0.35, 0.9, 1.0),
    "Ladrillo": ((0.45, 0.22, 0.14), 0.9, 0.0, 1.0),
    "Fachada1": ((0.74, 0.70, 0.62), 0.9, 0.0, 1.0),
    "Fachada2": ((0.70, 0.55, 0.47), 0.9, 0.0, 1.0),
    "Fachada3": ((0.60, 0.62, 0.64), 0.9, 0.0, 1.0),
    "Fachada4": ((0.78, 0.74, 0.55), 0.9, 0.0, 1.0),
    "Azotea": ((0.40, 0.39, 0.38), 0.95, 0.0, 1.0),
    "SueloUrbano": ((0.36, 0.35, 0.33), 1.0, 0.0, 1.0),
    "Verde": ((0.17, 0.36, 0.11), 1.0, 0.0, 1.0),
    "Cancha": ((0.12, 0.44, 0.22), 0.9, 0.0, 1.0),
    "CanchaDura": ((0.62, 0.30, 0.16), 0.9, 0.0, 1.0),
    "Tierra": ((0.55, 0.36, 0.22), 1.0, 0.0, 1.0),
    "Estacionamiento": ((0.24, 0.24, 0.25), 0.95, 0.0, 1.0),
    "Agua": ((0.10, 0.30, 0.45), 0.2, 0.0, 1.0),
    "Tronco": ((0.26, 0.17, 0.10), 1.0, 0.0, 1.0),
    "Follaje1": ((0.10, 0.30, 0.08), 1.0, 0.0, 1.0),
    "Follaje2": ((0.16, 0.38, 0.10), 1.0, 0.0, 1.0),
    "Follaje3": ((0.08, 0.24, 0.12), 1.0, 0.0, 1.0),
    "Podio": ((0.80, 0.10, 0.10), 0.6, 0.0, 1.0),
    "Bosque": ((0.10, 0.25, 0.08), 1.0, 0.0, 1.0),
    "Pradera": ((0.24, 0.40, 0.15), 1.0, 0.0, 1.0),
    "Nube": ((0.96, 0.96, 0.97), 1.0, 0.0, 1.0),
    "Naranja": ((0.95, 0.42, 0.04), 0.7, 0.0, 1.0),
    "Lampara": ((1.0, 1.0, 0.88), 0.3, 0.0, 1.0),
    "Piel": ((0.76, 0.56, 0.42), 0.8, 0.0, 1.0),
    "Pantalon": ((0.09, 0.10, 0.14), 0.8, 0.0, 1.0),
    "AsientoMarino": ((0.06, 0.09, 0.20), 0.7, 0.0, 1.0),
    "Andamio": ((0.17, 0.18, 0.20), 0.6, 0.5, 1.0),
    "Pasillo": ((0.60, 0.60, 0.58), 0.9, 0.0, 1.0),
    "Tecpro": ((0.75, 0.07, 0.05), 0.6, 0.0, 1.0),
    "Publicidad": ((0.04, 0.17, 0.55), 0.6, 0.0, 1.0),
}

# Supuestos verticales (m)
GRADA = {"fondo_escalon": 1.7, "peralte": 0.9, "altura_frente": 2.5, "altura_frente_principal": 4.5,
         "altura_max": 27.0, "modulo_m": 12.0, "ancho_pasillo_m": 1.2}
RADIO_ENTORNO_M = 330.0    # alrededor de la pista; más lejos no se modela (ahí no corren los carros)
RADIO_PALACIO_M = 230.0
MAX_ARBOLES = 2000         # tope para no engordar el modelo de la web
FOCO_FORO_SOL = np.array([-310.0, -95.0])     # centro del estadio: las gradas miran hacia aquí
FOCO_HARP_HELU = np.array([805.0, -300.0])    # centro del diamante
FOCO_PALILLO = np.array([-214.0, 177.0])
IDX_FORO_NORTE, IDX_FORO_SUR = 1988, 2885      # índices en entorno.json
IDX_PRINCIPALES = (1993, 1994)                 # tribunas Main, frente a los pits


# Materiales con imagen (generadas por `generar_texturas.py`); usan el mapa UV de la malla.
TEXTURAS = {"Anuncios": "anuncios.png", "Cartel": "cartel.png", "BanderaMexico": "bandera_mexico.png",
            "Dirigible": "dirigible.png", "Publico": "publico.png", "Pancartas": "pancartas.png"}
N_PANCARTAS = 4   # pancartas apiladas en `pancartas.png`
MOSAICO = 16.0   # el mosaico de anuncios mide 16 veces su alto


def material_textura(nombre: str):
    mat = bpy.data.materials.get(nombre) or bpy.data.materials.new(nombre)
    ruta = AQUI / "texturas" / TEXTURAS[nombre]
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}: ejecutar antes `python blender/generar_texturas.py`")
    try:
        mat.use_nodes = True
    except Exception:
        pass
    arbol = mat.node_tree
    bsdf = next(nd for nd in arbol.nodes if nd.type == "BSDF_PRINCIPLED")
    tex = next((nd for nd in arbol.nodes if nd.type == "TEX_IMAGE"), None) or arbol.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(ruta), check_existing=True)
    img.reload()
    tex.image = img
    arbol.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    mat.diffuse_color = {"BanderaMexico": (0.9, 0.9, 0.9, 1.0), "Publico": (0.25, 0.28, 0.40, 1.0)}.get(nombre, (0.04, 0.17, 0.55, 1.0))
    return mat


def material(nombre: str):
    if nombre in TEXTURAS:
        return material_textura(nombre)
    mat = bpy.data.materials.get(nombre) or bpy.data.materials.new(nombre)
    (r, g, b), rug, met, alfa = MATERIALES[nombre]
    mat.diffuse_color = (r, g, b, alfa)
    mat.roughness, mat.metallic = rug, met
    try:
        mat.use_nodes = True
    except Exception:
        pass
    if mat.node_tree:
        # se busca por tipo: el nombre del nodo cambia con el idioma de Blender
        bsdf = next((nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED"), None)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
            bsdf.inputs["Roughness"].default_value = rug
            bsdf.inputs["Metallic"].default_value = met
            bsdf.inputs["Alpha"].default_value = alfa
    return mat


class Malla:
    """Acumula caras (con su material) y crea un objeto de malla."""

    def __init__(self, nombre: str):
        self.nombre, self.v, self.f, self.m, self.mats, self.uv = nombre, [], [], [], [], []

    def _mat(self, nombre):
        if nombre not in self.mats:
            self.mats.append(nombre)
        return self.mats.index(nombre)

    def cara(self, puntos, mat, uv=None):
        base = len(self.v)
        self.v.extend(tuple(float(c) for c in p) for p in puntos)
        self.f.append(tuple(range(base, base + len(puntos))))
        self.m.append(self._mat(mat))
        self.uv.extend(uv if uv is not None else [(0.0, 0.0)] * len(puntos))

    def letrero(self, a, b, z0, z1, mat, hacia=None, s_a=None, s_b=None, mosaico=None, v0=0.0, v1=1.0):
        """Rectángulo vertical con textura entre a y b. Si se da `hacia` (un punto del
        lado desde donde se mira), el texto se lee de izquierda a derecha desde ahí.
        Con `mosaico` (largo en metros de una repetición) la imagen se repite a lo
        largo; `s_a` y `s_b` (metros acumulados) dan continuidad entre tramos vecinos."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        largo = float(np.hypot(*(b - a)))
        if s_a is None:
            s_a, s_b = 0.0, largo
        if hacia is not None:
            d = 0.5 * (a + b) - np.asarray(hacia, float)
            if np.dot(b - a, np.array([d[1], -d[0]])) < 0:     # desde ahí, b queda a la izquierda
                a, b, s_a, s_b = b, a, -s_b, -s_a
        u0, u1 = (0.0, 1.0) if mosaico is None else (s_a / mosaico, s_b / mosaico)
        self.cara([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)], mat,
                  uv=[(u0, v0), (u1, v0), (u1, v1), (u0, v1)])

    def persona(self, pie, mira, mat_ropa, brazos=False):
        """Figura de 1.72 m hecha con tres cajas (piernas, torso, cabeza), de frente
        hacia `mira`. Con `brazos`, los lleva en alto (para sostener una pancarta)."""
        pie, mira = np.asarray(pie, float), np.asarray(mira, float)
        hombros = np.array([-mira[1], mira[0]]) / np.hypot(*mira)
        self.caja(pie, hombros, 0.20, 0.12, 0.0, 0.85, "Pantalon")
        self.caja(pie, hombros, 0.26, 0.14, 0.85, 1.45, mat_ropa)
        self.caja(pie, hombros, 0.11, 0.11, 1.48, 1.72, "Piel")
        if brazos:
            for lado in (-1.0, 1.0):
                self.caja(pie + hombros * lado * 0.32, hombros, 0.05, 0.05, 1.35, 2.0, mat_ropa)

    def pancarta(self, a, b, z0, z1, k, hacia):
        """Pancarta número k del atlas `pancartas.png`, legible desde `hacia`."""
        k %= N_PANCARTAS
        self.letrero(a, b, z0, z1, "Pancartas", hacia=hacia, v0=1 - (k + 1) / N_PANCARTAS, v1=1 - k / N_PANCARTAS)

    def elipsoide(self, c, ejes, rumbo, mat, seg=10, anillos=8):
        """Elipsoide con el eje largo sobre `rumbo` (vector 2D). `mat` es un nombre o
        una función (anillo, segmento) -> nombre."""
        c = np.asarray(c, float)
        d = np.array([rumbo[0], rumbo[1], 0.0]) / np.hypot(*rumbo)
        lado, arriba = np.array([-d[1], d[0], 0.0]), np.array([0.0, 0.0, 1.0])
        a, b, cz = ejes
        elige = mat if callable(mat) else (lambda i, k: mat)
        filas = [[c + d * a * math.cos(math.pi * i / anillos) + (lado * b * math.cos(2 * math.pi * k / seg) + arriba * cz * math.sin(2 * math.pi * k / seg)) * math.sin(math.pi * i / anillos)
                  for k in range(seg)] for i in range(1, anillos)]
        nariz, cola = c + d * a, c - d * a
        for k in range(seg):
            k1 = (k + 1) % seg
            self.cara([nariz, filas[0][k], filas[0][k1]], elige(0, k))
            for i in range(len(filas) - 1):
                self.cara([filas[i][k], filas[i + 1][k], filas[i + 1][k1], filas[i][k1]], elige(i + 1, k))
            self.cara([filas[-1][k], cola, filas[-1][k1]], elige(anillos - 1, k))

    def revolucion(self, c, perfil, mat, seg=12):
        """Cuerpo de revolución alrededor de la vertical; `perfil` = [(altura, radio), ...]."""
        c = np.asarray(c, float)
        elige = mat if callable(mat) else (lambda k: mat)
        filas = [[(c[0] + r * math.cos(2 * math.pi * k / seg), c[1] + r * math.sin(2 * math.pi * k / seg), c[2] + h) for k in range(seg)] for h, r in perfil]
        for i in range(len(filas) - 1):
            for k in range(seg):
                k1 = (k + 1) % seg
                self.cara([filas[i][k], filas[i][k1], filas[i + 1][k1], filas[i + 1][k]], elige(k))

    def bandera(self, pie, direccion, z_base, alto_asta=4.5, ancho=2.4, alto=1.4):
        """Asta con una bandera de México; el verde queda del lado del asta."""
        pie, direccion = np.asarray(pie, float), np.asarray(direccion, float)
        direccion = direccion / np.hypot(*direccion)
        self.caja(pie, direccion, 0.05, 0.05, z_base, z_base + alto_asta, "Acero")
        tope = z_base + alto_asta
        self.letrero(pie, pie + direccion * ancho, tope - alto, tope, "BanderaMexico")

    def plano(self, poli, z, mat):
        self.cara([(x, y, z) for x, y in poli], mat)

    def pared(self, a, b, z0, z1, mat):
        self.cara([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)], mat)

    def prisma(self, poli, z0, z1, mat_pared, mat_techo=None, fondo=False):
        n = len(poli)
        for i in range(n):
            self.pared(poli[i], poli[(i + 1) % n], z0, z1, mat_pared)
        self.plano(poli, z1, mat_techo or mat_pared)
        if fondo:
            self.plano(poli[::-1], z0, mat_techo or mat_pared)

    def caja(self, centro, u, medio_largo, medio_ancho, z0, z1, mat, mat_techo=None, fondo=False):
        u = np.asarray(u, float); n = np.array([-u[1], u[0]]); c = np.asarray(centro, float)
        poli = [c - u * medio_largo - n * medio_ancho, c + u * medio_largo - n * medio_ancho,
                c + u * medio_largo + n * medio_ancho, c - u * medio_largo + n * medio_ancho]
        self.prisma(poli, z0, z1, mat, mat_techo, fondo)

    def crear(self, col):
        if not self.f:
            return None
        me = bpy.data.meshes.new(self.nombre)
        me.from_pydata(self.v, [], self.f)
        for nombre in self.mats:
            me.materials.append(material(nombre))
        me.polygons.foreach_set("material_index", np.asarray(self.m, dtype=np.int32))
        uv = np.asarray(self.uv, dtype=np.float32)
        if uv.any():     # solo las mallas con letreros llevan mapa UV (ahorra peso en el glTF)
            capa = me.uv_layers.new(name="UVMap")
            capa.data.foreach_set("uv", uv.ravel())
        me.update()
        return _objeto(col, self.nombre, me)


# ------------------------------------------------------------- geometría 2D
def limpiar(puntos) -> np.ndarray | None:
    """Polígono sin punto de cierre ni duplicados, en sentido antihorario."""
    p = np.asarray(puntos, float)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    if len(p) > 1:
        p = p[np.concatenate([[True], np.hypot(*np.diff(p, axis=0).T) > 1e-3])]
    if len(p) < 3:
        return None
    return p if geo.area_con_signo(p) > 0 else p[::-1]


def area(p) -> float:
    return abs(geo.area_con_signo(p))


def cinta(linea: np.ndarray, ancho) -> tuple[np.ndarray, np.ndarray]:
    """Bordes izquierdo y derecho de una cinta de ancho dado sobre una polilínea abierta."""
    t = np.gradient(linea, axis=0)
    t /= np.hypot(*t.T)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    a = np.broadcast_to(np.asarray(ancho, float), (len(linea),))[:, None] / 2
    return linea + n * a, linea - n * a


def remuestrear(linea: np.ndarray, paso: float) -> np.ndarray:
    seg = np.hypot(*np.diff(linea, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    nuevos = np.linspace(0, s[-1], max(2, int(s[-1] / paso) + 1))
    return np.column_stack([np.interp(nuevos, s, linea[:, 0]), np.interp(nuevos, s, linea[:, 1])])


# ------------------------------------------------------------------ gradas
def cadenas(poli: np.ndarray, dist: np.ndarray):
    """Separa el contorno de una tribuna en frente y fondo.

    Una tribuna es una banda: dos lados largos (frente y fondo) unidos por dos
    extremos cortos. Se elige el par de aristas de corte que deja los dos tramos
    más largos con el fondo (distancia entre ambos) más corto y parejo; el frente es el tramo más cercano
    al foco (la pista o el centro del estadio), según `dist` por vértice."""
    n = len(poli)
    lado = np.hypot(*(np.roll(poli, -1, axis=0) - poli).T)      # lado[i]: arista i -> i+1
    mejor = None
    u9 = np.linspace(0.0, 1.0, 9)
    for i in range(n):                                          # corte en la arista i-1 -> i
        for largo in range(2, n - 1):                           # tramo A: vértices i .. i+largo-1
            j = (i + largo - 1) % n                             # corte en la arista j -> j+1
            idx = [(i + k) % n for k in range(largo)]
            resto = [(i + largo + k) % n for k in range(n - largo)]
            la = sum(lado[q] for q in idx[:-1]); lb = sum(lado[q] for q in resto[:-1])
            if min(la, lb) < 1e-6:
                continue
            A, B = poli[idx], poli[resto][::-1]
            ua, ub = _param(A), _param(B)
            fondo = np.hypot(np.interp(u9, ua, A[:, 0]) - np.interp(u9, ub, B[:, 0]),
                             np.interp(u9, ua, A[:, 1]) - np.interp(u9, ub, B[:, 1]))
            nota = min(la, lb) / (fondo.mean() + 2 * fondo.std() + 1e-9)   # banda larga, de fondo corto y parejo
            if mejor is None or nota > mejor[0]:
                mejor = (nota, idx, resto)
    _, a, b = mejor
    if dist[a].mean() > dist[b].mean():
        a, b = b, a
    return poli[a], poli[b][::-1]


def _param(linea):
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(linea, axis=0).T))])
    return s / s[-1]


def tribuna(m: Malla, poli: np.ndarray, dist: np.ndarray, mat_zona: str, mat_asiento="AsientoMarino",
            altura_frente=None, techo=False, palcos=False) -> dict:
    """Grada escalonada entre el frente y el fondo de la huella.

    Estructura tipo andamio (fondo y costados oscuros), frente con una banda del
    color de la zona de boletos, asientos oscuros y pasillos claros cada módulo.
    Con `techo`, cubierta de membrana blanca en bóvedas con mástiles; con
    `palcos`, una franja acristalada sobre las últimas filas."""
    h0 = GRADA["altura_frente"] if altura_frente is None else altura_frente
    P, Q = cadenas(poli, dist)
    up, uq = _param(P), _param(Q)
    u = np.unique(np.round(np.concatenate([up, uq]), 3))
    largo = 0.5 * (np.hypot(*np.diff(P, axis=0).T).sum() + np.hypot(*np.diff(Q, axis=0).T).sum())
    extra = [np.linspace(a, b, int((b - a) * largo / GRADA["modulo_m"]) + 2)[1:-1] for a, b in zip(u[:-1], u[1:])]
    uc = np.unique(np.concatenate([u] + extra))                 # módulos: columnas y pasillos

    def muestras(uu):
        return (np.column_stack([np.interp(uu, up, P[:, 0]), np.interp(uu, up, P[:, 1])]),
                np.column_stack([np.interp(uu, uq, Q[:, 0]), np.interp(uu, uq, Q[:, 1])]))

    du = GRADA["ancho_pasillo_m"] / largo                       # un pasillo al inicio de cada módulo interior
    uf, pasillo = [uc[0]], []
    for a, b in zip(uc[:-1], uc[1:]):
        if a > uc[0] and b - a > 3 * du:
            uf += [a + du, b]; pasillo += [True, False]
        else:
            uf.append(b); pasillo.append(False)
    uf = np.array(uf)
    Pj, Qj = muestras(uf)
    Pc, Qc = muestras(uc)
    fondo = float(np.hypot(*(Qj - Pj).T).mean())
    R = max(2, int(round(fondo / GRADA["fondo_escalon"])))
    R = min(R, int((GRADA["altura_max"] - h0) / GRADA["peralte"]))
    z = h0 + GRADA["peralte"] * np.arange(R)
    z_tope = float(z[-1])
    filas = [Pj + (r / R) * (Qj - Pj) for r in range(R + 1)]
    nj = len(uf)
    s_frente = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Pj, axis=0).T))])
    for j in range(nj - 1):
        m.pared(filas[0][j + 1], filas[0][j], 0.0, z[0] - 2.2, "Andamio")                # frente
        m.letrero(filas[0][j], filas[0][j + 1], z[0] - 2.2, z[0] - 1.0, "Anuncios", hacia=filas[0][j] - (Qj[j] - Pj[j]),
                  s_a=s_frente[j], s_b=s_frente[j + 1], mosaico=1.2 * MOSAICO)           # anuncios
        m.pared(filas[0][j + 1], filas[0][j], z[0] - 1.0, z[0], mat_zona)                # banda de zona
        mat = "Pasillo" if pasillo[j] else mat_asiento
        for r in range(R):
            a0, a1, b0, b1 = filas[r][j], filas[r][j + 1], filas[r + 1][j], filas[r + 1][j + 1]
            if mat == "AsientoMarino":      # gradas del circuito: butacas con público
                u0, u1 = s_frente[j] / 12.8, s_frente[j + 1] / 12.8
                m.cara([(*a0, z[r]), (*a1, z[r]), (*b1, z[r]), (*b0, z[r])], "Publico", uv=[(u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)])
            else:
                m.cara([(*a0, z[r]), (*a1, z[r]), (*b1, z[r]), (*b0, z[r])], mat)
            if r < R - 1:
                m.pared(b1, b0, z[r], z[r + 1], mat)       # respaldos del color de las butacas
        m.pared(filas[R][j], filas[R][j + 1], 0.0, z_tope + 1.1, "Andamio")              # fondo
    for j, sentido in ((0, 1), (nj - 1, -1)):                                            # costados
        perfil = [(*filas[0][j], 0.0)]
        for r in range(R):
            perfil += [(*filas[r][j], z[r]), (*filas[r + 1][j], z[r])]
        perfil.append((*filas[R][j], 0.0))
        m.cara(perfil[::sentido], "Andamio")

    def A(t):
        return Pc + t * (Qc - Pc)

    nc = len(uc)
    if palcos:  # franja de palcos acristalados sobre las últimas filas
        t0, zp0, zp1 = 1 - 5.0 / fondo, z_tope + 0.3, z_tope + 3.8
        F, B = A(t0), A(1.0)
        for j in range(nc - 1):
            m.pared(F[j + 1], F[j], zp0, zp1, "Vidrio")
            m.cara([(*F[j], zp1), (*F[j + 1], zp1), (*B[j + 1], zp1), (*B[j], zp1)], "TechoBlanco")
            m.cara([(*F[j], zp0), (*B[j], zp0), (*B[j + 1], zp0), (*F[j + 1], zp0)], "TechoBlanco")
            m.pared(B[j], B[j + 1], z_tope, zp1, "Andamio")
        for j in range(nc):                                                              # divisiones entre palcos
            m.cara([(*F[j], zp0), (*B[j], zp0), (*B[j], zp1), (*F[j], zp1)], "TechoBlanco")
    if techo:   # membrana blanca en bóvedas, una por módulo, con mástiles y tirantes
        base = z_tope + (3.8 if palcos else 0.0)
        F, B, Mt = A(0.10), A(1.04), A(0.78)
        zf, zb = base + 6.0, base + 4.0
        comba = (0.0, 1.2, 1.6, 1.2, 0.0)
        for j in range(nc - 1):
            for q in range(4):
                f0, f1 = F[j] + (F[j + 1] - F[j]) * q / 4, F[j] + (F[j + 1] - F[j]) * (q + 1) / 4
                b0, b1 = B[j] + (B[j + 1] - B[j]) * q / 4, B[j] + (B[j + 1] - B[j]) * (q + 1) / 4
                m.cara([(*f0, zf + comba[q]), (*f1, zf + comba[q + 1]), (*b1, zb + comba[q + 1]), (*b0, zb + comba[q])], "TechoBlanco")
        for j in range(nc):
            d = (Qc[j] - Pc[j]) / np.hypot(*(Qc[j] - Pc[j]))
            lado = np.array([-d[1], d[0]])
            pie = A(1.0)[j] + d * 0.5
            m.caja(pie, lado, 0.3, 0.3, 0.0, zb, "Acero")                                # columna al fondo
            zm = zb + (zf - zb) * (1.04 - 0.78) / (1.04 - 0.10)
            m.caja(Mt[j], lado, 0.12, 0.12, zm, zm + 4.5, "Acero")                       # mástil
            for extremo, ze in ((F[j], zf), (B[j], zb)):                                 # tirantes
                m.cara([(*(Mt[j] - lado * 0.06), zm + 4.5), (*(Mt[j] + lado * 0.06), zm + 4.5),
                        (*(extremo + lado * 0.06), ze), (*(extremo - lado * 0.06), ze)], "Acero")
    if not techo:   # banderas de México en lo alto, una por módulo
        for j in range(nc - 1):
            m.bandera(Qc[j] + (Qc[j + 1] - Qc[j]) * 0.5, Qc[j + 1] - Qc[j], z_tope + 1.1)
    return {"filas": R, "altura": z_tope, "fondo_m": fondo, "frente": Pc, "atras": Qc}


def zona_asientos(idx: int, cen: np.ndarray, d_pista: float, nombre) -> str:
    """Color de asientos según la zona del mapa oficial de tribunas (aproximado por posición)."""
    x, y = cen
    if idx == IDX_FORO_NORTE:
        return "AsientoCafe"
    if idx == IDX_FORO_SUR:
        return "AsientoGris"
    if 640 < x < 940 and -420 < y < -150 and d_pista > 90:
        return "AsientoRojo"            # Estadio Alfredo Harp Helú
    if d_pista > 100:
        return "AsientoNeutro"          # gradas de canchas y otros recintos
    if -270 < x < 110 and y > 0:
        return "AsientoVerde"           # Main 1, 1A, 2
    if 400 < x < 660 and y > -80:
        return "AsientoNaranja"         # 2A
    if x > 860:
        return "AsientoAzul"            # frenada y curvas 1 a 3
    if x < -270 and y > 20:
        return "AsientoGris"            # exterior de la Peraltada
    return "AsientoAmarillo"            # sección media


# ---------------------------------------------------------------- edificios
def edificio_pits(col, poli: np.ndarray, altura: float = 13.0, n_fosos: int | None = None) -> dict:
    """Edificio de pits de tres niveles: fosos abajo, dos niveles acristalados de
    palcos (Paddock Club) con balcón, cubierta en voladizo hacia la pista y torre de control."""
    m = Malla("EdificioPits")
    m.prisma(poli, 0.0, altura, "MuroPits", "Azotea")
    lados = np.roll(poli, -1, axis=0) - poli
    i = int(np.hypot(*lados.T).argmax())                    # la fachada que da a la recta es el lado más largo
    p0, p1 = poli[i], poli[(i + 1) % len(poli)]
    if p0[0] > p1[0]:
        p0, p1 = p1, p0                                     # de poniente a oriente
    L = float(np.hypot(*(p1 - p0)))
    u = (p1 - p0) / L
    n = np.array([-u[1], u[0]])
    if np.dot(n, poli.mean(0) - p0) > 0:
        n = -n                                              # normal hacia afuera (hacia la pista)
    margen = 6.0
    if n_fosos is None:
        n_fosos = max(6, int((L - 2 * margen) / 8.8))       # un foso cada ~9 m de fachada
    ancho = (L - 2 * margen) / n_fosos
    niveles = ((4.5, 8.4), (8.7, 12.6))                     # piso y techo de cada nivel de palcos
    for k in range(n_fosos):
        c = p0 + u * (margen + (k + 0.5) * ancho) + n * 0.06
        m.pared(c - u * (ancho * 0.36), c + u * (ancho * 0.36), 0.0, 3.6, "Porton")
    for k in range(n_fosos + 1):                            # aletas que separan los palcos
        c = p0 + u * (margen + k * ancho) + n * 1.25
        m.caja(c, n, 1.25, 0.15, niveles[0][0], altura, "TechoBlanco")
        if k % 3 == 0:                                      # puntal inclinado que carga el voladizo
            a, b = p0 + u * (margen + k * ancho) + n * 2.5, p0 + u * (margen + k * ancho) + n * 6.0
            m.cara([(*(a - u * 0.15), niveles[1][0]), (*(a + u * 0.15), niveles[1][0]),
                    (*(b + u * 0.15), altura), (*(b - u * 0.15), altura)], "Acero")
    medio = 0.5 * (p0 + p1)
    for piso, techo in niveles:
        m.caja(medio + n * 1.25, u, L / 2, 1.25, piso - 0.3, piso, "TechoBlanco", fondo=True)   # balcón
        m.pared(p0 + n * 0.06, p1 + n * 0.06, piso + 0.3, techo, "Vidrio")                      # frente de palcos
        m.pared(p0 + n * 2.5, p1 + n * 2.5, piso, piso + 1.1, "Malla")                          # barandal
    m.caja(medio + n * 1.5, u, L / 2 + 1.5, 5.0, altura, altura + 0.5, "TechoBlanco", fondo=True)  # cubierta en voladizo
    c_torre = p1 - u * 12.0 - n * 12.0                                                           # torre de control (extremo oriente)
    m.caja(c_torre, u, 7.0, 7.0, altura, 22.0, "MuroPits")
    m.caja(c_torre, u, 9.0, 9.0, 22.0, 26.0, "Vidrio", "TechoBlanco", fondo=True)
    m.caja(c_torre, u, 9.6, 9.6, 26.0, 26.5, "TechoBlanco", fondo=True)
    frente = medio + n * 60.0
    m.letrero(p0 - u * 1.5 + n * 6.55, p1 + u * 1.5 + n * 6.55, altura + 0.5, altura + 2.1, "Anuncios", hacia=frente, mosaico=1.6 * MOSAICO)
    for lado in (n, -n, u, -u):                             # espectacular en las cuatro caras de la torre
        q = np.array([-lado[1], lado[0]])
        m.letrero(c_torre + lado * 7.05 - q * 6.0, c_torre + lado * 7.05 + q * 6.0, 15.0, 21.0, "Cartel", hacia=c_torre + lado * 60.0)
    for k in range(0, n_fosos + 1, 3):                      # banderas sobre la cubierta
        m.bandera(p0 + u * (margen + k * ancho) + n * 5.0, u, altura + 0.5, alto_asta=7.0, ancho=3.0, alto=1.7)
    m.bandera(c_torre, u, 26.5, alto_asta=7.0, ancho=4.0, alto=2.3)
    m.crear(col)
    return {"p0": p0, "p1": p1, "u": u, "n": n, "largo_m": L, "n_fosos": n_fosos, "ancho": ancho, "margen": margen}


def cupula(m: Malla, centro, radio, z0, alto, segmentos=32, anillos=6, mat="Cobre"):
    """Cúpula facetada: los vértices alternos sobresalen para dar el relieve en zigzag."""
    pts = []
    for i in range(anillos):
        phi = (math.pi / 2) * i / anillos
        fila = []
        for k in range(segmentos):
            a = 2 * math.pi * (k + 0.5 * (i % 2)) / segmentos
            r = radio * math.cos(phi) + (2.8 if (k + i) % 2 else 0.0) * math.cos(phi)
            zz = z0 + alto * math.sin(phi) + (1.6 if (k + i) % 2 else 0.0)
            fila.append((centro[0] + r * math.cos(a), centro[1] + r * math.sin(a), zz))
        pts.append(fila)
    cima = (centro[0], centro[1], z0 + alto)
    for i in range(anillos - 1):
        for k in range(segmentos):
            k1 = (k + 1) % segmentos
            a, b, c, d = pts[i][k], pts[i][k1], pts[i + 1][k1], pts[i + 1][k]
            if i % 2:
                m.cara([a, b, d], mat); m.cara([b, c, d], mat)
            else:
                m.cara([a, b, c], mat); m.cara([a, c, d], mat)
    for k in range(segmentos):
        m.cara([pts[-1][k], pts[-1][(k + 1) % segmentos], cima], mat)


def techo_abombado(m: Malla, poli, z0, z1, factor, mat):
    c = poli.mean(0)
    dentro = c + (poli - c) * factor
    n = len(poli)
    for i in range(n):
        j = (i + 1) % n
        m.cara([(*poli[i], z0), (*poli[j], z0), (*dentro[j], z1), (*dentro[i], z1)], mat)
    m.plano(dentro, z1, mat)


# ------------------------------------------------------------- construcción
def construir(csv: str = "MexicoCity.csv", entorno: str = "pistas/mexico/entorno.json") -> dict:
    """Con `MexicoCity.csv` reproduce el Hermanos Rodríguez. Con otro circuito:
    - edificio de pits = el edificio de fachada más larga (>= 60 m) junto a la recta de salida;
    - tribunas con techo y palcos = las de más de 800 m² junto a esa recta;
    - los polígonos `bosque` se pueblan con árboles al azar (OSM da el contorno del
      bosque, no cada árbol);
    todo ello son supuestos, como las alturas."""
    mexico = csv == "MexicoCity.csv"
    centro, w_der, w_izq = geo.leer_tum(RAIZ / "pistas" / "fuente_tum" / csv)
    nor = geo.normales_izq(centro)
    b_izq, b_der = geo.bordes(centro, w_der, w_izq)
    s, largo = geo.abscisa(centro)
    tang = geo.tangentes(centro)
    elementos = json.loads((RAIZ / entorno).read_text(encoding="utf-8"))["elementos"]
    informe = {"descartados_por_invadir_pista": []}
    cerca_salida = (s < 380.0) | (s > largo - 380.0)       # recta de salida/meta

    def d_pista(p):  # distancia de puntos a la línea central
        return geo.dist_a_polilinea(np.atleast_2d(p), centro)

    def invade_pista(poli) -> bool:
        cen = poli.mean(0)
        radio = np.hypot(*(poli - cen).T).max()
        cerca = np.hypot(*(centro - cen).T) < radio + 25
        if not cerca.any():
            return False
        d = geo.dist_a_polilinea(centro[cerca], poli)
        dentro = geo.dentro(centro[cerca], poli)
        ancho = np.maximum(w_der, w_izq)[cerca]
        return bool(np.any(dentro | (d < ancho + 1.5)))

    palacio_xy = next((np.array(e["puntos"], float) for e in elementos if e["nombre"] == "Palacio de los Deportes"), None)
    c_palacio = None if palacio_xy is None else 0.5 * (palacio_xy.min(0) + palacio_xy.max(0))

    def cerca(p) -> bool:
        """Solo se modela lo que rodea a la pista (y, en México, al Palacio de los Deportes)."""
        cen = np.atleast_2d(p).mean(0)
        if np.hypot(*(centro - cen).T).min() < RADIO_ENTORNO_M:
            return True
        return c_palacio is not None and bool(np.hypot(*(cen - c_palacio)) < RADIO_PALACIO_M)

    por_cat: dict[str, list] = {}
    for i, e in enumerate(elementos):
        por_cat.setdefault(e["categoria"], []).append((i, e))

    # ---- suelo
    suelo = _coleccion("Entorno_Suelo")
    margen = RADIO_ENTORNO_M + 60
    x0, y0 = centro[:, 0].min() - margen, centro[:, 1].min() - margen
    if c_palacio is not None:
        x0 = min(x0, c_palacio[0] - RADIO_PALACIO_M - 60)
    x1, y1 = centro[:, 0].max() + margen, centro[:, 1].max() + margen
    mat_suelo = "SueloUrbano" if mexico else "Pradera"      # los circuitos de prueba están en campo o parque
    m = Malla("SueloBase"); m.plano([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], -0.10, mat_suelo); m.crear(suelo)
    for cat, nombre, z, mat in (("verde", "AreasVerdes", -0.07, "Verde"), ("bosque", "Bosques", -0.072, "Bosque"), ("estacionamiento", "Estacionamientos", -0.065, "Estacionamiento"),
                                ("estadio", "Explanadas", -0.068, "Concreto"), ("agua", "Agua", -0.058, "Agua")):
        m = Malla(nombre)
        for _, e in por_cat.get(cat, []):
            if len(e["puntos"]) < 4 or e["puntos"][0] != e["puntos"][-1]:
                continue        # es una línea (arroyo, hilera de árboles), no un área
            p = limpiar(e["puntos"])
            if p is not None and cerca(p):
                m.plano(p, z, mat)
        m.crear(suelo)
    m = Malla("Canchas")
    for _, e in por_cat.get("cancha", []):
        p = limpiar(e["puntos"])
        if p is None or not cerca(p):
            continue
        nom = (e["nombre"] or "").lower()
        mat = "CanchaDura" if "basquet" in nom else ("Tierra" if "bullpen" in nom else "Cancha")
        m.plano(p, -0.06, mat)
    m.crear(suelo)

    # ---- pits: pit lane, edificio, muro, parrilla, pórtico
    pits = _coleccion("Entorno_Pits")
    lineas_pit = [np.array(e["puntos"], float) for _, e in por_cat.get("pista_osm", [])
                  if e["nombre"] and any(c in e["nombre"].lower() for c in ("pit", "box")) and len(e["puntos"]) > 1]
    pit_pts = np.vstack([remuestrear(l, 3.0) for l in lineas_pit]) if lineas_pit else None

    def d_pit(puntos) -> np.ndarray:
        """Distancia de cada punto al pit lane (infinito si OSM no lo trae)."""
        puntos = np.atleast_2d(puntos)
        if pit_pts is None:
            return np.full(len(puntos), np.inf)
        return np.array([np.hypot(*(pit_pts - q).T).min() for q in puntos])

    m = Malla("PitLane")
    for l in lineas_pit:
        l = remuestrear(l, 6.0)
        izq, der = cinta(l, 11.0)
        for j in range(len(l) - 1):
            m.cara([(*der[j], -0.005), (*der[j + 1], -0.005), (*izq[j + 1], -0.005), (*izq[j], -0.005)], "AsfaltoPits")
    m.crear(pits)

    paddock = None
    if mexico:
        paddock = next(limpiar(e["puntos"]) for _, e in por_cat["edificio"] if e["nombre"] == "Paddock")
    else:   # el edificio de fachada más larga junto a la recta de salida
        mejor = 60.0
        for _, e in por_cat.get("edificio", []):
            p = limpiar(e["puntos"])
            if p is None or np.hypot(*(centro[cerca_salida] - p.mean(0)).T).min() > 110 or invade_pista(p):
                continue
            lado_max = np.hypot(*(np.roll(p, -1, axis=0) - p).T).max()
            if lado_max > mejor:
                mejor, paddock = lado_max, p
    en_recta = np.zeros(len(centro), bool)
    lado_pits = -1.0
    if paddock is not None:
        fach = edificio_pits(pits, paddock, n_fosos=33 if mexico else None)
        informe["fachada_pits_m"] = round(fach["largo_m"], 1)
        a_, b_ = fach["p0"] - fach["u"] * 20.0, fach["p1"] + fach["u"] * 40.0
        ab = b_ - a_
        u_ = np.clip((centro - a_) @ ab / (ab @ ab), 0.0, 1.0)
        d_fach = np.hypot(*(a_ + u_[:, None] * ab - centro).T)
        en_recta = (d_fach < 70.0) & ((u_ > 0) & (u_ < 1))                    # tramo de pista frente al edificio
        i_ref = int(d_fach.argmin())
        lado_pits = 1.0 if np.dot(0.5 * (fach["p0"] + fach["p1"]) - centro[i_ref], nor[i_ref]) > 0 else -1.0
    huellas_fijas = [paddock] if paddock is not None else []

    m = Malla("MuroPits")   # muro entre la recta y el pit lane, frente al edificio
    w_pits = w_izq if lado_pits > 0 else w_der
    linea = centro + nor * (lado_pits * (w_pits + 1.8))[:, None]
    for i in np.flatnonzero(en_recta & np.roll(en_recta, -1)):
        j = (i + 1) % len(centro)
        m.pared(linea[i], linea[j], 0.0, 1.1, "Concreto"); m.pared(linea[j], linea[i], 0.0, 1.1, "Concreto")
        m.pared(linea[i], linea[j], 1.1, 2.6, "Malla")
        m.letrero(linea[i] - nor[i] * lado_pits * 0.12, linea[j] - nor[j] * lado_pits * 0.12, 0.15, 1.05, "Anuncios", hacia=centro[i],
                  s_a=s[i], s_b=s[i] + float(np.hypot(*(linea[j] - linea[i]))), mosaico=0.9 * MOSAICO)
    m.crear(pits)
    m = Malla("Parrilla")   # 20 cajones de salida en dos filas escalonadas
    for k in range(20):
        c, i, _ = geo.interpolar(centro, s, largo, largo - 6.0 - 8.0 * k)
        t = geo.tangentes(centro)[i]; nn = np.array([-t[1], t[0]])
        cc = c + nn * (2.6 if k % 2 == 0 else -2.6)
        for a, b in ((cc - nn * 1.1, cc + nn * 1.1), (cc - nn * 1.1 - t * 1.2, cc - nn * 1.1), (cc + nn * 1.1 - t * 1.2, cc + nn * 1.1)):
            d = (b - a) / np.hypot(*(b - a)); q = np.array([-d[1], d[0]]) * 0.06
            m.cara([(*(a - q), 0.006), (*(b - q), 0.006), (*(b + q), 0.006), (*(a + q), 0.006)], "LineaBlanca")
    m.crear(pits)
    m = Malla("PorticoSalida")
    t0 = geo.tangentes(centro)[0]; n0 = nor[0]
    pa, pb = centro[0] + n0 * (w_izq[0] + 2.5), centro[0] - n0 * (w_der[0] + 1.0)
    for p in (pa, pb):
        m.caja(p, t0, 0.3, 0.3, 0.0, 8.0, "Acero")
    m.caja(0.5 * (pa + pb), n0, np.hypot(*(pa - pb)) / 2, 0.4, 7.2, 8.0, "Acero", fondo=True)
    for sentido in (1.0, -1.0):                            # letrero por ambas caras
        m.letrero(pa + t0 * 0.42 * sentido, pb + t0 * 0.42 * sentido, 8.0, 9.2, "Anuncios", hacia=centro[0] + t0 * 40.0 * sentido)
    for p in (pa, pb):
        m.bandera(p, t0, 8.0, alto_asta=5.0, ancho=3.0, alto=1.7)
    for k in range(5):                                    # semáforo
        c = centro[0] + n0 * (k - 2) * 1.6
        m.caja(c, n0, 0.6, 0.25, 6.2, 7.2, "Porton", fondo=True)
    m.crear(pits)

    # ---- tribunas
    trib = _coleccion("Entorno_Tribunas")
    estadios = _coleccion("Entorno_Estadios")
    detalle = []
    huellas_tribuna = []
    for i, e in por_cat.get("tribuna", []):
        p = limpiar(e["puntos"])
        if p is None or not cerca(p):
            continue
        cen = p.mean(0)
        dp = float(d_pista(cen)[0])
        nombre = e["nombre"] or ""
        if "Gimnasio" in nombre or "Dugout" in nombre or "Palco" in nombre:   # son edificios, no gradas
            m = Malla(f"Edificio_{i}"); m.prisma(p, 0.0, max(e["altura_m"], 3.2), "Fachada3", "Azotea"); m.crear(estadios)
            continue
        if invade_pista(p):
            informe["descartados_por_invadir_pista"].append(f"tribuna {i}")
            continue
        mat = zona_asientos(i, cen, dp, nombre) if mexico else "AsientoAzul"
        if not mexico:
            dist, destino, nom = geo.dist_a_polilinea(p, centro), trib, f"Tribuna_{i}"
        elif i in (IDX_FORO_NORTE, IDX_FORO_SUR):
            dist, destino, nom = np.hypot(*(p - FOCO_FORO_SOL).T), estadios, f"EstadioGNP_{'Norte' if i == IDX_FORO_NORTE else 'Sur'}"
        elif mat == "AsientoRojo":
            dist, destino, nom = np.hypot(*(p - FOCO_HARP_HELU).T), estadios, f"HarpHelu_Grada_{i}"
        elif np.hypot(*(cen - FOCO_PALILLO)) < 130:
            dist, destino, nom = np.hypot(*(p - FOCO_PALILLO).T), estadios, f"Palillo_Grada_{i}"
        else:
            dist, destino, nom = geo.dist_a_polilinea(p, centro), trib, f"Tribuna_{i}"
        if mexico:
            principal = i in IDX_PRINCIPALES      # Main: elevadas, con membrana y Sky Boxes
        else:
            principal = area(p) > 800 and np.hypot(*(centro[cerca_salida] - cen).T).min() < 90
        m = Malla(nom)
        r = tribuna(m, p, dist, mat, mat_asiento="AsientoRojo" if mat == "AsientoRojo" else "AsientoMarino",
                    altura_frente=GRADA["altura_frente_principal"] if principal else None, techo=principal, palcos=principal)
        m.crear(destino)
        huellas_tribuna.append(p)
        detalle.append({"indice": i, "objeto": nom, "zona": mat, "filas": r["filas"], "altura_m": round(r["altura"], 1), "fondo_m": round(r["fondo_m"], 1)})
        if mexico and i == IDX_FORO_NORTE:   # podio frente a la tribuna norte del estadio
            j = len(r["frente"]) // 3
            base = r["frente"][j] + (FOCO_FORO_SOL - r["frente"][j]) / np.hypot(*(FOCO_FORO_SOL - r["frente"][j])) * 7.0
            ux = np.array([1.0, 0.0])
            mp = Malla("Podio")
            mp.caja(base, ux, 7.0, 3.0, 0.0, 1.6, "ConcretoOscuro")
            for dx, h in ((0.0, 1.0), (-2.2, 0.7), (2.2, 0.5)):
                mp.caja(base + ux * dx, ux, 1.0, 1.0, 1.6, 1.6 + h, "Podio")
            mp.caja(base + np.array([0.0, 3.2]), ux, 7.0, 0.2, 1.6, 6.5, "TechoBlanco")
            mp.crear(estadios)
    informe["tribunas"] = detalle

    # ---- estadios y edificios emblemáticos
    especiales = {"Paddock"}
    for _, e in por_cat.get("edificio", []):
        p = limpiar(e["puntos"])
        if p is None or not e["nombre"]:
            continue
        if e["nombre"] == "Palacio de los Deportes":
            m = Malla("PalacioDeLosDeportes")
            m.prisma(p, 0.0, 14.0, "Ladrillo", "ConcretoOscuro")
            c = 0.5 * (p.min(0) + p.max(0))
            cupula(m, c, 86.0, 14.0, 30.0)
            m.crear(estadios); especiales.add(e["nombre"])
        elif e["nombre"] == "Tridente":       # cubierta del Estadio Alfredo Harp Helú
            m = Malla("HarpHelu_Tridente")
            m.prisma(p, 30.0, 32.5, "TechoBlanco", "TechoBlanco", fondo=True)
            for q in p[::2]:
                cq = q + (p.mean(0) - q) * 0.04
                m.caja(cq, np.array([1.0, 0.0]), 0.8, 0.8, 0.0, 30.0, "Acero")
            m.crear(estadios); especiales.add(e["nombre"])
        elif e["nombre"] == "Velódromo CNAR":
            m = Malla("VelodromoCNAR")
            m.prisma(p, 0.0, 8.0, "Fachada3", "Acero")
            techo_abombado(m, p, 8.0, 13.0, 0.55, "Acero")
            m.crear(estadios); especiales.add(e["nombre"])

    # ---- ciudad: el resto de edificios como volúmenes
    ciudad = _coleccion("Entorno_Ciudad")
    m = Malla("Edificios")
    n_edif = 0
    for i, e in por_cat.get("edificio", []):
        if e["nombre"] in especiales:
            continue
        p = limpiar(e["puntos"])
        if p is None or area(p) < 4 or not cerca(p) or p is paddock or (paddock is not None and len(p) == len(paddock) and np.allclose(p, paddock)):
            continue
        if invade_pista(p):
            informe["descartados_por_invadir_pista"].append(f"edificio {i} ({e['nombre']})")
            continue
        h = e["altura_m"] if e["altura_m"] and e["altura_m"] > 0 else 8.0
        if area(p) < 60:
            h = min(h, 3.5)     # casetas y puestos: OSM les pone la altura por defecto de un edificio
        m.prisma(p, 0.0, h, f"Fachada{1 + i % 4}", "Azotea")
        n_edif += 1
    m.crear(ciudad)
    informe["edificios"] = n_edif

    # ---- barreras: muro y malla detrás de la escapatoria; publicidad y bloques Tecpro en las curvas
    barreras = _coleccion("Entorno_Barreras")
    from construir_pista import _dilatar, ancho_escapatoria, zonas_de_curva
    en_curva, k = zonas_de_curva(centro)
    m = Malla("MurosContencion")
    n_muro = 0
    for signo, w, izquierda in ((1.0, w_izq, True), (-1.0, w_der, False)):
        esc = ancho_escapatoria(centro, csv, izquierda)
        d = w + esc + 4.0
        linea = centro + nor * (signo * d)[:, None]
        libre = geo.dist_a_polilinea(linea, centro) > d - 1.0           # no cruza otro tramo de pista
        libre &= d_pit(linea) > 9.0                                     # ni el pit lane
        for h in huellas_tribuna + huellas_fijas:
            libre &= ~geo.dentro(linea, h)
        if signo == lado_pits:
            libre &= ~en_recta                                           # ahí va el muro de pits
        for i in np.flatnonzero(libre & np.roll(libre, -1)):
            j = (i + 1) % len(centro)
            if np.hypot(*(linea[j] - linea[i])) > 12:
                continue
            a, b = (linea[i], linea[j]) if signo > 0 else (linea[j], linea[i])
            m.pared(a, b, 0.0, 1.1, "Concreto"); m.pared(b, a, 0.0, 1.1, "Concreto")
            m.pared(a, b, 1.1, 2.8, "Malla")
            hacia = -nor[i] * signo * 0.12
            m.letrero(linea[i] + hacia, linea[j] + hacia, 0.15, 1.05, "Anuncios", hacia=centro[i],
                      s_a=s[i], s_b=s[i] + float(np.hypot(*(linea[j] - linea[i]))), mosaico=0.9 * MOSAICO)
            if esc[i] > 4.0:                                             # exterior de curva
                if i % 2 == 0:
                    t_ = (b - a) / np.hypot(*(b - a))
                    m.caja(0.5 * (a + b) - nor[i] * signo * 0.7, t_, 1.4, 0.45, 0.0, 1.0, "Tecpro" if (i // 2) % 2 else "TechoBlanco")
            n_muro += 1
    m.crear(barreras)
    informe["tramos_de_muro"] = n_muro

    m = Malla("Espectaculares")
    n_esp = 0
    for signo, w, izquierda in ((1.0, w_izq, True), (-1.0, w_der, False)):
        esc = ancho_escapatoria(centro, csv, izquierda)
        d = w + esc + 9.0
        linea = centro + nor * (signo * d)[:, None]
        libre = geo.dist_a_polilinea(linea, centro) > d - 1.0
        libre &= d_pit(linea) > 14.0
        for h in huellas_tribuna + huellas_fijas:           # ni dentro ni pegado a gradas o pits
            libre &= ~geo.dentro(linea, h) & (geo.dist_a_polilinea(linea, h) > 9.0)
        if signo == lado_pits:
            libre &= ~en_recta
        ultimo = -10**9
        for i in range(len(centro)):
            en_exterior = esc[i] > 0.6 * esc.max() and esc[i] >= esc[i - 1] and esc[i] > esc[(i + 1) % len(centro)]
            cada_tanto = s[i] - ultimo > 230.0
            if not libre[i] or not (en_exterior or cada_tanto) or s[i] - ultimo < 70.0:
                continue
            c, t_ = linea[i], tang[i]
            for q in (-5.0, 5.0):
                m.caja(c + t_ * q, t_, 0.25, 0.25, 0.0, 3.5, "Acero")
            m.letrero(c - t_ * 7.0, c + t_ * 7.0, 3.5, 10.5, "Cartel", hacia=centro[i])
            m.pared(c + t_ * 7.0 + nor[i] * signo * 0.15, c - t_ * 7.0 + nor[i] * signo * 0.15, 3.5, 10.5, "Andamio")   # reverso
            ultimo = s[i]; n_esp += 1
    m.crear(barreras)
    informe["espectaculares"] = n_esp

    # ---- detalles a pie de pista: torres de luz, casetas de comisarios, puentes publicitarios, carpas
    detalles = _coleccion("Entorno_Detalles")
    esc_lado = {1.0: ancho_escapatoria(centro, csv, True), -1.0: ancho_escapatoria(centro, csv, False)}
    w_lado = {1.0: w_izq, -1.0: w_der}

    def sitio(signo, extra, margen=6.0):
        """Puntos a `extra` metros detrás de la escapatoria y cuáles están libres de pista, pits y gradas."""
        d = w_lado[signo] + esc_lado[signo] + extra
        linea = centro + nor * (signo * d)[:, None]
        ok = (geo.dist_a_polilinea(linea, centro) > d - 1.0) & (d_pit(linea) > 10.0)
        for h in huellas_tribuna + huellas_fijas:
            ok &= ~geo.dentro(linea, h) & (geo.dist_a_polilinea(linea, h) > margen)
        if signo == lado_pits:
            ok &= ~en_recta
        return linea, ok

    m = Malla("TorresDeLuz")
    n_torres, ultimo, signo = 0, -1e9, 1.0
    lineas = {sg: sitio(sg, 13.0) for sg in (1.0, -1.0)}
    for i in range(len(centro)):
        if s[i] - ultimo < 260.0 or not lineas[signo][1][i]:
            continue
        c = lineas[signo][0][i]
        m.caja(c, tang[i], 0.3, 0.3, 0.0, 28.0, "Acero")
        m.caja(c - nor[i] * signo * 0.8, tang[i], 2.6, 0.5, 28.0, 30.0, "Lampara", fondo=True)
        ultimo, signo, n_torres = s[i], -signo, n_torres + 1
    m.crear(detalles)
    m = Malla("CasetasComisarios")
    n_casetas, ultimo, signo = 0, -1e9, -1.0
    lineas = {sg: sitio(sg, 6.3, margen=3.0) for sg in (1.0, -1.0)}
    for i in range(len(centro)):
        if s[i] - ultimo < 190.0 or not lineas[signo][1][i]:
            continue
        c = lineas[signo][0][i]
        m.caja(c, tang[i], 1.2, 1.0, 0.0, 2.3, "Naranja")
        m.caja(c, tang[i], 1.5, 1.3, 2.3, 2.5, "TechoBlanco", fondo=True)
        ultimo, signo, n_casetas = s[i], -signo, n_casetas + 1
    m.crear(detalles)
    m = Malla("PuentesPublicitarios")
    recta_limpia = ~_dilatar(en_curva, 10) & ~_dilatar(en_recta, 6) & (s > 300.0) & (s < largo - 300.0)
    ok_i, ok_d = sitio(1.0, 5.5)[1], sitio(-1.0, 5.5)[1]
    candidatos = np.flatnonzero(recta_limpia & ok_i & ok_d)
    puentes = []
    for fraccion in (0.33, 0.66):
        if len(candidatos):
            i = int(candidatos[np.abs(s[candidatos] - fraccion * largo).argmin()])
            if all(abs(s[i] - s[q]) > 400.0 for q in puentes):
                puentes.append(i)
    for i in puentes:
        pa, pb = centro[i] + nor[i] * (w_izq[i] + 5.5), centro[i] - nor[i] * (w_der[i] + 5.5)
        for pie in (pa, pb):
            for q in (-1.2, 1.2):
                m.caja(pie + tang[i] * q, tang[i], 0.3, 0.3, 0.0, 6.8, "Acero")
        m.caja(0.5 * (pa + pb), nor[i], np.hypot(*(pa - pb)) / 2 + 0.6, 1.6, 6.8, 7.2, "ConcretoOscuro", fondo=True)
        for sentido in (1.0, -1.0):
            m.letrero(pa + tang[i] * 1.6 * sentido, pb + tang[i] * 1.6 * sentido, 7.2, 8.6, "Anuncios",
                      hacia=centro[i] + tang[i] * 50.0 * sentido, mosaico=1.4 * MOSAICO)
    m.crear(detalles)
    n_carpas = 0
    if paddock is not None:      # carpas blancas de hospitalidad detrás del edificio de pits
        m = Malla("Carpas")
        fondo_ed = float(((paddock - fach["p0"]) @ -fach["n"]).max())
        for k in range(int(fach["largo_m"] // 15)):
            c = fach["p0"] + fach["u"] * (8.0 + 15.0 * k) - fach["n"] * (fondo_ed + 16.0)
            if np.hypot(*(centro - c).T).min() < 30.0 or any(geo.dentro(c[None, :], h)[0] for h in huellas_tribuna):
                continue
            m.caja(c, fach["u"], 4.5, 4.5, 0.0, 3.0, "TechoBlanco")
            esquinas = [c + fach["u"] * a_ * 4.8 + fach["n"] * b_ * 4.8 for a_, b_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for q in range(4):
                m.cara([(*esquinas[q], 3.0), (*esquinas[(q + 1) % 4], 3.0), (*c, 5.6)], "TechoBlanco")
            n_carpas += 1
        m.crear(detalles)
    informe.update({"torres_de_luz": n_torres, "casetas": n_casetas, "puentes": len(puentes), "carpas": n_carpas})

    # ---- gente: aficionados con pancartas detrás del muro y mecánicos frente a los fosos
    gente = _coleccion("Entorno_Gente")
    rng_g = np.random.default_rng(11)
    ropa = ("Podio", "TechoBlanco", "AsientoVerde", "AsientoAmarillo", "AsientoAzul", "Naranja")
    m = Malla("Aficionados")
    n_grupos, ultimo, signo = 0, -1e9, 1.0
    lineas = {sg: sitio(sg, 8.5, margen=3.0) for sg in (1.0, -1.0)}
    vista_aficionados = None
    for i in range(len(centro)):
        if s[i] - ultimo < 110.0 or not lineas[signo][1][i]:
            continue
        c = lineas[signo][0][i]
        mira = -nor[i] * signo                                            # de frente a la pista
        pies = [c + tang[i] * (q - 2.5) * 0.95 + mira * rng_g.uniform(-0.3, 0.3) for q in range(6)]
        for q, pie in enumerate(pies):
            m.persona(pie, mira, ropa[int(rng_g.integers(len(ropa)))], brazos=q in (1, 4))
        a, b = pies[1] + mira * 0.25, pies[4] + mira * 0.25               # dos de ellos sostienen la pancarta
        for p_ in (a, b):
            m.caja(p_, tang[i], 0.03, 0.03, 1.5, 3.1, "Acero")
        m.pancarta(a, b, 2.2, 3.1, n_grupos, hacia=centro[i])
        if vista_aficionados is None and s[i] > 150.0:
            vista_aficionados = (centro[i] - tang[i] * 9.0, c)
        ultimo, signo, n_grupos = s[i], -signo, n_grupos + 1
    m.crear(gente)
    n_pits = 0
    vista_pits = None
    if paddock is not None:
        m = Malla("GentePits")
        equipos = ("Podio", "AsientoAzul", "Naranja", "AsientoVerde", "AsientoAmarillo", "TechoBlanco", "Tecpro", "AsientoMarino", "Acero", "AsientoCafe")
        for k in range(fach["n_fosos"]):
            c = fach["p0"] + fach["u"] * (fach["margen"] + (k + 0.5) * fach["ancho"])
            equipo = equipos[(k // 2) % len(equipos)]                     # dos fosos por equipo
            for du, dn in ((-1.3, 2.6), (1.0, 3.4), (-0.2, 4.6)):         # tres mecánicos frente a cada foso
                m.persona(c + fach["u"] * du + fach["n"] * dn, fach["n"], equipo)
                n_pits += 1
            i = int(np.hypot(*(centro - c).T).argmin())
            if k % 2 == 0 and en_recta[i]:                                # un ingeniero en el muro de pits
                m.persona(centro[i] + nor[i] * lado_pits * (w_pits[i] + 2.8), -nor[i] * lado_pits, equipo)
                n_pits += 1
        m.crear(gente)
        medio_f = 0.5 * (fach["p0"] + fach["p1"])
        vista_pits = (medio_f - fach["u"] * 22.0 + fach["n"] * 13.0, medio_f + fach["n"] * 3.0)
    informe.update({"grupos_de_aficionados": n_grupos, "personas_en_pits": n_pits})

    # ---- en el aire: dirigible, helicóptero, globos y nubes (el visor hace volar a los dos primeros)
    aire = _coleccion("Entorno_Aire")
    c_pista = 0.5 * (centro.min(0) + centro.max(0))       # el mismo centro que calcula el visor
    tam = float((centro.max(0) - centro.min(0)).max())
    norte = np.array([0.0, 1.0])
    m = Malla("Dirigible")
    c = np.array([c_pista[0] + 0.28 * tam, c_pista[1], 150.0])
    a_, b_ = 30.0, 8.5
    m.elipsoide(c, (a_, b_, b_), norte, lambda i, k: "AsientoVerde" if i < 2 else ("Tecpro" if i >= 8 else "TechoBlanco"), seg=12, anillos=10)
    d3, l3, z3 = np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    for o in (z3, -z3, l3, -l3):                                                   # aletas de cola
        m.cara([c - d3 * 18 + o * 6.0, c - d3 * 27 + o * 12.0, c - d3 * 31 + o * 12.0, c - d3 * 29.5 + o * 2.5], "AsientoVerde")
    m.caja(c[:2], norte, 5.0, 1.4, c[2] - b_ - 2.4, c[2] - b_ + 0.4, "ConcretoOscuro", fondo=True)   # góndola
    ts, zs = np.linspace(-0.62 * a_, 0.62 * a_, 9), np.linspace(-0.42 * b_, 0.42 * b_, 4)
    for lado in (1.0, -1.0):                                                       # letrero curvo en cada costado
        def punto(t_, z_):
            return c + d3 * t_ + l3 * lado * (b_ * math.sqrt(max(0.0, 1 - (t_ / a_) ** 2 - (z_ / b_) ** 2)) + 0.15) + z3 * z_
        for q in range(len(ts) - 1):
            for r in range(len(zs) - 1):
                u0, u1 = q / (len(ts) - 1), (q + 1) / (len(ts) - 1)
                if lado > 0:                                                        # se lee de izquierda a derecha desde cada lado
                    u0, u1 = 1 - u0, 1 - u1
                v0, v1 = r / (len(zs) - 1), (r + 1) / (len(zs) - 1)
                m.cara([punto(ts[q], zs[r]), punto(ts[q + 1], zs[r]), punto(ts[q + 1], zs[r + 1]), punto(ts[q], zs[r + 1])],
                       "Dirigible", uv=[(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    m.crear(aire)
    m = Malla("Helicoptero")
    c = np.array([c_pista[0] - 0.16 * tam, c_pista[1], 85.0])
    m.elipsoide(c, (3.4, 1.1, 1.2), norte, "Podio", seg=8, anillos=6)
    m.caja(c[:2] - norte * 5.2, norte, 2.6, 0.18, c[2] - 0.1, c[2] + 0.35, "Podio", fondo=True)          # cola
    m.cara([c - d3 * 7.6 - z3 * 0.2, c - d3 * 8.2 + z3 * 1.6, c - d3 * 7.4 + z3 * 1.6, c - d3 * 6.8 - z3 * 0.2], "TechoBlanco")
    m.caja(c[:2], norte, 0.1, 0.1, c[2] + 1.1, c[2] + 1.6, "Acero")
    m.cara([(c[0] + 5.2 * math.cos(q * math.pi / 4), c[1] + 5.2 * math.sin(q * math.pi / 4), c[2] + 1.6) for q in range(8)], "Malla")   # disco del rotor
    for q in (-0.9, 0.9):
        m.caja(c[:2] + np.array([q, 0.0]), norte, 2.2, 0.06, c[2] - 1.75, c[2] - 1.65, "Acero", fondo=True)
    m.crear(aire)
    franjas = ("AsientoVerde", "TechoBlanco", "Tecpro")
    for q, (fr, alto) in enumerate(((0.18, 75.0), (0.50, 115.0), (0.82, 90.0))):   # globos con los colores de México
        _, i, _ = geo.interpolar(centro, s, largo, fr * largo)
        m = Malla(f"Globo_{q + 1}")
        c = np.array([*(centro[i] + nor[i] * 140.0), alto])
        m.revolucion(c, [(0.0, 1.4), (4.0, 5.0), (9.0, 8.6), (14.0, 9.6), (18.5, 8.2), (21.5, 4.5), (22.6, 0.4)], lambda k: franjas[k % 3], seg=12)
        m.caja(c[:2], norte, 0.8, 0.8, alto - 3.2, alto - 1.8, "Tronco", fondo=True)
        for ex, ey in ((-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8)):          # cuerdas
            m.cara([(c[0] + ex, c[1] + ey, alto - 1.8), (c[0] + ex * 1.05, c[1] + ey * 1.05, alto - 1.8), (c[0] + ex * 1.6, c[1] + ey * 1.6, alto + 0.6)], "Acero")
        m.crear(aire)
    m = Malla("Nubes")
    rng_n = np.random.default_rng(7)
    for _ in range(14):
        ang, rad = rng_n.uniform(0, 2 * math.pi), rng_n.uniform(0.45, 0.9) * tam + 250.0
        c = np.array([c_pista[0] + rad * math.cos(ang), c_pista[1] + rad * math.sin(ang), rng_n.uniform(330.0, 480.0)])
        for _ in range(3):                                                           # cada nube: tres bultos
            m.elipsoide(c + rng_n.uniform(-1, 1, 3) * (30.0, 18.0, 4.0), (rng_n.uniform(35, 60), rng_n.uniform(20, 32), rng_n.uniform(6, 10)),
                        (math.cos(ang), math.sin(ang)), "Nube", seg=7, anillos=5)
    m.crear(aire)

    # ---- árboles low-poly
    veg = _coleccion("Entorno_Vegetacion")
    rng = np.random.default_rng(2026)
    arboles = [e["puntos"][0] for _, e in por_cat.get("arbol", [])]
    alturas = [e["altura_m"] or 9.0 for _, e in por_cat.get("arbol", [])]
    n_osm = len(arboles)
    for _, e in por_cat.get("bosque", []):      # bosques: árboles al azar dentro del contorno (1 cada ~300 m²)
        p = limpiar(e["puntos"])
        if p is None:
            continue
        (xa, ya), (xb, yb) = p.min(0), p.max(0)
        cand = np.column_stack([rng.uniform(xa, xb, int((xb - xa) * (yb - ya) / 300) + 1), rng.uniform(ya, yb, int((xb - xa) * (yb - ya) / 300) + 1)])
        cand = cand[geo.dentro(cand, p)]
        arboles += cand.tolist(); alturas += rng.uniform(10.0, 18.0, len(cand)).tolist()
    arboles, alturas = np.array(arboles, float).reshape(-1, 2), np.array(alturas, float)
    d_arb = np.array([np.hypot(*(centro - q).T).min() for q in arboles]) if len(arboles) else np.zeros(0)
    ok = (d_arb > 16.0)
    if c_palacio is not None:
        ok &= (d_arb < RADIO_ENTORNO_M) | (np.hypot(*(arboles - c_palacio).T) < RADIO_PALACIO_M)
    else:
        ok &= d_arb < RADIO_ENTORNO_M
    arboles, alturas, d_arb = arboles[ok], alturas[ok], d_arb[ok]
    if len(arboles) > MAX_ARBOLES:               # se conservan los más cercanos a la pista
        cerca_primero = np.argsort(d_arb)[:MAX_ARBOLES]
        arboles, alturas = arboles[cerca_primero], alturas[cerca_primero]
    ok = d_pit(arboles) > 9.0 if len(arboles) else np.zeros(0, bool)
    for h in huellas_tribuna + huellas_fijas:
        ok &= ~geo.dentro(arboles, h)
    # ni sobre la escapatoria ni pegados al muro (los del bosque caen donde sea)
    for izquierda, w in ((True, w_izq), (False, w_der)):
        from construir_pista import ancho_escapatoria as _esc
        limite = (w + _esc(centro, csv, izquierda) + 7.0)
        cercano = np.array([np.hypot(*(centro - q).T).argmin() for q in arboles], int) if len(arboles) else np.zeros(0, int)
        lado = np.einsum("ij,ij->i", arboles - centro[cercano], nor[cercano]) > 0
        ok &= ~((lado == izquierda) & (np.hypot(*(arboles - centro[cercano]).T) < limite[cercano]))
    arboles, alturas = arboles[ok], alturas[ok]
    informe["arboles_de_osm"] = n_osm
    alturas = alturas * rng.uniform(0.75, 1.3, len(arboles))
    tipo = rng.integers(0, 3, len(arboles))
    giro = rng.uniform(0, 2 * math.pi, len(arboles))
    m = Malla("Arboles")
    for (x, y), h, tp, g in zip(arboles, alturas, tipo, giro):
        radio = h * (0.20 if tp == 2 else 0.33)
        base = 0.28 * h
        tr = 0.03 * h + 0.12
        anillo_t = [(x + tr * math.cos(g + a), y + tr * math.sin(g + a)) for a in (0, 2.09, 4.19)]
        for q in range(3):
            m.pared(anillo_t[q], anillo_t[(q + 1) % 3], 0.0, base + 0.1 * h, "Tronco")
        lados = 6
        anillo = [(x + radio * math.cos(g + 2 * math.pi * q / lados), y + radio * math.sin(g + 2 * math.pi * q / lados), base + (0.25 if tp == 2 else 0.32) * h) for q in range(lados)]
        mat = f"Follaje{1 + tp}"
        for q in range(lados):
            a, b = anillo[q], anillo[(q + 1) % lados]
            m.cara([(x, y, base), b, a], mat)
            m.cara([a, b, (x, y, h)], mat)
    m.crear(veg)
    informe["arboles"] = int(len(arboles))

    # ---- luz y cámaras de presentación
    luz = _coleccion("Entorno_Luz")
    sol = bpy.data.lights.new("Sol", "SUN"); sol.energy = 2.6; sol.color = (1.0, 0.94, 0.84); sol.angle = math.radians(2.0)
    ob = bpy.data.objects.new("Sol", sol); luz.objects.link(ob)
    ob.rotation_euler = (math.radians(52), 0.0, math.radians(-125))      # luz de tarde, desde el poniente
    mundo = bpy.context.scene.world or bpy.data.worlds.new("Mundo")
    bpy.context.scene.world = mundo
    mundo.color = (0.55, 0.68, 0.85)
    try:
        mundo.use_nodes = True
    except Exception:
        pass
    fondo = next((nd for nd in mundo.node_tree.nodes if nd.type == "BACKGROUND"), None) if mundo.node_tree else None
    if fondo:
        fondo.inputs[0].default_value = (0.50, 0.66, 0.90, 1.0); fondo.inputs[1].default_value = 0.6

    def camara(nombre, pos, mira, lente=28.0):  # noqa: E306
        cam = bpy.data.cameras.new(nombre); cam.lens = lente; cam.clip_start, cam.clip_end = 1.0, 8000.0
        ob = bpy.data.objects.new(nombre, cam); luz.objects.link(ob)
        ob.location = pos
        ob.rotation_euler = (Vector(mira) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
        return ob

    camara("Cam_Dirigible", (c_pista[0] + 0.28 * tam + 95.0, c_pista[1] - 70.0, 185.0), (c_pista[0] + 0.28 * tam - 60.0, c_pista[1] + 30.0, 95.0), 26.0)
    if vista_aficionados is not None:
        camara("Cam_Aficionados", (*vista_aficionados[0], 2.4), (*vista_aficionados[1], 2.2), 26.0)
    if vista_pits is not None:
        camara("Cam_GentePits", (*vista_pits[0], 3.2), (*vista_pits[1], 1.6), 24.0)
    if mexico:
        camara("Cam_Recta", (-215.0, 36.0, 13.0), (120.0, -14.0, 6.0), 22.0)
        camara("Cam_Pits", (-30.0, 44.0, 16.0), (-125.0, -22.0, 6.0), 22.0)
        camara("Cam_TribunaPrincipal", (-60.0, -8.0, 14.0), (-110.0, 62.0, 10.0), 20.0)
        camara("Cam_EstadioGNP", (-130.0, -150.0, 120.0), (-335.0, -92.0, 5.0), 26.0)
        camara("Cam_Curvas1a3", (880.0, -20.0, 70.0), (990.0, -200.0, 0.0), 24.0)
        camara("Cam_Aerea", (1150.0, -1450.0, 800.0), (270.0, -440.0, 0.0), 30.0)
        camara("Cam_Palacio", (-520.0, 120.0, 90.0), (-720.0, -80.0, 20.0), 30.0)
        camara("Cam_Piloto", (300.0, -38.5, 1.6), (700.0, -96.0, 3.0), 30.0)      # a ras de pista, en la recta
        camara("Cam_Horquilla", (700.0, -905.0, 22.0), (640.0, -975.0, 2.0), 24.0)
    else:   # cámaras generales, calculadas del trazado
        cx, cy = 0.5 * (centro.min(0) + centro.max(0))
        tam = float((centro.max(0) - centro.min(0)).max())
        camara("Cam_Aerea", (cx + 0.55 * tam, cy - 0.95 * tam, 0.6 * tam), (cx, cy, 0.0), 30.0)
        p_, i_, _ = geo.interpolar(centro, s, largo, largo - 120.0)
        q_, _, _ = geo.interpolar(centro, s, largo, 160.0)
        camara("Cam_Recta", (*(p_ + nor[i_] * 3.0), 14.0), (*q_, 5.0), 22.0)
        p_, i_, _ = geo.interpolar(centro, s, largo, 40.0)
        q_, _, _ = geo.interpolar(centro, s, largo, 300.0)
        camara("Cam_Piloto", (*p_, 1.6), (*q_, 3.0), 30.0)
        en_curva_, _k = zonas_de_curva(centro)
        i_c = int(np.flatnonzero(en_curva_)[0]) if en_curva_.any() else 0
        lado_ext = 1.0 if _k[i_c] < 0 else -1.0
        camara("Cam_Curva1", (*(centro[i_c - 12] + nor[i_c - 12] * lado_ext * -18.0), 16.0), (*centro[(i_c + 8) % len(centro)], 2.0), 24.0)
    return informe


if __name__ == "__main__":
    print(construir())
