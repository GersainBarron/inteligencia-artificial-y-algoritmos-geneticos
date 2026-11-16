"""Alinea el entorno de OpenStreetMap de un circuito con su trazado de TUM.

El trazado de TUM está en metros locales; OpenStreetMap en latitud/longitud.
1. Convierte OSM a metros (proyección local alrededor del autódromo).
2. Ajusta una transformación rígida (rotación + traslación) que lleva la línea
   central de TUM sobre la pista de OSM (highway=raceway), con ICP.
3. Expresa edificios, tribunas, estadios, árboles y áreas verdes en el marco
   de TUM, que es el que usarán el simulador, Blender y el visor.

Salidas (en pistas/<circuito>/):
    entorno.json      elementos del entorno en coordenadas de la pista
    alineacion.png    vista de control: pista sobre el entorno

Uso:
    conda activate evocarros
    python scripts/alinear_osm.py               # México
    python scripts/alinear_osm.py spa           # spa | monza | hungaroring (ver circuitos.py)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuitos import DIR_OSM, circuito  # noqa: E402

# Valores de México; `main` los cambia si se pide otro circuito.
NOMBRE, CSV = "Autódromo Hermanos Rodríguez", "MexicoCity.csv"
OSM = Path(r"D:\redes neuronales\osm\autodromo_osm.json")
TUM = RAIZ / "pistas" / "fuente_tum" / "MexicoCity.csv"
SALIDA = RAIZ / "pistas" / "mexico"

LAT0, LON0 = 19.40611, -99.09250  # centro del autódromo (Wikipedia)
R_TIERRA = 6_371_000.0
RADIO_ENTORNO = 1200.0  # metros alrededor de la pista que se conservan


def a_metros(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    x = np.radians(lon - LON0) * R_TIERRA * math.cos(math.radians(LAT0))
    y = np.radians(lat - LAT0) * R_TIERRA
    return np.column_stack([x, y])


def geometria(e: dict) -> np.ndarray | None:
    if "geometry" in e:
        g = e["geometry"]
        return a_metros(np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g]))
    if e.get("type") == "node":
        return a_metros(np.array([e["lat"]]), np.array([e["lon"]]))
    if e.get("type") == "relation" and "members" in e:
        exteriores = [m for m in e["members"] if m.get("role") == "outer" and "geometry" in m]
        if exteriores:
            g = exteriores[0]["geometry"]
            return a_metros(np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g]))
    return None


def remuestrear(xy: np.ndarray, paso: float) -> np.ndarray:
    seg = np.diff(xy, axis=0)
    long_seg = np.hypot(*seg.T)
    s = np.concatenate([[0], np.cumsum(long_seg)])
    nuevos = np.arange(0, s[-1], paso)
    return np.column_stack([np.interp(nuevos, s, xy[:, 0]), np.interp(nuevos, s, xy[:, 1])])


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def icp(origen: np.ndarray, destino: np.ndarray, theta0: float, iter_max: int = 80):
    """Transformación rígida que lleva origen sobre destino. Devuelve R, t, error RMS."""
    arbol = cKDTree(destino)
    R = rot(theta0)
    t = destino.mean(0) - origen.mean(0) @ R.T
    for _ in range(iter_max):
        movido = origen @ R.T + t
        dist, idx = arbol.query(movido)
        pares = destino[idx]
        mo, md = origen.mean(0), pares.mean(0)
        H = (origen - mo).T @ (pares - md)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        t = md - mo @ R.T
    dist, _ = arbol.query(origen @ R.T + t)
    return R, t, float(np.sqrt(np.mean(dist**2))), float(np.percentile(dist, 95))


BOSQUES = False   # México conserva sus categorías originales; los demás circuitos separan los bosques


def categoria(tags: dict) -> str | None:
    b, l, n, lu, a = (tags.get(k) for k in ("building", "leisure", "natural", "landuse", "amenity"))
    if tags.get("highway") == "raceway":
        return "pista_osm"
    if b == "grandstand" or l == "bleachers":
        return "tribuna"
    if l == "stadium":
        return "estadio"
    if b:
        return "edificio"
    if n == "tree":
        return "arbol"
    if (n == "wood" or lu == "forest") and BOSQUES:
        return "bosque"
    if n in ("tree_row", "wood") or l in ("park", "garden") or lu in ("grass", "greenfield", "recreation_ground"):
        return "verde"
    if a == "parking":
        return "estacionamiento"
    if l == "pitch":
        return "cancha"
    if "waterway" in tags or n == "water":
        return "agua"
    return None


def altura(tags: dict, cat: str) -> float:
    for clave in ("height", "building:height"):
        if clave in tags:
            try:
                return float(str(tags[clave]).replace("m", "").strip())
            except ValueError:
                pass
    if "building:levels" in tags:
        try:
            return 3.2 * float(tags["building:levels"])
        except ValueError:
            pass
    return {"tribuna": 12.0, "estadio": 0.0, "edificio": 8.0, "arbol": 9.0}.get(cat, 0.0)


def main() -> int:
    global OSM, TUM, SALIDA, LAT0, LON0, NOMBRE, CSV, BOSQUES, RADIO_ENTORNO
    clave, c = circuito(sys.argv)
    if clave != "mexico":
        OSM, TUM, SALIDA = DIR_OSM / c["osm"], RAIZ / "pistas" / "fuente_tum" / c["csv"], RAIZ / "pistas" / clave
        LAT0, LON0, NOMBRE, CSV, BOSQUES = c["lat0"], c["lon0"], c["nombre"], c["csv"], True
        RADIO_ENTORNO = 450.0   # solo se modela lo cercano a la pista
    osm = json.loads(OSM.read_text(encoding="utf-8"))["elements"]
    d = np.loadtxt(TUM, delimiter=",", comments="#")
    centro_tum = d[:, :2]

    # Pista de OSM: todos los tramos raceway, remuestreados cada 2 m.
    tramos = [geometria(e) for e in osm if e.get("tags", {}).get("highway") == "raceway"]
    pista_osm = np.vstack([remuestrear(g, 2.0) for g in tramos if g is not None and len(g) > 1])

    # Línea central de TUM remuestreada cada 4 m (cerrando el circuito).
    origen = remuestrear(np.vstack([centro_tum, centro_tum[:1]]), 4.0)

    # Probar varias rotaciones iniciales y quedarse con la de menor error.
    mejor = None
    for grados in range(0, 360, 10):
        R, t, rms, p95 = icp(origen, pista_osm, math.radians(grados))
        if mejor is None or rms < mejor[2]:
            mejor = (R, t, rms, p95)
    R, t, rms, p95 = mejor
    angulo = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    print(f"Alineación TUM -> OSM: rotación {angulo:.2f} grados, error RMS {rms:.2f} m, p95 {p95:.2f} m")

    # Inversa: OSM (metros) -> marco TUM.
    def a_tum(xy: np.ndarray) -> np.ndarray:
        return (xy - t) @ R

    arbol_pista = cKDTree(centro_tum)
    elementos = []
    for e in osm:
        tags = e.get("tags", {})
        cat = categoria(tags)
        g = geometria(e)
        if cat is None or g is None:
            continue
        g = a_tum(g)
        if arbol_pista.query(g.mean(0))[0] > RADIO_ENTORNO:
            continue
        elementos.append({
            "categoria": cat,
            "nombre": tags.get("name"),
            "altura_m": altura(tags, cat),
            "puntos": np.round(g, 2).tolist(),
        })

    SALIDA.mkdir(parents=True, exist_ok=True)
    meta = {
        "fuente": "© colaboradores de OpenStreetMap (ODbL); trazado TUM racetrack-database (LGPL-3.0)",
        "marco": f"coordenadas en metros del archivo TUM {CSV}",
        "alineacion": {"rotacion_grados": angulo, "rms_m": rms, "p95_m": p95,
                        "R": R.tolist(), "t": t.tolist(), "lat0": LAT0, "lon0": LON0},
        "elementos": elementos,
    }
    (SALIDA / "entorno.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    conteo = {}
    for el in elementos:
        conteo[el["categoria"]] = conteo.get(el["categoria"], 0) + 1
    print("Elementos en el entorno:", dict(sorted(conteo.items(), key=lambda kv: -kv[1])))
    print("Con nombre:", sorted({el["nombre"] for el in elementos
                                 if el["nombre"] and el["categoria"] in ("tribuna", "estadio")}))

    # Vista de control.
    colores = {"bosque": "#6fae6a", "edificio": "#c9c3b8", "tribuna": "#d9534f", "estadio": "#f0ad4e", "verde": "#a8d5a2",
               "estacionamiento": "#e6e6e6", "cancha": "#cde8c5", "agua": "#9cc9e8"}
    fig, ax = plt.subplots(figsize=(14, 12))
    for cat in ("verde", "bosque", "cancha", "estacionamiento", "agua", "estadio", "edificio", "tribuna"):
        parches = [Polygon(el["puntos"], closed=True) for el in elementos
                   if el["categoria"] == cat and len(el["puntos"]) >= 3]
        if parches:
            ax.add_collection(PatchCollection(parches, facecolor=colores[cat], edgecolor="none",
                                              alpha=0.9 if cat == "tribuna" else 0.7))
    arboles = np.array([el["puntos"][0] for el in elementos if el["categoria"] == "arbol"])
    if len(arboles):
        ax.scatter(*arboles.T, s=2, color="#3c8d3c", alpha=0.6)
    for el in elementos:
        if el["categoria"] == "pista_osm":
            ax.plot(*np.array(el["puntos"]).T, color="#1f77b4", lw=0.8, alpha=0.7)
    ax.plot(*np.vstack([centro_tum, centro_tum[:1]]).T, color="black", lw=1.2, label="TUM (línea central)")
    for el in elementos:
        if el["nombre"] and el["categoria"] in ("estadio",):
            c = np.mean(el["puntos"], axis=0)
            ax.annotate(el["nombre"], c, fontsize=9, ha="center")
    m = np.vstack([centro_tum.min(0) - 250, centro_tum.max(0) + 250])
    ax.set_xlim(m[:, 0]); ax.set_ylim(m[:, 1]); ax.set_aspect("equal")
    ax.set_title(f"{NOMBRE}: TUM (negro) sobre OpenStreetMap (azul = pista OSM)\n"
                 f"error RMS {rms:.1f} m · tribunas en rojo, estadios en naranja", fontsize=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SALIDA / "alineacion.png", dpi=110)
    print(f"Guardado {SALIDA / 'entorno.json'} y alineacion.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
