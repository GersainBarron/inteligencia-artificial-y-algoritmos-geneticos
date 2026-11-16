"""Descarga de OpenStreetMap los elementos del entorno de un circuito.

Los datos de OpenStreetMap son libres (licencia ODbL): se pueden usar citando
"© colaboradores de OpenStreetMap". Sirven para ubicar edificios, tribunas,
estadio y áreas verdes alrededor de la pista en Blender y en el visor.

Uso:
    conda activate evocarros
    python scripts/descargar_osm.py             # México
    python scripts/descargar_osm.py spa         # spa | monza | hungaroring (ver circuitos.py)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuitos import DIR_OSM, circuito  # noqa: E402

# Servidores públicos de Overpass; si uno falla (saturación, error 5xx) se prueba el siguiente.
URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter")
CABECERAS = {"User-Agent": "evocarros-investigacion/0.1 (proyecto personal de neuroevolucion)"}


def consulta(bbox) -> str:
    b = ",".join(str(v) for v in bbox)   # sur, oeste, norte, este
    return f"""
[out:json][timeout:180];
(
  way["highway"="raceway"]({b});
  way["building"]({b});
  relation["building"]({b});
  way["leisure"]({b});
  relation["leisure"]({b});
  way["landuse"]({b});
  relation["landuse"]({b});
  way["natural"]({b});
  relation["natural"]({b});
  way["amenity"="parking"]({b});
  way["waterway"]({b});
  node["natural"="tree"]({b});
);
out geom;
"""


def tipo(etiquetas: dict) -> str:
    for clave in ("highway", "building", "leisure", "landuse", "natural", "amenity", "waterway"):
        if clave in etiquetas:
            return f"{clave}={etiquetas[clave]}"
    return "otro"


def main() -> int:
    _, c = circuito(sys.argv)
    SALIDA = DIR_OSM / c["osm"]
    datos = None
    for intento in range(6):
        url = URLS[intento % len(URLS)]
        try:
            resp = requests.post(url, data={"data": consulta(c["bbox"])}, headers=CABECERAS, timeout=240)
            resp.raise_for_status()
            datos = resp.json()
            break
        except (requests.RequestException, ValueError) as error:
            print(f"  intento {intento + 1} con {url}: {error}")
            time.sleep(10)
    if datos is None:
        print("No se pudo descargar; intentar más tarde.")
        return 1
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")

    elementos = datos.get("elements", [])
    print(f"Guardado {SALIDA} con {len(elementos)} elementos\n")
    for nombre, n in Counter(tipo(e.get("tags", {})) for e in elementos).most_common(25):
        print(f"{n:5d}  {nombre}")
    print("\nElementos con nombre:")
    nombres = sorted({
        f"{e['tags']['name']}  [{tipo(e['tags'])}]"
        for e in elementos if "name" in e.get("tags", {})
    })
    for n in nombres:
        print("  " + n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
