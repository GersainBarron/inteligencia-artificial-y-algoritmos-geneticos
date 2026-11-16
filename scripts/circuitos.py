"""Datos de cada circuito para descargar y alinear su entorno de OpenStreetMap.

`bbox` es (sur, oeste, norte, este) en grados; `lat0`, `lon0` es el origen de la
proyección local (cerca del centro del circuito). Las coordenadas son
aproximadas: la alineación fina con el trazado de TUM la hace `alinear_osm.py`.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_OSM = Path(r"D:\redes neuronales\osm")

CIRCUITOS = {
    "mexico": {"nombre": "Autódromo Hermanos Rodríguez", "csv": "MexicoCity.csv", "osm": "autodromo_osm.json",
               "lat0": 19.40611, "lon0": -99.09250, "bbox": (19.394, -99.106, 19.418, -99.078)},
    "spa": {"nombre": "Spa-Francorchamps", "csv": "Spa.csv", "osm": "spa_osm.json",
            "lat0": 50.4372, "lon0": 5.9714, "bbox": (50.418, 5.948, 50.452, 5.992)},
    "monza": {"nombre": "Monza", "csv": "Monza.csv", "osm": "monza_osm.json",
              "lat0": 45.6205, "lon0": 9.2895, "bbox": (45.604, 9.270, 45.638, 9.306)},
    "hungaroring": {"nombre": "Hungaroring", "csv": "Budapest.csv", "osm": "hungaroring_osm.json",
                    "lat0": 47.5820, "lon0": 19.2510, "bbox": (47.571, 19.236, 47.593, 19.266)},
}


def circuito(argv) -> tuple[str, dict]:
    """Circuito pedido en la línea de comandos (por defecto, México)."""
    clave = argv[1].lower() if len(argv) > 1 else "mexico"
    if clave not in CIRCUITOS:
        raise SystemExit(f"Circuito desconocido: {clave}. Opciones: {', '.join(CIRCUITOS)}")
    return clave, CIRCUITOS[clave]
