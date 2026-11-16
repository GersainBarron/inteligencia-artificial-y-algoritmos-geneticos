"""Pruebas de la geometría de pistas y del archivo exportado desde Blender."""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "blender"))
sys.path.insert(0, str(RAIZ / "scripts"))
import pista_geom as geo  # noqa: E402
from validar_pista import validar_archivo  # noqa: E402


def test_trazados_tum_validos():
    """Las cuatro pistas del protocolo dan bordes y checkpoints válidos."""
    for nombre in ("MexicoCity", "Spa", "Monza", "Budapest"):
        c, w_der, w_izq = geo.leer_tum(RAIZ / "pistas" / "fuente_tum" / f"{nombre}.csv")
        izq, der = geo.bordes(c, w_der, w_izq)
        r = geo.validar(c, izq, der, geo.checkpoints(c, izq, der, 50))
        assert r["fallas"] == [], (nombre, r["fallas"])


def test_validacion_detecta_pista_rota():
    """Control negativo: una pista con los bordes intercambiados en un tramo debe fallar."""
    c, w_der, w_izq = geo.leer_tum(RAIZ / "pistas" / "fuente_tum" / "MexicoCity.csv")
    izq, der = geo.bordes(c, w_der, w_izq)
    izq2, der2 = izq.copy(), der.copy()
    izq2[100:120], der2[100:120] = der[100:120], izq[100:120]
    assert geo.validar(c, izq2, der2, geo.checkpoints(c, izq2, der2, 50))["fallas"]


def test_pista_mexico_exportada():
    _, r = validar_archivo(RAIZ / "pistas" / "mexico" / "pista.json")
    assert r["fallas"] == [], r["fallas"]
    assert r["sentido"] == "horario"
    assert abs(r["ancho_medio_m"] - 12.249) < 0.01
