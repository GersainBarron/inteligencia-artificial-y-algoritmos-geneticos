"""Tiempo de vuelta ideal de una pista con los límites físicos del carro (cota inferior).

Perfil de velocidad clásico de dos pasadas sobre la línea central:
v(s) <= min(v_max, sqrt(a_lat * radio)), aceleración <= a_acel, frenado <= a_freno.
Es una cota optimista: sigue la línea central (no la trazada ideal sería más rápida,
pero el carro real no es perfecto) y sirve para calibrar T_max y el tiempo inicial.

Uso: python scripts/perfil_velocidad.py [pistas/mexico/pista.json ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
V_MAX, A_ACEL, A_FRENO, A_LAT = 50.0, 6.0, 12.0, 1.5 * 9.81


def radios(c):
    a, b, d = np.roll(c, 1, 0), c, np.roll(c, -1, 0)
    ab, bd, ad = (np.linalg.norm(b - a, axis=1), np.linalg.norm(d - b, axis=1), np.linalg.norm(d - a, axis=1))
    cruz = np.abs((b - a)[:, 0] * (d - a)[:, 1] - (b - a)[:, 1] * (d - a)[:, 0])
    return np.where(cruz > 1e-9, ab * bd * ad / (2 * np.maximum(cruz, 1e-9)), np.inf)


def perfil(ruta):
    p = json.loads(Path(ruta).read_text(encoding="utf-8"))
    c = np.asarray(p["linea_central"] if "linea_central" in p else p["centro"], float)
    if c.shape[1] > 2:
        c = c[:, :2]
    ds = np.linalg.norm(np.roll(c, -1, 0) - c, axis=1)
    r = radios(c)
    # radio suavizado (ventana de 3) para no dejarse llevar por ruido de un punto
    r = np.minimum.reduce([r, np.maximum(np.roll(r, 1), r), np.maximum(np.roll(r, -1), r)])
    v = np.minimum(V_MAX, np.sqrt(A_LAT * r))
    n = len(c)
    for _ in range(3):  # vuelta cerrada: varias pasadas para que se propague
        for i in range(1, 2 * n):
            j, k = i % n, (i - 1) % n
            v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2 * A_ACEL * ds[k]))
        for i in range(2 * n, 0, -1):
            j, k = (i - 1) % n, i % n
            v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2 * A_FRENO * ds[j]))
    t_vuelta = float(np.sum(ds / np.maximum(0.5 * (v + np.roll(v, -1)), 0.1)))
    # salida desde parado: tiempo hasta 1/5 de vuelta (primera zona de frenado en todas)
    vp = v.copy(); vp[0] = 0.0
    for i in range(1, n):
        vp[i] = min(v[i], np.sqrt(vp[i - 1] ** 2 + 2 * A_ACEL * ds[i - 1]))
    s = np.concatenate([[0], np.cumsum(ds)])
    t = np.concatenate([[0], np.cumsum(ds / np.maximum(0.5 * (vp + np.roll(vp, -1)), 0.1))])
    return {"pista": p.get("nombre", ruta), "longitud_m": round(float(s[-1]), 1),
            "t_vuelta_ideal_s": round(t_vuelta, 1), "v_media_kmh": round(3.6 * s[-1] / t_vuelta, 1),
            "v_min_kmh": round(3.6 * float(v.min()), 1),
            "t_hasta_500m_s": round(float(np.interp(500, s, t)), 1),
            "t_hasta_1000m_s": round(float(np.interp(1000, s, t)), 1),
            "t_hasta_1500m_s": round(float(np.interp(1500, s, t)), 1)}


if __name__ == "__main__":
    rutas = sys.argv[1:] or [str(RAIZ / f"pistas/{x}/pista.json") for x in ("mexico", "spa", "monza", "hungaroring")]
    for r in rutas:
        print(json.dumps(perfil(r), ensure_ascii=False))
