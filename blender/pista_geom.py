"""Geometría de una pista a partir de un trazado de TUM racetrack-database.

Módulo sin dependencia de Blender (solo NumPy): lo usan los scripts de Blender
(`construir_pista.py`, `exportar_pista.py`) y la validación independiente
(`scripts/validar_pista.py`). Así la misma definición geométrica se aplica al
modelo 3D, al archivo `pista.json` y a las pruebas.

Marco de referencia: metros del archivo de TUM (x, y), el mismo de
`pistas/mexico/entorno.json`. En Blender: X = x, Y = y, Z = altura.

Convención (la de TUM): los puntos van en el sentido de circulación;
`w_tr_right_m` y `w_tr_left_m` son la distancia de la línea central a cada
borde, medida sobre la normal a la línea central.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ESQUEMA = "evocarros.pista/1"
TOL_EXTREMO_M = 0.01  # tolerancia de contacto entre checkpoints y bordes


# ---------------------------------------------------------------- lectura
def leer_tum(ruta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve línea central (n, 2), ancho derecho (n,) y ancho izquierdo (n,)."""
    d = np.loadtxt(str(ruta), delimiter=",", comments="#")
    centro, w_der, w_izq = d[:, :2], d[:, 2], d[:, 3]
    if np.hypot(*(centro[0] - centro[-1])) < 1e-6:  # punto de cierre duplicado
        centro, w_der, w_izq = centro[:-1], w_der[:-1], w_izq[:-1]
    return centro, w_der, w_izq


def sha256(ruta) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


# ------------------------------------------------- línea central y bordes
def tangentes(centro: np.ndarray) -> np.ndarray:
    """Tangente unitaria por diferencias centrales en una polilínea cerrada."""
    t = np.roll(centro, -1, axis=0) - np.roll(centro, 1, axis=0)
    return t / np.hypot(*t.T)[:, None]


def normales_izq(centro: np.ndarray) -> np.ndarray:
    """Normal unitaria hacia la izquierda del sentido de circulación."""
    t = tangentes(centro)
    return np.column_stack([-t[:, 1], t[:, 0]])


def bordes(centro, w_der, w_izq) -> tuple[np.ndarray, np.ndarray]:
    """Borde izquierdo y derecho: línea central desplazada sobre su normal."""
    n = normales_izq(centro)
    return centro + n * w_izq[:, None], centro - n * w_der[:, None]


def abscisa(poli: np.ndarray) -> tuple[np.ndarray, float]:
    """Abscisa curvilínea s de cada vértice y longitud total (polilínea cerrada)."""
    seg = np.hypot(*(np.roll(poli, -1, axis=0) - poli).T)
    return np.concatenate([[0.0], np.cumsum(seg)[:-1]]), float(seg.sum())


def area_con_signo(poli: np.ndarray) -> float:
    """Positiva = antihoraria; negativa = horaria."""
    x, y = poli.T
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def curvatura(centro: np.ndarray, suavizado: int = 5) -> np.ndarray:
    """Curvatura con signo (1/m) por el círculo que pasa por tres vértices
    consecutivos. Positiva = curva a la izquierda. Se suaviza con una media
    móvil circular de `suavizado` puntos para atenuar el ruido del muestreo."""
    a, b, c = np.roll(centro, 1, axis=0), centro, np.roll(centro, -1, axis=0)
    ab, bc, ac = b - a, c - b, c - a
    cruz = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    k = 2.0 * cruz / (np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ac.T))
    if suavizado > 1:
        m = suavizado // 2
        k = np.convolve(np.concatenate([k[-m:], k, k[:m]]), np.ones(suavizado) / suavizado, "valid")
    return k


# ------------------------------------------------------------ checkpoints
def interpolar(poli: np.ndarray, s: np.ndarray, largo: float, s_obj: float):
    """Punto de la polilínea cerrada en la abscisa s_obj. Devuelve (punto, i, u):
    el punto está en el segmento i -> i+1 a la fracción u."""
    s_obj = s_obj % largo
    i = int(np.searchsorted(s, s_obj, side="right") - 1)
    j = (i + 1) % len(poli)
    fin = s[j] if j else largo
    u = (s_obj - s[i]) / (fin - s[i])
    return poli[i] + u * (poli[j] - poli[i]), i, u


def checkpoints(centro, borde_izq, borde_der, n: int) -> list[dict]:
    """n líneas transversales equiespaciadas en longitud de arco, en orden.

    La línea k está en s = k·L/n (la 0 es la salida/meta). Sus extremos se
    interpolan con la misma fracción sobre los segmentos homólogos de cada
    borde, de modo que caen exactamente sobre las polilíneas de los bordes.
    """
    s, largo = abscisa(centro)
    m = len(centro)
    salida = []
    for k in range(n):
        s_k = k * largo / n
        c, i, u = interpolar(centro, s, largo, s_k)
        j = (i + 1) % m
        salida.append({
            "indice": k,
            "s_m": s_k,
            "centro": c,
            "izq": borde_izq[i] + u * (borde_izq[j] - borde_izq[i]),
            "der": borde_der[i] + u * (borde_der[j] - borde_der[i]),
        })
    return salida


# ------------------------------------------------------------- validación
def _segmentos(poli: np.ndarray, cerrada: bool = True) -> np.ndarray:
    b = np.roll(poli, -1, axis=0)
    seg = np.stack([poli, b], axis=1)
    return seg if cerrada else seg[:-1]


def cruces(seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Matriz booleana (na, nb): True si el segmento a_i y el b_j se cruzan en
    un punto interior de ambos (los contactos en los extremos no cuentan)."""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    a0, a1 = seg_a[:, None, 0], seg_a[:, None, 1]
    b0, b1 = seg_b[None, :, 0], seg_b[None, :, 1]
    eps = 1e-9
    d1, d2 = orient(a0, a1, b0), orient(a0, a1, b1)
    d3, d4 = orient(b0, b1, a0), orient(b0, b1, a1)
    return (d1 * d2 < -eps) & (d3 * d4 < -eps)


def autocruces(poli: np.ndarray) -> int:
    """Número de pares de segmentos no adyacentes de una polilínea cerrada que se cruzan."""
    seg = _segmentos(poli)
    m = cruces(seg, seg)
    n = len(seg)
    i, j = np.nonzero(m)
    lejos = (np.abs(i - j) > 1) & (np.abs(i - j) < n - 1)
    return int(np.count_nonzero(lejos) // 2)


def dist_a_polilinea(puntos: np.ndarray, poli: np.ndarray) -> np.ndarray:
    """Distancia mínima de cada punto a una polilínea cerrada."""
    a = poli
    ab = np.roll(poli, -1, axis=0) - a
    den = np.einsum("ij,ij->i", ab, ab)
    salida = np.empty(len(puntos))
    for k, p in enumerate(puntos):
        u = np.clip(np.einsum("ij,ij->i", p - a, ab) / den, 0.0, 1.0)
        salida[k] = np.hypot(*(a + u[:, None] * ab - p).T).min()
    return salida


def proyectar(p: np.ndarray, poli: np.ndarray) -> np.ndarray:
    """Punto de la polilínea cerrada más cercano a p."""
    ab = np.roll(poli, -1, axis=0) - poli
    u = np.clip(np.einsum("ij,ij->i", p - poli, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    q = poli + u[:, None] * ab
    return q[np.hypot(*(q - p).T).argmin()]


def dentro(puntos: np.ndarray, poli: np.ndarray) -> np.ndarray:
    """Punto en polígono (regla par-impar) para cada punto."""
    x, y = puntos[:, 0][:, None], puntos[:, 1][:, None]
    x0, y0 = poli[:, 0][None, :], poli[:, 1][None, :]
    x1, y1 = np.roll(poli[:, 0], -1)[None, :], np.roll(poli[:, 1], -1)[None, :]
    cruza = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (np.count_nonzero(cruza & (x < xc), axis=1) % 2) == 1


def validar(centro, borde_izq, borde_der, cps) -> dict:
    """Comprobaciones geométricas que debe cumplir una pista para el simulador.
    Devuelve un dict con cada medida y la lista `fallas` (vacía si todo pasa)."""
    r: dict = {"fallas": []}
    s, largo = abscisa(centro)
    r["longitud_m"] = largo
    r["sentido"] = "horario" if area_con_signo(centro) < 0 else "antihorario"
    r["autocruces_izq"] = autocruces(borde_izq)
    r["autocruces_der"] = autocruces(borde_der)
    r["autocruces_centro"] = autocruces(centro)
    r["cruces_entre_bordes"] = int(np.count_nonzero(cruces(_segmentos(borde_izq), _segmentos(borde_der))))
    # el exterior es el borde de mayor área; la línea central queda entre ambos
    ext, inte = (borde_izq, borde_der) if abs(area_con_signo(borde_izq)) > abs(area_con_signo(borde_der)) else (borde_der, borde_izq)
    r["borde_exterior"] = "izquierdo" if ext is borde_izq else "derecho"
    r["centro_fuera_de_pista"] = int(np.count_nonzero(~dentro(centro, ext) | dentro(centro, inte)))
    d_i, d_d = dist_a_polilinea(centro, borde_izq), dist_a_polilinea(centro, borde_der)
    r["holgura_min_izq_m"], r["holgura_min_der_m"] = float(d_i.min()), float(d_d.min())
    r["ancho_libre_min_m"] = float(dist_a_polilinea(borde_izq, borde_der).min())
    # checkpoints
    seg_cp = np.array([[c["izq"], c["der"]] for c in cps])
    r["n_checkpoints"] = len(cps)
    s_cp = np.array([c["s_m"] for c in cps])
    r["cp_en_orden"] = bool(np.all(np.diff(s_cp) > 0))
    paso = np.diff(np.concatenate([s_cp, [s_cp[0] + largo]]))
    r["cp_paso_min_m"], r["cp_paso_max_m"] = float(paso.min()), float(paso.max())
    # Los extremos tocan los bordes; para no contar ese contacto como cruce (los
    # vértices pasan por float32 en Blender y por el redondeo a mm del JSON) se
    # prueba el segmento recortado TOL_EXTREMO_M en cada punta.
    u = (seg_cp[:, 1] - seg_cp[:, 0]) / np.hypot(*(seg_cp[:, 1] - seg_cp[:, 0]).T)[:, None]
    seg_rec = np.stack([seg_cp[:, 0] + TOL_EXTREMO_M * u, seg_cp[:, 1] - TOL_EXTREMO_M * u], axis=1)
    r["cp_cruzan_bordes"] = int(np.count_nonzero(cruces(seg_rec, _segmentos(borde_izq))) + np.count_nonzero(cruces(seg_rec, _segmentos(borde_der))))
    m = cruces(seg_cp, seg_cp)
    r["cp_se_cruzan"] = int(np.count_nonzero(np.triu(m, 1)))
    r["cp_extremos_fuera_de_borde_m"] = float(max(
        dist_a_polilinea(seg_cp[:, 0], borde_izq).max(), dist_a_polilinea(seg_cp[:, 1], borde_der).max()))
    largo_cp = np.hypot(*(seg_cp[:, 0] - seg_cp[:, 1]).T)
    r["cp_largo_min_m"], r["cp_largo_max_m"] = float(largo_cp.min()), float(largo_cp.max())
    # ningún checkpoint debe cortar la línea central más de una vez (su centro
    # está sobre ella por construcción; un segundo corte indicaría que invade otro tramo)
    r["cp_cortes_extra_del_centro"] = int(np.count_nonzero(
        np.count_nonzero(cruces(seg_cp, _segmentos(centro)), axis=1) > 1))

    def exigir(cond, texto):
        if not cond:
            r["fallas"].append(texto)
    exigir(r["autocruces_izq"] == 0, "el borde izquierdo se cruza a sí mismo")
    exigir(r["autocruces_der"] == 0, "el borde derecho se cruza a sí mismo")
    exigir(r["autocruces_centro"] == 0, "la línea central se cruza a sí misma")
    exigir(r["cruces_entre_bordes"] == 0, "los bordes se cruzan entre sí")
    exigir(r["centro_fuera_de_pista"] == 0, "hay puntos de la línea central fuera de la pista")
    exigir(r["ancho_libre_min_m"] > 4.0, "la pista se estrecha a menos de 4 m")
    exigir(r["cp_en_orden"], "los checkpoints no están en orden")
    exigir(r["cp_cruzan_bordes"] == 0, "algún checkpoint atraviesa un borde")
    exigir(r["cp_se_cruzan"] == 0, "hay checkpoints que se cruzan entre sí")
    exigir(r["cp_extremos_fuera_de_borde_m"] < 5e-3, "extremos de checkpoints fuera de los bordes")
    exigir(r["cp_cortes_extra_del_centro"] == 0, "algún checkpoint corta la línea central más de una vez")
    return r
