"""Pista del simulador: carga `pista.json` y responde las consultas geométricas.

Fuente de verdad: `pistas/<nombre>/pista.json` (esquema `evocarros.pista/1`,
exportado desde Blender y validado con `scripts/validar_pista.py`).

Consultas que ofrece:
- `progreso_local(p, i)`: abscisa s (m) del punto p proyectado sobre la línea
  central, buscando solo cerca del índice i de la consulta anterior. La búsqueda
  local evita confundir tramos de la pista que pasan cerca entre sí.
- `rayos(p, rumbo, angulos, alcances)`: distancia al primer borde en cada rayo.
- `toca_borde(esquinas)`: si el contorno del carro corta algún borde.

Los bordes se guardan como segmentos y se indexan en una rejilla: cada celda
conoce los segmentos a menos de `margen` metros, así cada consulta solo revisa
unas decenas de segmentos en lugar de los ~1,700 de la pista. Los choques usan
otra rejilla con margen de 5 m, porque solo importan los bordes junto al carro.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CELDA_M = 25.0
MARGEN_CHOQUE_M = 5.0   # >= media diagonal del carro (2.2 m) + avance por paso (1.7 m)


def _como_xy(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, :2] if a.ndim == 2 and a.shape[1] > 2 else a


def _checkpoint(cp) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Acepta los formatos razonables de checkpoint: dict con izq/der o lista [[x,y],[x,y]]."""
    if isinstance(cp, dict):
        izq = cp.get("izq", cp.get("izquierdo", cp.get("a")))
        der = cp.get("der", cp.get("derecho", cp.get("b")))
        return np.asarray(izq, float)[:2], np.asarray(der, float)[:2], cp.get("s_m")
    return np.asarray(cp[0], float)[:2], np.asarray(cp[1], float)[:2], None


@dataclass
class Pista:
    nombre: str
    centro: np.ndarray            # (N, 2)
    s: np.ndarray                 # (N,) abscisa de cada punto, s[0] = 0
    longitud: float
    borde_izq: np.ndarray         # (N, 2)
    borde_der: np.ndarray         # (N, 2)
    cp_izq: np.ndarray            # (K, 2)
    cp_der: np.ndarray            # (K, 2)
    cp_s: np.ndarray              # (K,)
    margen: float = 110.0         # >= alcance máximo de un rayo + media diagonal del carro
    seg: np.ndarray = field(init=False, repr=False)       # (M, 4) x1 y1 x2 y2
    _rejilla: dict = field(init=False, repr=False)
    _origen: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        segs = []
        for b in (self.borde_izq, self.borde_der):
            segs.append(np.hstack([b, np.roll(b, -1, axis=0)]))
        self.seg = np.vstack(segs)
        self._indexar()
        d = np.diff(np.vstack([self.centro, self.centro[:1]]), axis=0)
        self._tramo_len = np.linalg.norm(d, axis=1)
        self._tramo_dir = d / np.maximum(self._tramo_len[:, None], 1e-12)

    # ------------------------------------------------------------ carga
    @classmethod
    def desde_json(cls, ruta: str | Path, margen: float = 110.0) -> "Pista":
        p = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls.desde_dict(p, margen)

    @classmethod
    def desde_dict(cls, p: dict, margen: float = 110.0) -> "Pista":
        centro = _como_xy(p["linea_central"])
        s = np.asarray(p["s_m"], float) if "s_m" in p else np.concatenate(
            [[0], np.cumsum(np.linalg.norm(np.diff(centro, axis=0), axis=1))])
        longitud = float(p.get("longitud_m") or s[-1] + np.linalg.norm(centro[0] - centro[-1]))
        izq, der, cs = [], [], []
        for cp in p["checkpoints"]:
            a, b, sc = _checkpoint(cp)
            izq.append(a); der.append(b); cs.append(sc)
        if any(x is None for x in cs):
            paso = longitud / len(cs)
            cs = [k * paso for k in range(len(cs))]
        return cls(nombre=p.get("nombre", ""), centro=centro, s=s, longitud=longitud,
                   borde_izq=_como_xy(p["borde_izquierdo"]), borde_der=_como_xy(p["borde_derecho"]),
                   cp_izq=np.asarray(izq), cp_der=np.asarray(der), cp_s=np.asarray(cs, float), margen=margen)

    # ------------------------------------------------------------ rejilla
    def _indexar(self):
        todos = np.vstack([self.borde_izq, self.borde_der])
        self._origen = todos.min(axis=0) - self.margen - CELDA_M
        self._rejilla = self._rejilla_con_margen(self.margen)        # rayos
        self._rejilla_choque = self._rejilla_con_margen(MARGEN_CHOQUE_M)
        self._vacio = np.zeros((0, 4))

    def _rejilla_con_margen(self, margen: float) -> dict:
        lo = np.minimum(self.seg[:, :2], self.seg[:, 2:]) - margen
        hi = np.maximum(self.seg[:, :2], self.seg[:, 2:]) + margen
        i0 = np.floor((lo - self._origen) / CELDA_M).astype(int)
        i1 = np.floor((hi - self._origen) / CELDA_M).astype(int)
        celdas: dict[tuple[int, int], list[int]] = {}
        for k in range(len(self.seg)):
            for cx in range(i0[k, 0], i1[k, 0] + 1):
                for cy in range(i0[k, 1], i1[k, 1] + 1):
                    celdas.setdefault((cx, cy), []).append(k)
        return {c: self.seg[np.asarray(v)] for c, v in celdas.items()}

    def _celda(self, p) -> tuple[int, int]:
        return (int((p[0] - self._origen[0]) // CELDA_M), int((p[1] - self._origen[1]) // CELDA_M))

    def segmentos_cerca(self, p) -> np.ndarray:
        """Segmentos a menos de `margen` m de la celda de p (para rayos)."""
        return self._rejilla.get(self._celda(p), self._vacio)

    # ------------------------------------------------------------ consultas
    def progreso_local(self, p, i_prev: int, ventana: int = 12) -> tuple[float, int]:
        """Abscisa s de p y el índice del tramo, buscando en i_prev ± ventana."""
        n = len(self.centro)
        idx = (np.arange(i_prev - ventana, i_prev + ventana + 1)) % n
        a = self.centro[idx]
        d = self._tramo_dir[idx]
        t = np.clip(np.einsum("ij,ij->i", np.asarray(p) - a, d), 0.0, self._tramo_len[idx])
        q = a + d * t[:, None]
        k = int(np.argmin(np.einsum("ij,ij->i", q - p, q - p)))
        i = int(idx[k])
        return float(self.s[i] + t[k]), i

    def indice_inicial(self, p) -> int:
        return int(np.argmin(np.linalg.norm(self.centro - np.asarray(p), axis=1)))

    def rayos(self, p, rumbo: float, angulos: np.ndarray, alcances: np.ndarray) -> np.ndarray:
        """Distancia (m) al primer borde en cada rayo; si no hay borde, el alcance."""
        segs = self.segmentos_cerca(p)
        dist = np.array(alcances, dtype=float)
        if len(segs) == 0:
            return dist
        th = rumbo + np.asarray(angulos)
        r = np.stack([np.cos(th), np.sin(th)], axis=1)             # (R, 2)
        a = segs[:, :2]; e = segs[:, 2:] - a                         # (M, 2)
        w = a - np.asarray(p)                                        # (M, 2)
        den = r[:, None, 0] * e[None, :, 1] - r[:, None, 1] * e[None, :, 0]   # (R, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / den
            u = (w[None, :, 0] * r[:, None, 1] - w[None, :, 1] * r[:, None, 0]) / den
        ok = (np.abs(den) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(ok, t, np.inf).min(axis=1)
        return np.minimum(dist, t)

    def toca_borde(self, esquinas: np.ndarray) -> bool:
        """True si algún lado del polígono del carro corta un segmento de borde."""
        c = esquinas.mean(axis=0)
        segs = self._rejilla_choque.get(self._celda(c))
        if segs is None:
            # sin bordes a menos de 5 m de la celda: o va por el centro de una recta ancha
            # (no toca nada) o está fuera del área indexada (fuera de pista)
            return self._celda(c) not in self._rejilla
        p1 = esquinas; p2 = np.roll(esquinas, -1, axis=0)            # (4, 2)
        q1 = segs[:, :2]; q2 = segs[:, 2:]                           # (M, 2)

        def orient(a, b, c):
            return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

        P1, P2 = p1[:, None, :], p2[:, None, :]
        Q1, Q2 = q1[None, :, :], q2[None, :, :]
        d1, d2 = orient(P1, P2, Q1), orient(P1, P2, Q2)
        d3, d4 = orient(Q1, Q2, P1), orient(Q1, Q2, P2)
        return bool(np.any((d1 * d2 <= 0) & (d3 * d4 <= 0)))

    def ancho_medio(self) -> float:
        return float(np.mean(np.linalg.norm(self.borde_izq - self.borde_der, axis=1)))
