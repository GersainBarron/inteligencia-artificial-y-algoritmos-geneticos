"""Episodio compilado con Numba: el mismo modelo que `episodio.py`, para la red 7:8:2.

`episodio.correr` sigue siendo la referencia legible (y la que usan los pilotos
escritos a mano y la condición D). Este módulo repite exactamente las mismas reglas
(física v0.4, rayos, choque, estancamiento, sentido contrario, vuelta, tiempo límite
y aptitud) en un solo bucle compilado, sin llamadas a NumPy por paso.

Equivalencia: `tests/test_rapido.py` compara ambas versiones paso a paso. Pueden diferir
en el último bit de algún cálculo (orden de las sumas en el producto de la red,
implementación de tan/cos), así que se exige igualdad de trayectorias hasta 1e-6 m en
los primeros cientos de pasos y la misma aptitud dentro de 1e-6 en episodios cortos,
no igualdad binaria en episodios largos (un sistema caótico amplifica el último bit).

Si Numba no está instalado, `disponible` es False y el entrenador usa `episodio.correr`.
"""
from __future__ import annotations

import math

import numpy as np

from .episodio import AVANCE_MIN_M, BONO_B, RETROCESO_MAX_M, VENTANA_ESTANCAMIENTO_S, Resultado
from .fisica import DT, FISICA
from .genoma import Cuerpo
from .pista import CELDA_M, MARGEN_CHOQUE_M, Pista

try:
    from numba import njit
    disponible = True
except ImportError:  # pragma: no cover
    disponible = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco

MOTIVOS = ("tiempo", "choque", "estancamiento", "sentido_contrario", "vuelta")


# ------------------------------------------------------------------ datos de la pista en arreglos
def _csr(pista: Pista, rejilla: dict, nx: int, ny: int):
    """Rejilla dict -> CSR: inicio[celda], fin[celda] sobre un arreglo de segmentos."""
    inicio = np.zeros(nx * ny, np.int64)
    fin = np.zeros(nx * ny, np.int64)
    bloques, pos = [], 0
    for cy in range(ny):
        for cx in range(nx):
            segs = rejilla.get((cx, cy))
            k = cy * nx + cx
            inicio[k] = pos
            if segs is not None:
                bloques.append(segs)
                pos += len(segs)
            fin[k] = pos
    datos = np.vstack(bloques) if bloques else np.zeros((0, 4))
    return np.ascontiguousarray(datos, dtype=np.float64), inicio, fin


def arreglos_pista(pista: Pista) -> tuple:
    if getattr(pista, "_arreglos", None) is not None:
        return pista._arreglos
    todos = np.vstack([pista.borde_izq, pista.borde_der])
    mx = todos.max(axis=0) + pista.margen + CELDA_M
    nx, ny = (np.ceil((mx - pista._origen) / CELDA_M).astype(int) + 1).tolist()
    seg_r, ini_r, fin_r = _csr(pista, pista._rejilla, nx, ny)
    seg_c, ini_c, fin_c = _csr(pista, pista._rejilla_choque, nx, ny)
    cp_mid = 0.5 * (pista.cp_izq + pista.cp_der)
    pista._arreglos = (np.ascontiguousarray(pista.centro), np.ascontiguousarray(pista.s),
                       np.ascontiguousarray(pista._tramo_dir), np.ascontiguousarray(pista._tramo_len),
                       float(pista.longitud), np.ascontiguousarray(pista.cp_s), np.ascontiguousarray(cp_mid),
                       float(pista._origen[0]), float(pista._origen[1]), int(nx), int(ny),
                       seg_r, ini_r, fin_r, seg_c, ini_c, fin_c)
    return pista._arreglos


# ------------------------------------------------------------------ núcleo compilado
@njit(cache=True)
def _celda(x, y, ox, oy, nx, ny):
    cx = int(math.floor((x - ox) / CELDA_M))
    cy = int(math.floor((y - oy) / CELDA_M))
    if cx < 0 or cy < 0 or cx >= nx or cy >= ny:
        return -1
    return cy * nx + cx


@njit(cache=True)
def _progreso(px, py, i_prev, centro, s, tdir, tlen):
    n = centro.shape[0]
    mejor_d = 1e300
    mejor_i = 0
    mejor_t = 0.0
    for k in range(-12, 13):
        i = (i_prev + k) % n
        ax = centro[i, 0]; ay = centro[i, 1]
        dx = tdir[i, 0]; dy = tdir[i, 1]
        t = (px - ax) * dx + (py - ay) * dy
        if t < 0.0:
            t = 0.0
        elif t > tlen[i]:
            t = tlen[i]
        qx = ax + dx * t - px; qy = ay + dy * t - py
        d = qx * qx + qy * qy
        if d < mejor_d:
            mejor_d = d; mejor_i = i; mejor_t = t
    return s[mejor_i] + mejor_t, mejor_i


@njit(cache=True)
def _rayos(px, py, rumbo, ang, alc, out, ox, oy, nx, ny, seg, ini, fin):
    for r in range(ang.shape[0]):
        out[r] = alc[r]
    c = _celda(px, py, ox, oy, nx, ny)
    if c < 0:
        return
    for r in range(ang.shape[0]):
        th = rumbo + ang[r]
        rx = math.cos(th); ry = math.sin(th)
        mejor = 1e300
        for k in range(ini[c], fin[c]):
            ax = seg[k, 0]; ay = seg[k, 1]
            ex = seg[k, 2] - ax; ey = seg[k, 3] - ay
            wx = ax - px; wy = ay - py
            den = rx * ey - ry * ex
            if abs(den) <= 1e-12:
                continue
            t = (wx * ey - wy * ex) / den
            u = (wx * ry - wy * rx) / den
            if t >= 0.0 and u >= 0.0 and u <= 1.0 and t < mejor:
                mejor = t
        if mejor < out[r]:
            out[r] = mejor


@njit(cache=True)
def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _toca(esq, ox, oy, nx, ny, seg_c, ini_c, fin_c, ini_r, fin_r):
    mx = 0.25 * (esq[0, 0] + esq[1, 0] + esq[2, 0] + esq[3, 0])
    my = 0.25 * (esq[0, 1] + esq[1, 1] + esq[2, 1] + esq[3, 1])
    c = _celda(mx, my, ox, oy, nx, ny)
    if c < 0:
        return True
    if fin_c[c] == ini_c[c]:
        return fin_r[c] == ini_r[c]
    for a in range(4):
        b = (a + 1) % 4
        p1x = esq[a, 0]; p1y = esq[a, 1]; p2x = esq[b, 0]; p2y = esq[b, 1]
        for k in range(ini_c[c], fin_c[c]):
            q1x = seg_c[k, 0]; q1y = seg_c[k, 1]; q2x = seg_c[k, 2]; q2y = seg_c[k, 3]
            d1 = _orient(p1x, p1y, p2x, p2y, q1x, q1y)
            d2 = _orient(p1x, p1y, p2x, p2y, q2x, q2y)
            d3 = _orient(q1x, q1y, q2x, q2y, p1x, p1y)
            d4 = _orient(q1x, q1y, q2x, q2y, p2x, p2y)
            if d1 * d2 <= 0.0 and d3 * d4 <= 0.0:
                return True
    return False


@njit(cache=True)
def _episodio(centro, s, tdir, tlen, L, cp_s, cp_mid, ox, oy, nx, ny,
              seg_r, ini_r, fin_r, seg_c, ini_c, fin_c,
              W1, b1, W2, b2, ang, alc, v_max, x0, y0, rumbo0, t_limite,
              v_tope, a_acel, a_freno, a_resist, a_lat, ejes, giro_max, largo, ancho,
              registrar, reg):
    n = centro.shape[0]
    # índice inicial: punto más cercano de toda la línea central
    mejor = 1e300; i_t = 0
    for i in range(n):
        d = (centro[i, 0] - x0) ** 2 + (centro[i, 1] - y0) ** 2
        if d < mejor:
            mejor = d; i_t = i
    x = x0; y = y0; rumbo = rumbo0; v = 0.0
    s_ant, i_t = _progreso(x, y, i_t, centro, s, tdir, tlen)
    vueltas = 0
    avance = s_ant; avance_max = s_ant; s_ini = s_ant
    nh = int(round(VENTANA_ESTANCAMIENTO_S / DT)) + 1
    hist = np.zeros(nh); hn = 0; hpos = 0
    ent = np.zeros(7); d_r = np.zeros(ang.shape[0]); oc = np.zeros(8); sal = np.zeros(2)
    esq = np.zeros((4, 2))
    n_pasos = int(math.ceil(t_limite / DT))
    motivo = 0; t_vuelta = -1.0; paso_i = 0
    K = cp_s.shape[0]
    vm = v_max if v_max < v_tope else v_tope
    for paso_i in range(1, n_pasos + 1):
        _rayos(x, y, rumbo, ang, alc, d_r, ox, oy, nx, ny, seg_r, ini_r, fin_r)
        for r in range(5):
            ent[r] = d_r[r] / alc[r]
        ent[5] = v / v_tope
        s_local = avance - vueltas * L
        sm = s_local % L
        j = 0                                     # searchsorted(cp_s, sm, side="right") % K
        while j < K and cp_s[j] <= sm:
            j += 1
        j = j % K
        a_cp = math.atan2(cp_mid[j, 1] - y, cp_mid[j, 0] - x) - rumbo
        ent[6] = ((a_cp + math.pi) % (2 * math.pi) - math.pi) / math.pi
        for h in range(8):
            acc = b1[h]
            for i in range(7):
                acc += W1[h, i] * ent[i]
            oc[h] = math.tanh(acc)
        for o in range(2):
            acc = b2[o]
            for h in range(8):
                acc += W2[o, h] * oc[h]
            sal[o] = math.tanh(acc)
        if registrar:
            k = paso_i - 1
            reg[k, 0] = x; reg[k, 1] = y; reg[k, 2] = rumbo; reg[k, 3] = v
            for i in range(7):
                reg[k, 4 + i] = ent[i]
            reg[k, 11] = sal[0]; reg[k, 12] = sal[1]
        # física (fisica.paso)
        vol = sal[0]; ped = sal[1]
        if vol > 1.0: vol = 1.0
        if vol < -1.0: vol = -1.0
        if ped > 1.0: ped = 1.0
        if ped < -1.0: ped = -1.0
        if ped > 0.0:
            a = ped * a_acel
        else:
            a = ped * a_freno - a_resist
        v1 = v + a * DT
        if v1 < 0.0: v1 = 0.0
        if v1 > vm: v1 = vm
        v_med = 0.5 * (v + v1)
        kappa = math.tan(vol * giro_max) / ejes
        if v_med > 0.1:
            k_lim = a_lat / (v_med * v_med)
            if kappa > k_lim: kappa = k_lim
            if kappa < -k_lim: kappa = -k_lim
        dth = v_med * kappa * DT
        th_med = rumbo + 0.5 * dth
        x += v_med * math.cos(th_med) * DT
        y += v_med * math.sin(th_med) * DT
        rumbo = (rumbo + dth + math.pi) % (2 * math.pi) - math.pi
        v = v1
        t = paso_i * DT
        s_n, i_t = _progreso(x, y, i_t, centro, s, tdir, tlen)
        if s_n - s_ant < -0.5 * L:
            vueltas += 1
        elif s_n - s_ant > 0.5 * L:
            vueltas -= 1
        s_ant = s_n
        avance = s_n + vueltas * L
        if avance > avance_max:
            avance_max = avance
        hist[hpos] = avance_max; hpos = (hpos + 1) % nh
        if hn < nh: hn += 1
        # esquinas (fisica.esquinas)
        cr = math.cos(rumbo); sr = math.sin(rumbo); hl = largo / 2; hw = ancho / 2
        esq[0, 0] = x + cr * hl - sr * hw;    esq[0, 1] = y + sr * hl + cr * hw
        esq[1, 0] = x + cr * hl - sr * -hw;   esq[1, 1] = y + sr * hl + cr * -hw
        esq[2, 0] = x + cr * -hl - sr * -hw;  esq[2, 1] = y + sr * -hl + cr * -hw
        esq[3, 0] = x + cr * -hl - sr * hw;   esq[3, 1] = y + sr * -hl + cr * hw
        if _toca(esq, ox, oy, nx, ny, seg_c, ini_c, fin_c, ini_r, fin_r):
            motivo = 1; break
        if avance_max - s_ini >= L:
            motivo = 4; t_vuelta = t; break
        if avance < avance_max - RETROCESO_MAX_M:
            motivo = 3; break
        if hn == nh and hist[(hpos - 1) % nh] - hist[hpos % nh] < AVANCE_MIN_M:
            motivo = 2; break
    return avance_max - s_ini, t_vuelta, motivo, paso_i


def correr_red(pista: Pista, pesos: np.ndarray, cuerpo: Cuerpo, t_limite: float, t_max: float,
               registrar: bool = False, salida=None) -> Resultado:
    """Equivale a `episodio.correr(pista, Red(pesos), cuerpo, ...)`, compilado."""
    from .episodio import aptitud, posicion_salida
    A = arreglos_pista(pista)
    p = np.asarray(pesos, float)
    W1 = np.ascontiguousarray(p[:56].reshape(8, 7)); b1 = p[56:64].copy()
    W2 = np.ascontiguousarray(p[64:80].reshape(2, 8)); b2 = p[80:82].copy()
    e = posicion_salida(pista) if salida is None else salida
    f = FISICA
    n_max = int(math.ceil(t_limite / DT))
    reg = np.zeros((n_max if registrar else 1, 13))
    av, tv, mot, pasos = _episodio(*A, W1, b1, W2, b2, np.ascontiguousarray(cuerpo.angulos, dtype=float),
                                   np.ascontiguousarray(cuerpo.alcances_m, dtype=float), float(cuerpo.v_max),
                                   float(e.x), float(e.y), float(e.rumbo), float(t_limite),
                                   f.v_tope, f.a_acel, f.a_freno, f.a_resist, f.a_lat, f.ejes, f.giro_max,
                                   f.largo, f.ancho, registrar, reg)
    t_vuelta = None if tv < 0 else float(tv)
    apt, n_cp = aptitud(pista, float(av), t_vuelta, t_max)
    registro = None
    if registrar:
        r = reg[:pasos]
        registro = {"x": r[:, 0].tolist(), "y": r[:, 1].tolist(), "rumbo": r[:, 2].tolist(), "v": r[:, 3].tolist(),
                    "entradas": [row for row in r[:, 4:11]], "salidas": [row for row in r[:, 11:13]],
                    "oculta": [None] * pasos}
    return Resultado(aptitud=apt, avance_m=float(av), fraccion_vuelta=float(av / pista.longitud), checkpoints=n_cp,
                     vuelta_completa=t_vuelta is not None, t_vuelta=t_vuelta, t_final=pasos * DT,
                     motivo=MOTIVOS[mot], pasos=int(pasos), registro=registro)
