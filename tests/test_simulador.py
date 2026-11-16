"""Pruebas del simulador (fase 2): pista, física, rayos, choque, episodio y aptitud.

Usan `pistas/mexico/pista.json` si existe; si no, construyen una pista equivalente
directamente del CSV de TUM (línea central ± anchos medidos), con 50 checkpoints
equiespaciados, para poder correr en cualquier máquina.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evocarros.sim import episodio as ep
from evocarros.sim.fisica import DT, FISICA, EstadoCarro, esquinas, paso
from evocarros.sim.genoma import N_GENOMA, Cuerpo, Red, decodificar_cuerpo, genoma_inicial
from evocarros.sim.pista import Pista
from evocarros.sim.referencia import PilotoCentro

RAIZ = Path(__file__).resolve().parents[1]


def pista_desde_csv(csv: Path, n_cp: int = 50) -> dict:
    d = np.loadtxt(csv, delimiter=",", comments="#")
    c, w_der, w_izq = d[:, :2], d[:, 2], d[:, 3]
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    izq, der = c + n * w_izq[:, None], c - n * w_der[:, None]
    seg = np.linalg.norm(np.roll(c, -1, 0) - c, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)[:-1]])
    L = float(seg.sum())
    cps = []
    for k in range(n_cp):
        sk = k * L / n_cp
        i = int(np.searchsorted(s, sk, side="right") - 1)
        f = (sk - s[i]) / seg[i]
        j = (i + 1) % len(c)
        cps.append({"indice": k, "s_m": sk,
                    "izq": list(izq[i] + (izq[j] - izq[i]) * f), "der": list(der[i] + (der[j] - der[i]) * f)})
    return {"nombre": csv.stem, "linea_central": c.tolist(), "s_m": s.tolist(), "longitud_m": L,
            "borde_izquierdo": izq.tolist(), "borde_derecho": der.tolist(), "checkpoints": cps}


@pytest.fixture(scope="module")
def mexico() -> Pista:
    js = RAIZ / "pistas" / "mexico" / "pista.json"
    if js.exists():
        return Pista.desde_json(js)
    return Pista.desde_dict(pista_desde_csv(RAIZ / "pistas" / "fuente_tum" / "MexicoCity.csv"))


def correr_con_estado(pista, piloto, cuerpo, t_limite=250.0):
    return ep.correr(pista, piloto, cuerpo, t_limite=t_limite, t_max=250.0)


CUERPO_A = decodificar_cuerpo(np.zeros(N_GENOMA), "A")


# ------------------------------------------------------------------ pista
def test_pista_basica(mexico):
    assert 4200 < mexico.longitud < 4400
    assert len(mexico.cp_s) == 50
    assert 11.5 < mexico.ancho_medio() < 13.0


def test_progreso_avanza_por_la_linea_central(mexico):
    i, s_ant = 0, -1.0
    for k in range(0, len(mexico.centro) - 1, 7):
        s, i = mexico.progreso_local(mexico.centro[k], i, ventana=12)
        assert s >= s_ant - 1e-6
        s_ant = s


def test_rayos_ven_los_bordes(mexico):
    e = ep.posicion_salida(mexico)
    d = mexico.rayos((e.x, e.y), e.rumbo, np.radians([90.0, -90.0]), np.array([100.0, 100.0]))
    ancho = d.sum()
    assert 8.0 < ancho < 20.0, ancho                     # izquierda + derecha = ancho de pista
    d0 = mexico.rayos((e.x, e.y), e.rumbo, np.array([0.0]), np.array([30.0]))
    assert d0[0] == pytest.approx(30.0)                  # al frente hay recta: nada a 30 m


def test_choque(mexico):
    e = ep.posicion_salida(mexico)
    assert not mexico.toca_borde(esquinas(e))
    lateral = np.array([-math.sin(e.rumbo), math.cos(e.rumbo)])
    # montado sobre cada borde (anchos en la salida: ~6.6 m a la izquierda, ~7.1 m a la derecha).
    # Un carro desplazado por completo fuera de la pista no toca ninguna línea; en la simulación
    # eso no ocurre porque avanza a lo más 1.7 m por paso y cruza el borde antes.
    for signo in (1.0, -1.0):
        d = signo * 6.8
        sobre = EstadoCarro(e.x + lateral[0] * d, e.y + lateral[1] * d, e.rumbo)
        assert mexico.toca_borde(esquinas(sobre))


# ------------------------------------------------------------------ física
def test_fisica_aceleracion_y_tope():
    e = EstadoCarro(0, 0, 0)
    for _ in range(30):
        paso(e, 0.0, 1.0, 50.0)
    assert e.v == pytest.approx(6.0, abs=0.01)           # 6 m/s² durante 1 s
    for _ in range(30 * 20):
        paso(e, 0.0, 1.0, 35.0)
    assert e.v == pytest.approx(35.0)                    # recorta a v_max del cuerpo
    assert e.y == pytest.approx(0.0, abs=1e-9)


def test_fisica_limite_lateral():
    e = EstadoCarro(0, 0, 0, v=30.0)
    r0 = e.rumbo
    paso(e, 1.0, 0.0, 50.0)
    giro = (e.rumbo - r0) / DT                           # rad/s
    assert e.v * giro <= FISICA.a_lat * 1.001            # a_lat = v * omega
    e2 = EstadoCarro(0, 0, 0, v=0.0)
    for _ in range(10):
        paso(e2, 0.0, -1.0, 50.0)
    assert e2.v == 0.0                                   # sin reversa


# ------------------------------------------------------------------ episodio
def test_carro_quieto_muere_por_estancamiento(mexico):
    r = ep.correr(mexico, lambda x: np.array([0.0, 0.0]), CUERPO_A, t_limite=30.0, t_max=250.0)
    assert r.motivo == "estancamiento"
    assert r.t_final == pytest.approx(3.0 + DT, abs=0.05)
    assert r.aptitud >= 0.0


def test_carro_girando_choca_y_conserva_avance(mexico):
    r = ep.correr(mexico, lambda x: np.array([1.0, 1.0]), CUERPO_A, t_limite=30.0, t_max=250.0)
    assert r.motivo == "choque"
    assert r.aptitud == pytest.approx(r.avance_m / (mexico.longitud / 50), rel=1e-6)


def test_red_aleatoria_determinista(mexico):
    g = genoma_inicial(np.random.default_rng(7), "A")
    r1 = ep.correr(mexico, Red(g[:82]), CUERPO_A, t_limite=30.0, t_max=250.0)
    r2 = ep.correr(mexico, Red(g[:82]), CUERPO_A, t_limite=30.0, t_max=250.0)
    assert (r1.aptitud, r1.pasos, r1.motivo) == (r2.aptitud, r2.pasos, r2.motivo)


def test_piloto_de_referencia_completa_la_vuelta(mexico):
    r = correr_con_estado(mexico, PilotoCentro(mexico), Cuerpo(CUERPO_A.angulos, CUERPO_A.alcances_m, 50.0))
    assert r.motivo == "vuelta", r
    assert 125.0 <= r.t_vuelta <= 250.0
    assert r.aptitud > 50.0                              # 50 checkpoints + bono


# ------------------------------------------------------------------ repeticiones
def test_repeticion_ida_y_vuelta(mexico, tmp_path):
    from evocarros.sim import repeticion as rp
    g = genoma_inicial(np.random.default_rng(3), "B")
    cuerpo = decodificar_cuerpo(g, "B")
    red = Red(g[:82])
    res = ep.correr(mexico, red, cuerpo, t_limite=30.0, t_max=250.0, registrar=True)
    c = rp.carro("g0000_i000", cuerpo, res, pesos=g[:82], rango=1)
    ruta = rp.guardar(tmp_path / "gen_0000.json.gz", [c], pista=mexico.nombre, pista_archivo="pistas/mexico/pista.json",
                      corrida="prueba", condicion="B-GA", semilla=3, generacion=0)
    doc = rp.cargar(ruta)
    pc = doc["carros"][0]
    assert len(pc["pasos"]["x"]) == res.pasos
    # el visor recalcula la capa oculta con pesos + entradas: debe coincidir con la salida guardada
    red2 = Red(np.array(pc["controlador"]["pesos"]))
    for k in range(0, res.pasos, 17):
        sal = red2(np.array(pc["pasos"]["entradas"][k]))
        assert np.allclose(sal, pc["pasos"]["salidas"][k], atol=2e-3)
