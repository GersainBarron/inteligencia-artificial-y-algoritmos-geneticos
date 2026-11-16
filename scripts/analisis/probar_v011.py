"""Comprobaciones del envoltorio v0.11 (control R-GA). Uso: python scripts/analisis/probar_v011.py"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "scripts"))
import entrenar_v011 as V  # noqa: E402
from evocarros.neuro import entrenar as E  # noqa: E402
from evocarros.sim.genoma import N_PESOS, N_RAYOS, decodificar_cuerpo, genoma_inicial  # noqa: E402

for s in (1, 7, 20):
    rng0 = np.random.default_rng(s)
    ref = [genoma_inicial(rng0, "A-GA") for _ in range(50)]     # original, como Corrida.nuevo
    V.instalar(s)
    rng = np.random.default_rng(s)
    pob = [E.genoma_inicial(rng, "R-GA") for _ in range(50)]
    assert all(np.array_equal(a[:N_PESOS], b[:N_PESOS]) for a, b in zip(ref, pob)), "pesos iniciales distintos"
    ang, alc = V.rayos_control(s)
    assert all(np.array_equal(g[N_PESOS:N_PESOS + N_RAYOS], ang) for g in pob)
    c = decodificar_cuerpo(pob[0], "R-GA")
    assert np.allclose(c.angulos, ang) and c.v_max == 35.0 and np.allclose(c.alcances_m / 12.249, alc)
    h = E.mutar(pob[0], "R-GA", np.random.default_rng(0))
    assert np.array_equal(h[N_PESOS:], pob[0][N_PESOS:]) and not np.array_equal(h[:N_PESOS], pob[0][:N_PESOS])
    rngb = np.random.default_rng(s)
    b = [E.genoma_inicial(rngb, "B-GA") for _ in range(50)]
    assert all(np.array_equal(x, y) for x, y in zip(ref, b)), "B cambió"
    print(f"semilla {s}: ok  ángulos {np.degrees(ang).round(1)}  alcances W {alc.round(2)}")
print("todo ok; protocolo", E.PROTOCOLO)
