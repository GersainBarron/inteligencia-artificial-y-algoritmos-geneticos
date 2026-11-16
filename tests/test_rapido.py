"""El episodio compilado (rapido.py) reproduce al episodio de referencia (episodio.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evocarros.sim import episodio as ep  # noqa: E402
from evocarros.sim import rapido  # noqa: E402
from evocarros.sim.genoma import N_PESOS, Red, decodificar_cuerpo, genoma_inicial  # noqa: E402

from test_simulador import mexico  # noqa: E402,F401  (fixture)

pytestmark = pytest.mark.skipif(not rapido.disponible, reason="Numba no está instalado")


@pytest.mark.parametrize("semilla,condicion", [(k, c) for k in range(12) for c in ("A", "B")])
def test_mismo_resultado_que_referencia(mexico, semilla, condicion):
    g = genoma_inicial(np.random.default_rng(semilla), condicion)
    if condicion == "B":
        rng = np.random.default_rng(100 + semilla)
        g[N_PESOS:N_PESOS + 5] = rng.uniform(-1.5, 1.5, 5)
        g[N_PESOS + 5:N_PESOS + 10] = rng.uniform(0.5, 8.0, 5)
    cuerpo = decodificar_cuerpo(g, condicion)
    r1 = ep.correr(mexico, Red(g[:N_PESOS]), cuerpo, t_limite=45.0, t_max=250.0, registrar=True)
    r2 = rapido.correr_red(mexico, g[:N_PESOS], cuerpo, t_limite=45.0, t_max=250.0, registrar=True)
    assert (r1.motivo, r1.pasos, r1.vuelta_completa) == (r2.motivo, r2.pasos, r2.vuelta_completa)
    assert r1.aptitud == pytest.approx(r2.aptitud, abs=1e-6)
    n = min(r1.pasos, 300)
    assert np.allclose(r1.registro["x"][:n], r2.registro["x"][:n], atol=1e-6)
    assert np.allclose(np.array(r1.registro["entradas"][:n]), np.array(r2.registro["entradas"][:n]), atol=1e-6)


def test_salida_perturbada(mexico):
    g = genoma_inicial(np.random.default_rng(5), "A")
    cuerpo = decodificar_cuerpo(g, "A")
    s = ep.posicion_salida(mexico, lateral=1.2, d_rumbo=0.04)
    r1 = ep.correr(mexico, Red(g[:N_PESOS]), cuerpo, t_limite=30.0, t_max=250.0, salida=s)
    r2 = rapido.correr_red(mexico, g[:N_PESOS], cuerpo, t_limite=30.0, t_max=250.0, salida=s)
    assert (r1.motivo, r1.pasos) == (r2.motivo, r2.pasos)
    assert r1.aptitud == pytest.approx(r2.aptitud, abs=1e-6)
