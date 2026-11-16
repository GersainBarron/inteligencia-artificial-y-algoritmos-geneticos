"""Entrenador de neuroevolución: algoritmo genético (mu+lambda) al estilo Togelius y
estrategias evolutivas de OpenAI (Salimans et al., 2017).

Uso:
    python -m evocarros.neuro.entrenar --condicion A-GA --semilla 1
    python -m evocarros.neuro.entrenar --condicion B-ES --semilla 1
    python -m evocarros.neuro.entrenar --condicion A-GA --semilla 1 --episodios 2000   # prueba corta

Reanuda solo desde <runs>/<corrida>/checkpoint.pkl si existe.

Protocolo v0.5:
- 50 padres + 50 hijos; cada hijo es copia mutada de un padre elegido al azar; sobreviven
  los 50 mejores de los 100. Mutación gaussiana: sigma 0.1 en pesos; 10° en ángulos y
  0.2 W en alcances (B y C); 1 m/s en velocidad máxima (C). Sin cruza.
- El simulador es determinista: los padres conservan su aptitud y solo se evalúan los
  hijos. Cuando el tiempo límite sube, se reevalúa toda la población. El presupuesto se
  mide en episodios (100,000 por corrida), igual para GA y ES.
- ES de OpenAI (v0.10): theta inicial = el mejor de los mismos 50 genomas iniciales que usa el
  genético con esa semilla (50 episodios, contados en el presupuesto); así ambos algoritmos
  parten de la misma información y ES no queda atrapado si un theta al azar no se mueve
  (aptitud 0 en todas las perturbaciones = sin señal). Después, 100 perturbaciones por generación (50 pares +-eps),
  sigma 0.05, aptitudes a rangos centrados, paso con Adam (alfa 0.01) y decaimiento L2
  0.005. Los genes que la condición no evoluciona no reciben ruido. La velocidad
  máxima (C) se representa en decenas de m/s para que sigma tenga una escala parecida.
  Cada generación evalúa también theta (1 episodio más, contado en el presupuesto).
- Tiempo límite adaptativo (v0.6): 30 s; ×1.5 cuando >= 50% de la población sobrevive
  hasta el límite o completa la vuelta; hasta T_max = 2 × vuelta ideal (250 s en México).
- Convergencia (v0.6): >= 95% de la población completa la vuelta y el mejor tiempo de
  vuelta no mejora más de 0.5% en los últimos 10,000 episodios (equivale a 100
  generaciones de ES y 200 del genético, para que la regla sea igual entre algoritmos).
- Control: lee <corrida>/control.json al terminar cada generación (correr|pausar|detener).
- Paros de seguridad: GPU > 85 °C, < 20 GB libres, aptitud NaN o error del simulador.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from evocarros.util import reemplazar  # noqa: E402
from evocarros.sim import episodio as ep  # noqa: E402
from evocarros.sim import rapido  # noqa: E402
from evocarros.sim import repeticion as rp  # noqa: E402
from evocarros.sim.genoma import N_GENOMA, N_PESOS, N_RAYOS, Red, decodificar_cuerpo, genoma_inicial  # noqa: E402
from evocarros.sim.pista import Pista  # noqa: E402

MU, LAMBDA = 50, 50
SIGMA_PESOS, SIGMA_ANG, SIGMA_ALC, SIGMA_V = 0.1, math.radians(10.0), 0.2, 1.0
T_INICIAL, FACTOR_T = 30.0, 1.5
VENTANA_CONV_EP, MEJORA_MIN = 10_000, 0.005
ES_N, ES_SIGMA, ES_ALFA, ES_L2 = 100, 0.05, 0.01, 0.005
ES_ESCALA_V = 10.0      # theta guarda v_max / 10
GPU_MAX_C, DISCO_MIN_GB = 85.0, 20.0
PROTOCOLO = "v0.10"
COLUMNAS = ["generacion", "episodios", "aptitud_max", "aptitud_media", "aptitud_mediana", "pct_vuelta",
            "mejor_t_vuelta", "diversidad", "t_limite", "segundos"]

# ------------------------------------------------------------------ trabajadores
_P: Pista | None = None


def _init(ruta_pista: str):
    global _P
    _P = Pista.desde_json(ruta_pista)


def _evaluar(args):
    g, condicion, t_limite, t_max, registrar = args
    cuerpo = decodificar_cuerpo(g, condicion)
    if rapido.disponible:   # mismo modelo, compilado (tests/test_rapido.py)
        r = rapido.correr_red(_P, g[:N_PESOS], cuerpo, t_limite=t_limite, t_max=t_max, registrar=registrar)
    else:
        r = ep.correr(_P, Red(g[:N_PESOS]), cuerpo, t_limite=t_limite, t_max=t_max, registrar=registrar)
    if registrar:
        return r, cuerpo
    r.registro = None
    return r


# ------------------------------------------------------------------ utilidades
def runs_por_defecto() -> Path:
    if os.environ.get("EVOCARROS_RUNS"):
        return Path(os.environ["EVOCARROS_RUNS"])
    d = Path(r"D:\redes neuronales\runs")
    return d if d.exists() else RAIZ / "runs"


def temp_gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def mutar(g: np.ndarray, condicion: str, rng: np.random.Generator) -> np.ndarray:
    h = g.copy()
    h[:N_PESOS] += rng.normal(0.0, SIGMA_PESOS, N_PESOS)
    if condicion[0] in "BC":
        h[N_PESOS:N_PESOS + N_RAYOS] += rng.normal(0.0, SIGMA_ANG, N_RAYOS)
        h[N_PESOS + N_RAYOS:N_PESOS + 2 * N_RAYOS] += rng.normal(0.0, SIGMA_ALC, N_RAYOS)
        h[N_PESOS + N_RAYOS:N_PESOS + 2 * N_RAYOS] = np.clip(h[N_PESOS + N_RAYOS:N_PESOS + 2 * N_RAYOS], 0.5, 8.0)
    if condicion[0] == "C":
        h[N_GENOMA - 1] = np.clip(h[N_GENOMA - 1] + rng.normal(0.0, SIGMA_V), 15.0, 50.0)
    return h


class Corrida:
    def __init__(self, a):
        self.a = a
        self.id = a.corrida or f"{a.condicion}_s{a.semilla:02d}"
        self.dir = a.runs / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pista_rel = a.pista
        self.pista = Pista.desde_json(RAIZ / a.pista)
        self.t_max = a.t_max
        self.ckpt = self.dir / "checkpoint.pkl"

    # ---------------------------------------------------------- estado
    def nuevo(self):
        self.rng = np.random.default_rng(self.a.semilla)
        self.pob = [genoma_inicial(self.rng, self.a.condicion) for _ in range(MU)]
        self.ids = [f"g0000_i{k:03d}" for k in range(MU)]
        self.padres = [None] * MU
        self.res = None
        self.gen = 0
        self.episodios = 0
        self.t_limite = T_INICIAL
        self.historial_mejor = []      # mejor tiempo de vuelta por generación
        self.segundos = 0.0
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.a).items()}
        cfg.update(mu=MU, lambda_=LAMBDA, sigma_pesos=SIGMA_PESOS, sigma_ang_grados=10.0, sigma_alc_W=SIGMA_ALC,
                   sigma_v=SIGMA_V, t_inicial=T_INICIAL, factor_t=FACTOR_T, protocolo=PROTOCOLO,
                   motor="numba" if rapido.disponible else "python",
                   creada=datetime.now().isoformat(timespec="seconds"))
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(self.dir / "fitness.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNAS)

    def guardar(self):
        tmp = self.ckpt.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({k: getattr(self, k) for k in ("rng", "pob", "ids", "padres", "res", "gen", "episodios",
                                                       "t_limite", "historial_mejor", "segundos")}, f)
        reemplazar(tmp, self.ckpt)

    def cargar(self) -> bool:
        if not self.ckpt.exists():
            return False
        with open(self.ckpt, "rb") as f:
            for k, v in pickle.load(f).items():
                setattr(self, k, v)
        return True

    def estado(self, estado: str, extra: dict | None = None):
        mejor_t = min((r.t_vuelta for r in self.res if r.t_vuelta), default=None) if self.res else None
        ritmo = self.segundos / max(self.episodios, 1)
        d = {"corrida": self.id, "condicion": self.a.condicion, "semilla": self.a.semilla, "estado": estado,
             "generacion": self.gen, "episodios": self.episodios, "episodios_max": self.a.episodios,
             "generaciones_max": None,
             "mejor_aptitud": max(r.aptitud for r in self.res) if self.res else None,
             "pct_vuelta": 100.0 * np.mean([r.vuelta_completa for r in self.res]) if self.res else None,
             "mejor_t_vuelta": mejor_t, "t_limite": self.t_limite,
             "eta_s": ritmo * max(self.a.episodios - self.episodios, 0) if estado == "corriendo" else None,
             "pista": self.pista_rel, "actualizado": datetime.now().isoformat(timespec="seconds")}
        d.update(extra or {})
        tmp = self.dir / "estado.json.tmp"
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        reemplazar(tmp, self.dir / "estado.json")

    def orden(self) -> str:
        try:
            return json.loads((self.dir / "control.json").read_text(encoding="utf-8")).get("orden", "correr")
        except (OSError, json.JSONDecodeError):
            return "correr"

    # ---------------------------------------------------------- evolución
    def evaluar(self, pool, genomas):
        trabajos = [(g, self.a.condicion, self.t_limite, self.t_max, False) for g in genomas]
        res = pool.map(_evaluar, trabajos, chunksize=max(1, len(trabajos) // (4 * self.a.procesos)))
        self.episodios += len(res)
        for r in res:
            if not math.isfinite(r.aptitud):
                raise FloatingPointError("aptitud inválida (NaN o infinito)")
        return res

    def generacion(self, pool):
        t0 = time.time()
        if self.res is None:                       # primera evaluación de la población inicial
            self.res = self.evaluar(pool, self.pob)
        hijos, ids_h, padres_h = [], [], []
        for k in range(LAMBDA):
            i = int(self.rng.integers(MU))
            hijos.append(mutar(self.pob[i], self.a.condicion, self.rng))
            ids_h.append(f"g{self.gen + 1:04d}_i{k:03d}")
            padres_h.append(self.ids[i])
        res_h = self.evaluar(pool, hijos)
        todos = list(zip(self.pob + hijos, self.ids + ids_h, self.padres + padres_h, self.res + res_h))
        todos.sort(key=lambda x: -x[3].aptitud)
        todos = todos[:MU]
        self.pob = [x[0] for x in todos]; self.ids = [x[1] for x in todos]
        self.padres = [x[2] for x in todos]; self.res = [x[3] for x in todos]
        self.gen += 1
        # tiempo límite adaptativo: reevalúa la población con el límite nuevo
        if self.subir_limite():
            self.res = self.evaluar(pool, self.pob)
            orden = np.argsort([-r.aptitud for r in self.res])
            self.pob = [self.pob[i] for i in orden]; self.ids = [self.ids[i] for i in orden]
            self.padres = [self.padres[i] for i in orden]; self.res = [self.res[i] for i in orden]
        self.segundos += time.time() - t0
        self.registrar_generacion(time.time() - t0)

    def subir_limite(self) -> bool:
        """v0.6: sube si >= 50% sobrevive hasta el límite o completa la vuelta."""
        vivos = np.mean([r.motivo in ("tiempo", "vuelta") for r in self.res])
        if vivos >= 0.5 and self.t_limite < self.t_max:
            self.t_limite = min(self.t_max, self.t_limite * FACTOR_T)
            return True
        return False

    def registrar_generacion(self, seg):
        apt = np.array([r.aptitud for r in self.res])
        tv = [r.t_vuelta for r in self.res if r.t_vuelta]
        mejor_t = min(tv) if tv else None
        self.historial_mejor.append((self.episodios, mejor_t))
        P = np.array(self.pob)
        div = float(np.mean(np.linalg.norm(P - P.mean(axis=0), axis=1)))
        fila = [self.gen, self.episodios, round(apt.max(), 4), round(apt.mean(), 4), round(float(np.median(apt)), 4),
                round(100 * len(tv) / len(self.res), 1), "" if mejor_t is None else round(mejor_t, 3),
                round(div, 4), round(self.t_limite, 2), round(seg, 2)]
        with open(self.dir / "fitness.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fila)

    def campeon(self):
        d = self.dir / "champions"
        d.mkdir(exist_ok=True)
        r = self.res[0]
        cuerpo = decodificar_cuerpo(self.pob[0], self.a.condicion)
        (d / f"gen_{self.gen:04d}.json").write_text(json.dumps({
            "id": self.ids[0], "padre": self.padres[0], "generacion": self.gen, "aptitud": r.aptitud,
            "motivo": r.motivo, "t_vuelta": r.t_vuelta, "genoma": self.pob[0].tolist(),   # precisión completa: redondear cambia a carros frágiles
            "angulos_grados": np.degrees(cuerpo.angulos).round(3).tolist(),
            "alcances_m": cuerpo.alcances_m.round(3).tolist(), "v_max": cuerpo.v_max}, ensure_ascii=False), encoding="utf-8")

    def repeticion(self, pool):
        n = min(rp.N_GUARDADOS, len(self.pob))
        trabajos = [(self.pob[k], self.a.condicion, self.t_limite, self.t_max, True) for k in range(n)]
        salidas = pool.map(_evaluar, trabajos)
        carros = [rp.carro(self.ids[k], c, r, pesos=self.pob[k][:N_PESOS], padre=self.padres[k], rango=k + 1)
                  for k, (r, c) in enumerate(salidas)]
        rp.guardar(self.dir / "replays" / f"gen_{self.gen:04d}.json.gz", carros, pista=self.pista.nombre,
                   pista_archivo=self.pista_rel, corrida=self.id, condicion=self.a.condicion,
                   semilla=self.a.semilla, generacion=self.gen)

    def convergio(self) -> bool:
        if np.mean([r.vuelta_completa for r in self.res]) < 0.95 or not self.historial_mejor:
            return False
        ep_ahora, ahora = self.historial_mejor[-1]
        antes = [t for e, t in self.historial_mejor if e <= ep_ahora - VENTANA_CONV_EP]
        if not antes or antes[-1] is None or ahora is None:
            return False
        return (antes[-1] - ahora) / antes[-1] < MEJORA_MIN

    def seguridad(self) -> str | None:
        g = temp_gpu()
        if g is not None and g > GPU_MAX_C:
            return f"GPU a {g:.0f} °C"
        libre = shutil.disk_usage(self.dir).free / 1e9
        if libre < DISCO_MIN_GB:
            return f"solo {libre:.1f} GB libres"
        return None


class CorridaES(Corrida):
    """Estrategias evolutivas de OpenAI con muestreo antitético, rangos centrados, Adam y L2."""

    CLAVES = ("rng", "theta", "m", "v", "paso_adam", "pob", "ids", "padres", "res", "gen", "episodios",
              "t_limite", "historial_mejor", "segundos")

    def mascara(self) -> np.ndarray:
        m = np.zeros(N_GENOMA)
        m[:N_PESOS] = 1.0
        if self.a.condicion[0] in "BC":
            m[N_PESOS:N_PESOS + 2 * N_RAYOS] = 1.0
        if self.a.condicion[0] == "C":
            m[N_GENOMA - 1] = 1.0
        return m

    @staticmethod
    def a_genoma(theta: np.ndarray) -> np.ndarray:
        g = theta.copy()
        g[N_GENOMA - 1] *= ES_ESCALA_V
        return g

    def nuevo(self):
        super().nuevo()                      # misma población inicial que el genético (misma semilla)
        self.theta = None
        self.m = np.zeros(N_GENOMA); self.v = np.zeros(N_GENOMA); self.paso_adam = 0

    def inicial(self, pool):
        """theta = el mejor de los 50 genomas iniciales (v0.10)."""
        res = self.evaluar(pool, self.pob)
        k = int(np.argmax([r.aptitud for r in res]))
        g0 = self.pob[k]
        self.theta = g0.copy()
        self.theta[N_GENOMA - 1] /= ES_ESCALA_V
        self.pob = [g0]; self.ids = [f"g0000_i{k:03d}"]; self.padres = [None]; self.res = [res[k]]

    def guardar(self):
        tmp = self.ckpt.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({k: getattr(self, k) for k in self.CLAVES}, f)
        reemplazar(tmp, self.ckpt)

    def generacion(self, pool):
        t0 = time.time()
        mask = self.mascara()
        eps = self.rng.normal(0.0, 1.0, (ES_N // 2, N_GENOMA)) * mask
        eps = np.vstack([eps, -eps])
        thetas = self.theta + ES_SIGMA * eps
        genomas = [self.a_genoma(t) for t in thetas] + [self.a_genoma(self.theta)]
        res = self.evaluar(pool, genomas)
        f = np.array([r.aptitud for r in res[:-1]])
        rangos = np.empty(len(f)); rangos[np.argsort(f)] = np.arange(len(f))
        u = rangos / (len(f) - 1) - 0.5                       # rangos centrados en [-0.5, 0.5]
        grad = (u @ eps) / (len(f) * ES_SIGMA) - ES_L2 * self.theta * mask
        # Adam (ascenso)
        self.paso_adam += 1
        b1, b2 = 0.9, 0.999
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * grad * grad
        mh = self.m / (1 - b1 ** self.paso_adam); vh = self.v / (1 - b2 ** self.paso_adam)
        self.theta = self.theta + ES_ALFA * mh / (np.sqrt(vh) + 1e-8) * mask
        if self.a.condicion[0] in "BC":
            k0, k1 = N_PESOS + N_RAYOS, N_PESOS + 2 * N_RAYOS
            self.theta[k0:k1] = np.clip(self.theta[k0:k1], 0.5, 8.0)
        if self.a.condicion[0] == "C":
            self.theta[N_GENOMA - 1] = np.clip(self.theta[N_GENOMA - 1], 1.5, 5.0)
        self.gen += 1
        orden = np.argsort([-r.aptitud for r in res])
        etiquetas = [f"g{self.gen:04d}_p{k:03d}" for k in range(len(res) - 1)] + [f"g{self.gen:04d}_theta"]
        self.pob = [genomas[i] for i in orden]; self.res = [res[i] for i in orden]
        self.ids = [etiquetas[i] for i in orden]
        self.padres = [f"g{self.gen - 1:04d}_theta"] * len(orden)
        self.subir_limite()            # en ES no hace falta reevaluar: cada generación evalúa todo de nuevo
        self.segundos += time.time() - t0
        self.registrar_generacion(time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condicion", default="A-GA", choices=["A-GA", "B-GA", "C-GA", "A-ES", "B-ES", "C-ES"])
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--corrida", default=None)
    ap.add_argument("--pista", default="pistas/mexico/pista.json")
    ap.add_argument("--t-max", type=float, default=250.0)
    ap.add_argument("--episodios", type=int, default=int(os.environ.get("EVOCARROS_EPISODIOS", 100_000)))
    ap.add_argument("--procesos", type=int, default=12)
    ap.add_argument("--runs", type=Path, default=runs_por_defecto())
    a = ap.parse_args()
    c = CorridaES(a) if a.condicion.endswith("ES") else Corrida(a)
    if not c.cargar():
        c.nuevo()
    with Pool(a.procesos, initializer=_init, initargs=(str(RAIZ / a.pista),)) as pool:
        if c.res is None:
            if isinstance(c, CorridaES):
                c.inicial(pool)
            else:
                c.res = c.evaluar(pool, c.pob)
            c.registrar_generacion(0.0)
            c.repeticion(pool)
            c.campeon()
        try:
            while True:
                o = c.orden()
                if o == "detener":
                    c.guardar(); c.estado("detenida"); print("detenida por el panel"); return
                if o == "pausar":
                    c.guardar(); c.estado("pausada")
                    while c.orden() == "pausar":
                        time.sleep(5)
                    continue
                motivo = c.seguridad()
                if motivo:
                    c.guardar(); c.estado("pausada", {"motivo_pausa": motivo}); print("pausa:", motivo)
                    time.sleep(60)
                    continue
                if c.episodios >= a.episodios:
                    c.guardar(); c.repeticion(pool); c.campeon(); c.estado("terminada", {"fin": "presupuesto"}); return
                if c.convergio():
                    c.guardar(); c.repeticion(pool); c.campeon(); c.estado("terminada", {"fin": "convergencia"}); return
                c.generacion(pool)
                if c.gen % 10 == 0:
                    c.campeon()
                if (c.gen <= 100 and c.gen % 10 == 0) or c.gen % 50 == 0:
                    c.repeticion(pool)
                if c.gen % 25 == 0:
                    c.guardar()
                c.estado("corriendo")
        except Exception as e:  # paro de seguridad por error: queda registrado
            c.guardar()
            c.estado("pausada", {"motivo_pausa": f"error: {e}", "traza": traceback.format_exc()[-2000:]})
            raise


if __name__ == "__main__":
    main()
