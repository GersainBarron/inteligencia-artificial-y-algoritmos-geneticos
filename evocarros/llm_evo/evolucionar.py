"""Mini-AlphaEvolve propio (condición D, fase 6).

Uso:
    python -m evocarros.llm_evo.evolucionar --semilla 1
    python -m evocarros.llm_evo.evolucionar --semilla 1 --programas 40      # prueba corta

Ciclo (protocolo v0.9): elegir isla y padre (MAP-Elites), armar el prompt con métricas,
retroalimentación y 2 inspiraciones de otras celdas, pedir el cambio al modelo local (el chico
en la fase normal; el grande solo tras 500 programas sin mejora, porque en 8 GB de VRAM no
caben los dos y alternarlos obliga a recargar), aplicar los bloques SEARCH/REPLACE, evaluar en
el contenedor aislado con la cascada de 2 etapas y guardar todo en la base.
- Presupuesto: 5,000 programas. Estancamiento: 500 programas sin mejorar la mejor aptitud ->
  se pasa al modelo grande en todas las mutaciones (sustituto del modelo de API, que no está
  configurado); si tampoco mejora en otros 500, la corrida termina.
- Mismo contrato que las demás corridas: estado.json, fitness.csv (una fila cada 25
  programas), control.json, champions/ y replays/ (el campeón como código, con las líneas
  ejecutadas en cada paso para el visor).
- Se reanuda sola: la base SQLite es el estado.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from evocarros.util import reemplazar  # noqa: E402
from evocarros.llm_evo import llm  # noqa: E402
from evocarros.llm_evo.base import N_ISLAS, Base  # noqa: E402
from evocarros.llm_evo.evaluador import evaluar  # noqa: E402
from evocarros.neuro.entrenar import runs_por_defecto, temp_gpu  # noqa: E402
from evocarros.sim import repeticion as rp  # noqa: E402
from evocarros.sim.episodio import Resultado  # noqa: E402
from evocarros.sim.genoma import N_GENOMA, decodificar_cuerpo  # noqa: E402

FILA = 25
MIGRAR_CADA = 50
P_GRANDE = 0.0      # v0.9: en la GPU de 8 GB no caben ambos modelos; el grande solo al estancarse
PROTOCOLO = "v0.12"


class Evolucion:
    def __init__(self, a):
        self.a = a
        self.id = a.corrida or f"D-LLM_s{a.semilla:02d}"
        self.dir = a.runs / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tmp = self.dir / "tmp"
        self.base = Base(self.dir / "programas.sqlite")
        self.lock = threading.Lock()
        self.rng = random.Random(a.semilla * 7919 + self.base.contar())
        self.t0 = time.time()
        self.fallas_modelo = 0
        if not (self.dir / "config.json").exists():
            cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
            cfg.update(protocolo=PROTOCOLO, condicion="D-LLM", islas=N_ISLAS, p_grande=P_GRANDE,
                       migrar_cada=MIGRAR_CADA, api="no configurada (sustituto: modelo grande local)",
                       creada=datetime.now().isoformat(timespec="seconds"))
            (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        if not (self.dir / "fitness.csv").exists():
            with open(self.dir / "fitness.csv", "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["generacion", "episodios", "aptitud_max", "aptitud_media", "aptitud_mediana",
                                        "pct_vuelta", "mejor_t_vuelta", "diversidad", "t_limite", "segundos",
                                        "programas", "validos_pct"])

    # ------------------------------------------------------------ estado y control
    def orden(self) -> str:
        try:
            return json.loads((self.dir / "control.json").read_text(encoding="utf-8")).get("orden", "correr")
        except (OSError, json.JSONDecodeError):
            return "correr"

    def estado(self, estado: str, extra: dict | None = None):
        m = self.base.mejor()
        n = self.base.contar()
        ult = self.base.ultimos(100)
        d = {"corrida": self.id, "condicion": "D-LLM", "semilla": self.a.semilla, "estado": estado,
             "generacion": n // FILA, "programas": n, "programas_max": self.a.programas,
             "episodios": self.base.episodios(), "mejor_aptitud": m and m["aptitud"],
             "pct_vuelta": 100.0 * sum(u["vuelta"] for u in ult) / max(len(ult), 1),
             "mejor_t_vuelta": m and m["t_vuelta"], "modo": self.modo(), "celdas": self.base.celdas_ocupadas(),
             "eta_s": None, "pista": "pistas/mexico/pista.json",
             "actualizado": datetime.now().isoformat(timespec="seconds")}
        d.update(extra or {})
        tmp = self.dir / "estado.json.tmp"
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        reemplazar(tmp, self.dir / "estado.json")

    def sin_mejora(self) -> int:
        """Programas desde la última mejora de la mejor aptitud."""
        filas = self.base.c.execute("SELECT id, aptitud FROM programas WHERE valido = 1 ORDER BY id").fetchall()
        mejor, id_mejor = -1e9, 0
        for pid, apt in filas:
            if apt > mejor + 1e-9:
                mejor, id_mejor = apt, pid
        return self.base.desde(id_mejor)

    def modo(self) -> str:
        s = self.sin_mejora()
        return "normal" if s < self.a.estancamiento else ("grande" if s < 2 * self.a.estancamiento else "fin")

    # ------------------------------------------------------------ un programa
    def elegir(self):
        isla = self.base.contar() % N_ISLAS
        el = self.base.elites(isla) or self.base.elites()
        if not el:
            return None, isla, []
        el.sort(key=lambda e: -e["aptitud"])
        padre = el[0] if self.rng.random() < 0.7 else self.rng.choice(el)
        otros = [e for e in self.base.elites() if e["celda"] != padre["celda"] and e["id"] != padre["id"]]
        insp = self.rng.sample(otros, min(2, len(otros)))
        return padre, isla, insp

    def un_programa(self, k: int):
        t0 = time.time()
        with self.lock:
            padre, isla, insp = self.elegir()
            grande = self.modo() == "grande" or self.rng.random() < P_GRANDE
        modelo = self.a.modelo_grande if grande else self.a.modelo_chico
        if padre is None:
            codigo, desc = llm.PROGRAMA_INICIAL, "programa inicial"
            pid_padre, modelo = None, "inicial"
        else:
            try:
                resp = llm.pedir(modelo, llm.prompt(padre, insp), semilla=self.a.semilla * 100000 + k)
            except Exception as e:
                # Falla de infraestructura (Ollama caído, lento o recargando): no es un programa.
                # No se registra; se avisa en estado.json, se espera y el bucle reintenta.
                with self.lock:
                    self.fallas_modelo += 1
                    self.estado("corriendo", {"aviso": f"el modelo no responde ({self.fallas_modelo} seguidas): {e}"[:200]})
                time.sleep(min(600, 30 * self.fallas_modelo))
                return None, None
            self.fallas_modelo = 0
            codigo, desc = llm.aplicar(padre["codigo"], resp)
            pid_padre = padre["id"]
        if codigo is None:
            res = {"ok": False, "etapa": 0, "episodios": 0, "error": desc}
            codigo = padre["codigo"] if padre else ""
        else:
            res = evaluar(codigo, self.tmp, modo=self.a.sandbox)
        with self.lock:
            pid = self.base.agregar(pid_padre, isla, codigo, res, modelo, time.time() - t0)
            n = self.base.contar()
            if n % MIGRAR_CADA == 0:
                self.base.migrar()
            if n % FILA == 0:
                self.fila(n)
        return pid, res

    def fila(self, n: int):
        ult = self.base.ultimos(FILA)
        m = self.base.mejor()
        apts = sorted(u["aptitud"] for u in ult if u["valido"]) or [0.0]
        with open(self.dir / "fitness.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([n // FILA, self.base.episodios(), round(m["aptitud"], 4) if m else "",
                                    round(sum(apts) / len(apts), 4), round(apts[len(apts) // 2], 4),
                                    round(100 * sum(u["vuelta"] for u in ult) / len(ult), 1),
                                    "" if not m or m["t_vuelta"] is None else round(m["t_vuelta"], 3),
                                    self.base.celdas_ocupadas(), 250.0, round(time.time() - self.t0, 1), n,
                                    round(100 * sum(u["valido"] for u in ult) / len(ult), 1)])

    # ------------------------------------------------------------ campeón y repetición
    def campeon(self):
        m = self.base.mejor()
        if not m:
            return
        d = self.dir / "champions"
        d.mkdir(exist_ok=True)
        (d / f"prog_{m['id']:05d}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        res = evaluar(m["codigo"], self.tmp, registrar=True, modo=self.a.sandbox)
        if not res.get("registro"):
            return
        reg = res.pop("registro")
        r = Resultado(aptitud=res["aptitud"], avance_m=res["avance_m"], fraccion_vuelta=res["fraccion_vuelta"],
                      checkpoints=int(res["aptitud"]), vuelta_completa=res["vuelta_completa"], t_vuelta=res["t_vuelta"],
                      t_final=res["t_final"], motivo=res["motivo"], pasos=len(reg["x"]), registro=reg)
        c = rp.carro(f"prog_{m['id']:05d}", decodificar_cuerpo([0.0] * N_GENOMA, "A"), r,
                     codigo=m["codigo"], rango=1, lineas=reg["lineas"])
        n = self.base.contar()
        rp.guardar(self.dir / "replays" / f"gen_{n // FILA:04d}.json.gz", [c], pista="Autódromo Hermanos Rodríguez",
                   pista_archivo="pistas/mexico/pista.json", corrida=self.id, condicion="D-LLM",
                   semilla=self.a.semilla, generacion=n // FILA)

    # ------------------------------------------------------------ bucle
    def correr(self):
        if self.base.contar() == 0:
            self.un_programa(0)
            self.campeon()
        mejor_previo = (self.base.mejor() or {}).get("aptitud")
        ultimo_campeon = self.base.contar()
        with ThreadPoolExecutor(self.a.workers) as pool:
            while True:
                o = self.orden()
                if o == "detener":
                    self.estado("detenida"); return
                if o == "pausar":
                    self.estado("pausada")
                    while self.orden() == "pausar":
                        time.sleep(5)
                    continue
                g = temp_gpu()
                if g is not None and g > 85:
                    self.estado("pausada", {"motivo_pausa": f"GPU a {g:.0f} °C"}); time.sleep(60); continue
                n = self.base.contar()
                if n >= self.a.programas or self.modo() == "fin":
                    self.campeon()
                    self.estado("terminada", {"fin": "presupuesto" if n >= self.a.programas else "estancamiento"})
                    return
                lote = min(self.a.workers * 2, self.a.programas - n)
                list(pool.map(self.un_programa, range(n, n + lote)))
                m = self.base.mejor()
                if m and (mejor_previo is None or m["aptitud"] > mejor_previo + 1e-9) and self.base.contar() - ultimo_campeon >= 50:
                    self.campeon(); ultimo_campeon = self.base.contar(); mejor_previo = m["aptitud"]
                self.estado("corriendo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--corrida", default=None)
    ap.add_argument("--programas", type=int, default=5000)
    ap.add_argument("--estancamiento", type=int, default=500)
    ap.add_argument("--modelo-chico", default="qwen2.5-coder:7b")
    ap.add_argument("--modelo-grande", default="qwen3.5:9b")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sandbox", default="docker", choices=["docker", "local"])
    ap.add_argument("--runs", type=Path, default=runs_por_defecto())
    a = ap.parse_args()
    Evolucion(a).correr()


if __name__ == "__main__":
    main()
