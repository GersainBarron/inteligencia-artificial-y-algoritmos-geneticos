"""Evalúa un programa piloto (condición D) DENTRO del contenedor aislado.

Uso (lo llama evocarros/llm_evo/evaluador.py con docker run, sin red, solo lectura):
    python /app/sandbox/evaluar_programa.py /entrada/programa.py /entrada/peticion.json /salida/resultado.json

El programa debe definir:
    def manejar(rayos, velocidad, angulo_cp) -> (volante, pedal)
  rayos: lista de 5 distancias normalizadas en [0, 1] (1 = nada dentro del alcance), con los
         rayos fijos de la condición A (-60°, -30°, 0°, 30°, 60°; 49 m de alcance).
  velocidad: v / 50 m/s, en [0, 1].
  angulo_cp: ángulo al centro del siguiente checkpoint / pi, en [-1, 1] (+ = a la izquierda).
  volante y pedal en [-1, 1] (volante +1 = izquierda; pedal > 0 acelera, < 0 frena).
Son exactamente las 7 entradas y 2 salidas de la red 7:8:2, para que D sea comparable con A.

Cascada (protocolo v0.9): etapa 1, 45 s desde la salida (pasa si avanza >= 1,200 m, es decir,
supera la curva 1); etapa 2, vuelta completa con T_max = 250 s, con la misma aptitud que las
redes. Las pistas de prueba nunca se usan aquí.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from evocarros.sim import episodio as ep  # noqa: E402
from evocarros.sim.fisica import DT  # noqa: E402
from evocarros.sim.genoma import N_GENOMA, decodificar_cuerpo  # noqa: E402
from evocarros.sim.pista import Pista  # noqa: E402

ETAPA1_T, ETAPA1_AVANCE = 45.0, 1200.0
T_MAX = 250.0
LIMITE_PARED_S = 90.0     # tiempo real máximo de toda la evaluación


class Piloto:
    """Envuelve manejar(): recorta salidas, rechaza valores no numéricos y opcionalmente
    registra qué líneas del programa se ejecutan en cada paso (para el visor)."""

    def __init__(self, codigo: str, nombre: str, trazar: bool = False):
        self.ns = {"__name__": "piloto", "math": math}
        exec(compile(codigo, nombre, "exec"), self.ns)
        if "manejar" not in self.ns or not callable(self.ns["manejar"]):
            raise ValueError("el programa no define manejar(rayos, velocidad, angulo_cp)")
        self.f = self.ns["manejar"]
        self.nombre = nombre
        self.trazar = trazar
        self.lineas: list[list[int]] = []
        self.t0 = time.time()

    def _traza(self, frame, evento, arg):
        if frame.f_code.co_filename == self.nombre and evento == "line":
            self._actual.append(frame.f_lineno)
        return self._traza

    def __call__(self, entradas):
        if time.time() - self.t0 > LIMITE_PARED_S:
            raise TimeoutError("el programa tardó demasiado")
        args = ([float(x) for x in entradas[:5]], float(entradas[5]), float(entradas[6]))
        if self.trazar:
            self._actual = []
            sys.settrace(self._traza)
            try:
                out = self.f(*args)
            finally:
                sys.settrace(None)
            self.lineas.append(sorted(set(self._actual)))
        else:
            out = self.f(*args)
        vol, ped = float(out[0]), float(out[1])
        if not (math.isfinite(vol) and math.isfinite(ped)):
            raise ValueError("manejar devolvió un valor no numérico")
        return np.array([max(-1.0, min(1.0, vol)), max(-1.0, min(1.0, ped))])


def retro(pista: Pista, r: ep.Resultado) -> dict:
    """Métricas y retroalimentación en texto para el prompt."""
    reg = r.registro
    v = np.array(reg["v"]) if reg and reg["v"] else np.zeros(1)
    vol = np.array([s[0] for s in reg["salidas"]]) if reg and reg["salidas"] else np.zeros(1)
    suav = float(np.mean(np.abs(np.diff(vol)))) if len(vol) > 1 else 0.0
    d = {"velocidad_media_kmh": round(float(v.mean()) * 3.6, 1), "suavidad_volante": round(suav, 4)}
    paso_cp = pista.longitud / len(pista.cp_s)
    if r.vuelta_completa:
        d["texto"] = (f"Completó la vuelta en {r.t_vuelta:.1f} s (velocidad media {d['velocidad_media_kmh']} km/h). "
                      f"Para mejorar hay que bajar el tiempo de vuelta sin chocar.")
    else:
        k = len(reg["x"]) - 1
        ent = reg["entradas"][k]
        d["texto"] = (f"Terminó por {r.motivo.replace('_', ' ')} a los {r.t_final:.1f} s, en s = {r.avance_m:.0f} m "
                      f"(checkpoint {int(r.avance_m // paso_cp)} de {len(pista.cp_s)}, {100 * r.fraccion_vuelta:.0f}% de la vuelta), "
                      f"a {reg['v'][k] * 3.6:.0f} km/h. Rayos en ese momento: "
                      f"{', '.join(f'{x:.2f}' for x in ent[:5])}; angulo_cp {ent[6]:+.2f}; "
                      f"última salida volante {reg['salidas'][k][0]:+.2f}, pedal {reg['salidas'][k][1]:+.2f}.")
    return d


def evaluacion_final(codigo: str, peticion: dict) -> dict:
    """Misma evaluación final que las redes (scripts/evaluar_campeon.py): por pista, salida
    estándar + salidas perturbadas (lateral, giro en radianes) dadas por el anfitrión."""
    cuerpo = decodificar_cuerpo(np.zeros(N_GENOMA), "A")
    out = {}
    for nombre, d in peticion["pistas"].items():
        pista = Pista.desde_json(d["ruta"])
        filas = []
        for k, (lat, dr) in enumerate(peticion["perturbaciones"]):
            s = ep.posicion_salida(pista, lateral=lat, d_rumbo=dr)
            try:
                r = ep.correr(pista, Piloto(codigo, "programa.py"), cuerpo, t_limite=d["t_max"], t_max=d["t_max"], salida=s)
                filas.append({"salida": k, "completa": r.vuelta_completa, "t_vuelta": r.t_vuelta,
                              "fraccion_vuelta": min(r.fraccion_vuelta, 1.0), "motivo": r.motivo})
            except Exception as e:
                filas.append({"salida": k, "completa": False, "t_vuelta": None, "fraccion_vuelta": 0.0,
                              "motivo": f"error: {type(e).__name__}"})
        out[nombre] = filas
    return out


def main():
    prog, pet, sal = (Path(a) for a in sys.argv[1:4])
    peticion = json.loads(pet.read_text(encoding="utf-8"))
    codigo = prog.read_text(encoding="utf-8")
    if peticion.get("evaluacion_final"):
        sal.write_text(json.dumps({"ok": True, "pistas": evaluacion_final(codigo, peticion)}), encoding="utf-8")
        return
    pista = Pista.desde_json(peticion.get("pista", "/app/pistas/mexico/pista.json"))
    cuerpo = decodificar_cuerpo(np.zeros(N_GENOMA), "A")      # rayos fijos y 35 m/s, como A
    trazar = bool(peticion.get("registrar", False))
    out = {"ok": False, "etapa": 0, "episodios": 0}
    try:
        piloto = Piloto(codigo, "programa.py")
        r1 = ep.correr(pista, piloto, cuerpo, t_limite=ETAPA1_T, t_max=T_MAX, registrar=True)
        out.update(etapa=1, episodios=1)
        r = r1
        if r1.avance_m >= ETAPA1_AVANCE or r1.vuelta_completa:
            piloto = Piloto(codigo, "programa.py", trazar=trazar)
            r = ep.correr(pista, piloto, cuerpo, t_limite=T_MAX, t_max=T_MAX, registrar=True)
            out.update(etapa=2, episodios=2)
        out.update(ok=True, aptitud=r.aptitud, avance_m=r.avance_m, fraccion_vuelta=r.fraccion_vuelta,
                   vuelta_completa=r.vuelta_completa, t_vuelta=r.t_vuelta, motivo=r.motivo, t_final=r.t_final,
                   **retro(pista, r))
        if trazar and out["etapa"] == 2:
            reg = r.registro
            out["registro"] = {"x": reg["x"], "y": reg["y"], "rumbo": reg["rumbo"], "v": reg["v"],
                               "entradas": [list(map(float, e)) for e in reg["entradas"]],
                               "salidas": [list(map(float, s)) for s in reg["salidas"]],
                               "lineas": piloto.lineas}
    except Exception as e:  # cualquier error del programa es un programa inválido, no un fallo del sistema
        out.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
    sal.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
