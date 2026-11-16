"""Prueba manual de la física: manejar un carro con el teclado (fase 2).

Uso (en la AORUS, ambiente evocarros):

    python scripts/manejar_teclado.py [pistas/mexico/pista.json]

Flechas: arriba acelera, abajo frena, izquierda/derecha giran. R reinicia, Q sale.
Vista desde arriba que sigue al carro, con los 5 rayos fijos de la condición A,
velocidad, avance, checkpoint y tiempo. Aplica las mismas reglas de fin que el
episodio (choque, estancamiento, sentido contrario, vuelta) para que se sienta
exactamente lo que vivirán los carros evolucionados.

Objetivo de la prueba: que acelerar, frenar y girar se sientan razonables, que la
horquilla del Foro Sol se pueda tomar a ~50 km/h y que tocar un borde mate al carro.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from evocarros.sim import episodio as ep  # noqa: E402
from evocarros.sim.fisica import DT, FISICA, esquinas, paso  # noqa: E402
from evocarros.sim.genoma import N_GENOMA, decodificar_cuerpo  # noqa: E402
from evocarros.sim.pista import Pista  # noqa: E402

if "--sin-ventana" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class Juego:
    def __init__(self, pista: Pista):
        self.p = pista
        self.cuerpo = decodificar_cuerpo(np.zeros(N_GENOMA), "A")
        self.teclas = set()
        self.reiniciar()

    def reiniciar(self):
        self.e = ep.posicion_salida(self.p)
        self.i = self.p.indice_inicial((self.e.x, self.e.y))
        self.s_ant, self.i = self.p.progreso_local((self.e.x, self.e.y), self.i)
        self.s_ini = self.s_ant
        self.vueltas = 0
        self.avance = self.avance_max = self.s_ant
        self.hist = deque(maxlen=int(round(ep.VENTANA_ESTANCAMIENTO_S / DT)) + 1)
        self.t = 0.0
        self.motivo = None
        self.vol = 0.0

    def controles(self):
        objetivo = (1.0 if "left" in self.teclas else 0.0) - (1.0 if "right" in self.teclas else 0.0)
        self.vol += max(-0.15, min(0.15, objetivo - self.vol))   # volante con algo de inercia
        pedal = (1.0 if "up" in self.teclas else 0.0) - (1.0 if "down" in self.teclas else 0.0)
        return self.vol, pedal

    def tic(self):
        if self.motivo:
            return
        vol, ped = self.controles()
        paso(self.e, vol, ped, FISICA.v_tope)
        self.t += DT
        s, self.i = self.p.progreso_local((self.e.x, self.e.y), self.i)
        if s - self.s_ant < -0.5 * self.p.longitud:
            self.vueltas += 1
        elif s - self.s_ant > 0.5 * self.p.longitud:
            self.vueltas -= 1
        self.s_ant = s
        self.avance = s + self.vueltas * self.p.longitud
        self.avance_max = max(self.avance_max, self.avance)
        self.hist.append(self.avance_max)
        if self.p.toca_borde(esquinas(self.e)):
            self.motivo = "choque"
        elif self.avance_max - self.s_ini >= self.p.longitud:
            self.motivo = f"vuelta en {self.t:.1f} s"
        elif self.avance < self.avance_max - ep.RETROCESO_MAX_M:
            self.motivo = "sentido contrario"
        elif len(self.hist) == self.hist.maxlen and self.hist[-1] - self.hist[0] < ep.AVANCE_MIN_M:
            self.motivo = "estancamiento"

    def rayos(self):
        d = self.p.rayos((self.e.x, self.e.y), self.e.rumbo, self.cuerpo.angulos, self.cuerpo.alcances_m)
        th = self.e.rumbo + self.cuerpo.angulos
        return d, np.stack([self.e.x + d * np.cos(th), self.e.y + d * np.sin(th)], axis=1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ruta = Path(args[0]) if args else RAIZ / "pistas" / "mexico" / "pista.json"
    pista = Pista.desde_json(ruta if ruta.is_absolute() else RAIZ / ruta)
    j = Juego(pista)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.canvas.manager.set_window_title(f"evocarros: manejo manual ({pista.nombre})")
    for b in (pista.borde_izq, pista.borde_der):
        bb = np.vstack([b, b[:1]])
        ax.plot(bb[:, 0], bb[:, 1], color="0.2", lw=1.5)
    for a, b in zip(pista.cp_izq, pista.cp_der):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:red", lw=0.6, alpha=0.5)
    carro, = ax.fill([0], [0], color="tab:blue", zorder=5)
    lineas_rayos = [ax.plot([], [], color="tab:orange", lw=1)[0] for _ in range(5)]
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace",
                  bbox=dict(facecolor="white", alpha=0.85))
    ax.set_aspect("equal")
    ax.set_facecolor("#dfe8d8")

    def tecla(ev, abajo):
        k = (ev.key or "").lower()
        if abajo and k == "r":
            j.reiniciar()
        elif abajo and k == "q":
            plt.close(fig)
        elif k in ("up", "down", "left", "right"):
            (j.teclas.add if abajo else j.teclas.discard)(k)

    fig.canvas.mpl_connect("key_press_event", lambda ev: tecla(ev, True))
    fig.canvas.mpl_connect("key_release_event", lambda ev: tecla(ev, False))

    def dibujar():
        esq = esquinas(j.e)
        carro.set_xy(esq)
        d, fin = j.rayos()
        for ln, f in zip(lineas_rayos, fin):
            ln.set_data([j.e.x, f[0]], [j.e.y, f[1]])
        ax.set_xlim(j.e.x - 70, j.e.x + 70)
        ax.set_ylim(j.e.y - 70, j.e.y + 70)
        apt, ncp = ep.aptitud(pista, j.avance_max - j.s_ini, None, 250.0)
        hud.set_text(f"v {j.e.v * 3.6:6.1f} km/h   t {j.t:6.1f} s\n"
                     f"avance {j.avance_max - j.s_ini:7.1f} m   checkpoint {ncp}\n"
                     f"rayos {' '.join(f'{x:5.1f}' for x in d)}\n"
                     + (f">>> {j.motivo.upper()}   (R reinicia)" if j.motivo else "flechas: manejar   R: reiniciar   Q: salir"))
        fig.canvas.draw_idle()

    def ciclo():
        j.tic()
        dibujar()

    if "--sin-ventana" in sys.argv:   # prueba automática: acelera 3 s y guarda una captura
        j.teclas.add("up")
        for _ in range(90):
            j.tic()
        dibujar()
        fig.savefig(RAIZ / "manejar_teclado_prueba.png", dpi=80)
        print(f"OK v={j.e.v * 3.6:.1f} km/h, avance={j.avance_max - j.s_ini:.1f} m, motivo={j.motivo}")
        return
    timer = fig.canvas.new_timer(interval=int(DT * 1000))
    timer.add_callback(ciclo)
    timer.start()
    plt.show()


if __name__ == "__main__":
    main()
