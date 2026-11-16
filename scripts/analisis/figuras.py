"""Figuras y tablas derivadas de la fase 8 (protocolo v0.11/v0.12). Solo lee.

Uso:  python scripts/analisis/figuras.py
Lee resultados/resumen_redes_v011.csv, resultados/ablacion.csv y el fitness.csv de cada
corrida. Escribe en resultados/figuras/:
  fig1_curvas_aptitud.png        mediana y rango intercuartil de la aptitud máxima contra episodios
  fig2_primera_vuelta.png        fracción acumulada de corridas que ya completaron una vuelta
  fig3_polar_rayos.png           rayos de los campeones (ángulo y alcance)
  fig4_generalizacion.png        vueltas completas y avance mediano por pista
  fig5_robustez.png              % de salidas perturbadas completadas, por corrida
  fig6_tiempo_normalizado.png    tiempo de vuelta / vuelta ideal para la v_max del campeón
  fig7_ablacion.png              caída de aptitud al apagar cada rayo, según su ángulo
y en resultados/: tiempo_normalizado.csv y ablacion_resumen.csv.
Colores: uno por cuerpo (A azul, B naranja, C verde agua, R amarillo), igual en todas
las figuras; GA y ES van en paneles separados, no en más colores.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "scripts"))
from evocarros.neuro.entrenar import runs_por_defecto  # noqa: E402
import perfil_velocidad as pv  # noqa: E402

RES, FIG = RAIZ / "resultados", RAIZ / "resultados" / "figuras"
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "R": "#eda100"}
NOMBRE = {"A": "A: rayos fijos de diseño", "B": "B: rayos evolutivos", "C": "C: rayos y velocidad evolutivos",
          "R": "R: rayos fijos al azar"}
TINTA, TINTA2, TENUE, REJILLA, EJE, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
AZULES = LinearSegmentedColormap.from_list("azules", ["#f0efec", "#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])
ORDEN = ["A-GA", "B-GA", "C-GA", "R-GA", "A-ES", "B-ES", "C-ES"]
PISTAS = [("mexico", "México (entrenamiento)"), ("spa", "Spa"), ("monza", "Monza"), ("hungaroring", "Hungaroring")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": EJE, "axes.labelcolor": TINTA2,
                     "xtick.color": TENUE, "ytick.color": TENUE, "axes.facecolor": FONDO, "figure.facecolor": FONDO,
                     "savefig.facecolor": FONDO, "axes.titlesize": 11, "axes.titlecolor": TINTA,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": REJILLA,
                     "grid.linewidth": 0.8, "legend.frameon": False})


def guardar(fig, nombre):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / nombre, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("figura:", nombre)


def curvas(df):
    runs = runs_por_defecto()
    rej = np.arange(0, 100_001, 250)
    serie = {}
    for cid in df.corrida:
        f = pd.read_csv(runs / cid / "fitness.csv")
        e, a = f.episodios.to_numpy(), f.aptitud_max.to_numpy()
        idx = np.searchsorted(e, rej, side="right") - 1          # último registro con episodios <= x
        serie[cid] = np.where(idx >= 0, a[np.clip(idx, 0, None)], np.nan)   # tras converger conserva el último valor
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, alg, n in zip(ejes, ("GA", "ES"), (20, 10)):
        for cuerpo in "ABCR":
            ids = df[(df.algoritmo == alg) & (df.cuerpo == cuerpo)].corrida
            if not len(ids):
                continue
            m = np.array([serie[c] for c in ids])
            q1, med, q3 = np.nanpercentile(m, [25, 50, 75], axis=0)
            ax.fill_between(rej / 1000, q1, q3, color=COLOR[cuerpo], alpha=0.15, linewidth=0)
            ax.plot(rej / 1000, med, color=COLOR[cuerpo], linewidth=2, label=NOMBRE[cuerpo])
        ax.axhline(50, color=TENUE, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(99, 50.6, "vuelta completa (aptitud ≥ 50)", ha="right", va="bottom", color=TINTA2, fontsize=8.5)
        ax.set_title(f"{'Algoritmo genético' if alg == 'GA' else 'Estrategias evolutivas (ES)'}, {n} semillas por condición", loc="left")
        ax.set_xlabel("Episodios (miles)"); ax.set_xlim(0, 100); ax.set_ylim(0, 58)
    ejes[0].set_ylabel("Aptitud máxima de la población")
    h, l = ejes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Aptitud contra episodios: mediana (línea) y rango intercuartil (banda)", x=0.01, ha="left", fontsize=12, color=TINTA)
    guardar(fig, "fig1_curvas_aptitud.png")


def primera_vuelta(df):
    rej = np.arange(0, 100_001, 250)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, alg, n in zip(ejes, ("GA", "ES"), (20, 10)):
        for cuerpo in "ABCR":
            g = df[(df.algoritmo == alg) & (df.cuerpo == cuerpo)]
            if not len(g):
                continue
            t = g.episodios_primera_vuelta.fillna(np.inf).to_numpy()
            ax.step(rej / 1000, [100 * np.mean(t <= x) for x in rej], where="post", color=COLOR[cuerpo], linewidth=2, label=NOMBRE[cuerpo])
        ax.set_title(f"{'Algoritmo genético' if alg == 'GA' else 'Estrategias evolutivas (ES)'}, {n} semillas", loc="left")
        ax.set_xlabel("Episodios (miles)"); ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ejes[0].set_ylabel("% de corridas con al menos una vuelta")
    ejes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Fracción acumulada de corridas que ya lograron la primera vuelta", x=0.01, ha="left", fontsize=12, color=TINTA)
    guardar(fig, "fig2_primera_vuelta.png")


def polar(df):
    paneles = [("B-GA", "B-GA: rayos evolutivos (20 campeones)"), ("C-GA", "C-GA: rayos y velocidad (20)"),
               ("R-GA", "R-GA: fijos al azar, no evolucionan (20)"), ("B-ES", "B-ES (10)"), ("C-ES", "C-ES (10)")]
    fig, ejes = plt.subplots(2, 3, figsize=(12, 8.4), subplot_kw={"projection": "polar"})
    ejes = ejes.ravel()
    ini_a, ini_r = np.radians([-60, -30, 0, 30, 60]), 4.0 * 12.249
    for ax, (cond, titulo) in zip(ejes, paneles):
        g = df[df.condicion == cond]
        for ang, alc in zip(g.genes_ang_rad, g.genes_alc_W):
            a = (np.array(json.loads(ang)) + math.pi) % (2 * math.pi) - math.pi
            r = np.clip(np.array(json.loads(alc)), 0.5, 8.0) * 12.249
            for ai, ri in zip(a, r):
                ax.plot([ai, ai], [0, ri], color=COLOR[cond[0]], linewidth=1, alpha=0.45)
            ax.scatter(a, r, s=9, color=COLOR[cond[0]], zorder=3)
        for ai in ini_a:
            ax.plot([ai, ai], [0, ini_r], color=TINTA, linewidth=1.6, linestyle=(0, (3, 2)))
        ax.set_title(titulo, loc="left", pad=14)
    ax = ejes[5]
    ax.set_axis_off()
    ax.text(0.0, 0.85, "Cómo leerla", transform=ax.transAxes, fontsize=11, color=TINTA)
    ax.text(0.0, 0.2, "Cada línea es un rayo de un campeón:\nla dirección es su ángulo respecto al\nfrente del carro (0° arriba; positivo =\nizquierda) y el largo es su alcance en\nmetros (máximo 98 m = 8 W).\n\nLíneas negras punteadas: configuración\ninicial común (−60° a 60°, 49 m),\nque es también la fija de la condición A.",
            transform=ax.transAxes, fontsize=9.5, color=TINTA2, va="bottom", linespacing=1.5)
    for ax in ejes[:5]:
        ax.set_theta_zero_location("N"); ax.set_theta_direction(1)
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], ["0°", "45°", "90°", "135°", "±180°", "−135°", "−90°", "−45°"])
        ax.set_rlim(0, 100); ax.set_rticks([25, 50, 75, 100]); ax.set_rlabel_position(157.5)
        ax.tick_params(labelsize=8); ax.spines["polar"].set_color(EJE)
    fig.suptitle("Rayos de los campeones: ángulo y alcance", x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    guardar(fig, "fig3_polar_rayos.png")


def generalizacion(df):
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, modo in zip(ejes, ("completa", "fraccion")):
        M, T = np.zeros((len(ORDEN), 4)), [[""] * 4 for _ in ORDEN]
        for i, cond in enumerate(ORDEN):
            g = df[df.condicion == cond]
            for j, (p, _) in enumerate(PISTAS):
                if modo == "completa":
                    k = int(g[f"{p}_completa"].astype(bool).sum())
                    M[i, j], T[i][j] = 100 * k / len(g), f"{k} de {len(g)}"
                else:
                    M[i, j] = 100 * g[f"{p}_fraccion"].median(); T[i][j] = f"{M[i, j]:.0f}%"
        ax.imshow(M, cmap=AZULES, vmin=0, vmax=100, aspect="auto")
        for i in range(len(ORDEN)):
            for j in range(4):
                ax.text(j, i, T[i][j], ha="center", va="center", fontsize=9.5, color="#ffffff" if M[i, j] > 55 else TINTA)
        ax.set_xticks(range(4), [n for _, n in PISTAS]); ax.set_yticks(range(len(ORDEN)), ORDEN)
        ax.tick_params(length=0, labelcolor=TINTA2); ax.grid(False)
        ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, len(ORDEN)), minor=True)
        ax.grid(which="minor", color=FONDO, linewidth=2); ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Campeones que completan la vuelta (salida estándar)" if modo == "completa"
                     else "Avance mediano del campeón (% de la vuelta)", loc="left")
    fig.suptitle("Generalización: campeones entrenados solo en México, evaluados sin reentrenar", x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    guardar(fig, "fig4_generalizacion.png")


def tira(ax, df, col, conds, rng):
    for i, cond in enumerate(conds):
        v = df[df.condicion == cond][col].dropna().to_numpy(float)
        if not len(v):
            continue
        x = i + rng.uniform(-0.22, 0.22, len(v))
        ax.scatter(x, v, s=26, color=COLOR[cond[0]], alpha=0.8, edgecolor=FONDO, linewidth=0.8, zorder=3)
        ax.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color=TINTA, linewidth=2, zorder=4)
    ax.set_xticks(range(len(conds)), conds); ax.tick_params(axis="x", length=0, labelcolor=TINTA2)
    ax.grid(axis="x", visible=False)


def robustez(df):
    rng = np.random.default_rng(1)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (p, n) in zip(ejes, (PISTAS[0], PISTAS[3])):
        tira(ax, df, f"{p}_pct_perturbadas", ORDEN, rng)
        ax.set_title(n, loc="left"); ax.set_ylim(-4, 104)
    ejes[0].set_ylabel("% de 20 salidas perturbadas completadas")
    fig.suptitle("Robustez: cada punto es el campeón de una corrida; la raya negra es la mediana", x=0.01, ha="left", fontsize=12, color=TINTA)
    guardar(fig, "fig5_robustez.png")


def tiempo_normalizado(df):
    ruta, cache = str(RAIZ / "pistas/mexico/pista.json"), {}

    def ideal(v):
        v = round(float(v), 2)
        if v not in cache:
            pv.V_MAX = v
            cache[v] = pv.perfil(ruta)["t_vuelta_ideal_s"]
            pv.V_MAX = 50.0
        return cache[v]

    d = df[df.mejor_t_vuelta.notna()].copy()
    d["v_max"] = np.where(d.cuerpo == "C", d.v_max_gen.clip(15, 50), 35.0)
    d["t_ideal"] = [ideal(v) for v in d.v_max]
    d["t_normalizado"] = d.mejor_t_vuelta / d.t_ideal
    d[["corrida", "condicion", "semilla", "v_max", "mejor_t_vuelta", "t_ideal", "t_normalizado"]].to_csv(RES / "tiempo_normalizado.csv", index=False)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    rng = np.random.default_rng(1)
    tira(ejes[0], d, "mejor_t_vuelta", ORDEN, rng)
    ejes[0].set_title("Tiempo de vuelta crudo (s)", loc="left"); ejes[0].set_ylabel("segundos")
    tira(ejes[1], d, "t_normalizado", ORDEN, rng)
    ejes[1].axhline(1.0, color=TENUE, linewidth=1, linestyle=(0, (4, 3)))
    ejes[1].set_title("Tiempo ÷ vuelta ideal por la línea central para la v_max de ese campeón", loc="left")
    for ax in ejes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.08 * (hi - lo), hi)
        for i, cond in enumerate(ORDEN):
            ax.annotate(f"n = {int((d.condicion == cond).sum())}", (i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=TENUE)
    fig.suptitle("Tiempos de vuelta, solo corridas que completan (descriptivo)", x=0.01, ha="left", fontsize=12, color=TINTA)
    guardar(fig, "fig6_tiempo_normalizado.png")
    return d


def ablacion():
    a = pd.read_csv(RES / "ablacion.csv")
    base = a[a.rayo == -1].set_index("corrida")
    r = a[a.rayo >= 0].copy()
    r["completa_intacto"] = r.corrida.map(base.pct_vueltas) > 0
    r["sector"] = pd.cut(r.angulo_grados.abs(), [-0.1, 30, 90, 180.1], labels=["frontal (|θ| ≤ 30°)", "lateral (30° a 90°)", "trasero (> 90°)"])
    filas = []
    for (cond, sec), g in r[r.completa_intacto].groupby(["condicion", "sector"], observed=True):
        q = np.percentile(g.caida_aptitud, [25, 50, 75])
        filas.append({"condicion": cond, "sector": sec, "n_rayos": len(g), "caida_aptitud_q1": q[0], "caida_aptitud_mediana": q[1],
                      "caida_aptitud_q3": q[2], "pct_rayos_con_caida_mayor_a_5": 100 * float((g.caida_aptitud > 5).mean()),
                      "caida_pct_vueltas_media": float(g.caida_pct_vueltas.mean())})
    pd.DataFrame(filas).to_csv(RES / "ablacion_resumen.csv", index=False)
    paneles = ["B-GA", "C-GA", "A-GA", "R-GA", "B-ES", "C-ES"]
    fig, ejes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, cond in zip(ejes.ravel(), paneles):
        g = r[r.condicion == cond]
        s, n = g[g.completa_intacto], g[~g.completa_intacto]
        ax.scatter(n.angulo_grados, n.caida_aptitud, s=22, facecolor="none", edgecolor=TENUE, linewidth=1, label="campeón que no completa la vuelta")
        ax.scatter(s.angulo_grados, s.caida_aptitud, s=24, color=COLOR[cond[0]], alpha=0.8, edgecolor=FONDO, linewidth=0.6, label="campeón que completa la vuelta")
        ax.axhline(0, color=EJE, linewidth=1)
        ax.set_title(f"{cond} ({g.corrida.nunique()} campeones × 5 rayos)", loc="left")
        ax.set_xlim(-185, 185); ax.set_xticks([-180, -90, -30, 0, 30, 90, 180])
    ejes[1, 1].set_xlabel("Ángulo del rayo apagado (°; 0 = frente, positivo = izquierda)")
    for ax in ejes[:, 0]:
        ax.set_ylabel("Caída de aptitud media (21 salidas)")
    h, l = ejes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Ablación: cuánto cae la aptitud al fijar en 1 la lectura de un rayo", x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    guardar(fig, "fig7_ablacion.png")


def main():
    df = pd.read_csv(RES / "resumen_redes_v011.csv")
    curvas(df); primera_vuelta(df); polar(df); generalizacion(df); robustez(df); tiempo_normalizado(df); ablacion()


if __name__ == "__main__":
    main()
