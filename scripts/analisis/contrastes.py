"""Contrastes de la fase 8 según el plan de análisis del protocolo v0.11 (vigente v0.12).

Uso:  python scripts/analisis/contrastes.py
Lee resultados/resumen_redes_v011.csv y resultados/choques_tempranos.csv.
Escribe en resultados/: contrastes_confirmatorios.csv, contrastes_exploratorios.csv,
descriptivos.csv, generalizacion.csv y contrastes_v011.md (resumen legible).

Confirmatorio (brazo genético, 20 semillas, seis pruebas bilaterales con Holm, alfa 0.05):
  1 H1   B-GA vs A-GA, aptitud final            Mann-Whitney U
  2 H1c  B-GA vs R-GA, aptitud final            Mann-Whitney U
  3 H2a  campeones B-GA: alcance del rayo de menor |ángulo| menos mediana de los otros 4
         Wilcoxon de rangos con signo contra 0
  4 H2b  campeones B-GA: norma del desplazamiento medio de los 10 genes de rayos, en
         unidades de mutación (10° y 0.2 W); permutación por cambio de signo por semilla
  5 H3a  C-GA vs B-GA, mejor tiempo de vuelta (no completar = peor rango, empatadas)
  6 H3b  C-GA vs B-GA, fracción de choques en los primeros 10,000 episodios
Valores p de Mann-Whitney: exactos sin empates; con empates, permutación (1,000,000).
Tamaño de efecto: delta de Cliff (primer grupo menos segundo) con IC 95% bootstrap
(10,000), diferencia de Hodges-Lehmann, mediana y cuartiles. Sensibilidad: Wilcoxon
pareado por semilla y los mismos contrastes solo con semillas 1-10 y solo 11-20.
Todo lo demás es exploratorio (p sin ajustar).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "resultados"
SEMILLA_AZAR = 2026
T_NO_COMPLETA = 251.0        # s; mayor que cualquier vuelta válida (T_max = 250 s): peor rango
EP_CENSURA = 100_001         # episodios; mayor que el presupuesto: peor rango
ANG_INI = np.radians([-60.0, -30.0, 0.0, 30.0, 60.0])
ALC_INI, SIG_ANG, SIG_ALC = 4.0, math.radians(10.0), 0.2


# ------------------------------------------------------------------ estadística
def cliff(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x[:, None] - y[None, :]
    return float((np.sum(d > 0) - np.sum(d < 0)) / d.size)


def cliff_ic(x, y, n=10_000, seed=SEMILLA_AZAR):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs = x[rng.integers(0, len(x), (n, len(x)))]
    ys = y[rng.integers(0, len(y), (n, len(y)))]
    d = xs[:, :, None] - ys[:, None, :]
    v = (np.sum(d > 0, axis=(1, 2)) - np.sum(d < 0, axis=(1, 2))) / (len(x) * len(y))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def hodges_lehmann(x, y):
    return float(np.median(np.asarray(x, float)[:, None] - np.asarray(y, float)[None, :]))


def p_mwu(x, y, n=1_000_000, seed=SEMILLA_AZAR):
    """p bilateral de Mann-Whitney: exacta sin empates; con empates, permutación de etiquetas."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = np.concatenate([x, y])
    if len(np.unique(z)) == len(z):
        return float(stats.mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue), "exacta"
    r = stats.rankdata(z)
    nx, mu = len(x), len(x) * (len(z) + 1) / 2
    obs = abs(r[:nx].sum() - mu)
    rng = np.random.default_rng(seed)
    cuenta, hechos = 0, 0
    while hechos < n:
        k = min(100_000, n - hechos)
        perm = rng.permuted(np.tile(r, (k, 1)), axis=1)[:, :nx].sum(axis=1)
        cuenta += int(np.sum(np.abs(perm - mu) >= obs - 1e-9))
        hechos += k
    return (cuenta + 1) / (n + 1), "permutación"


def cuartiles(v):
    v = np.asarray(v, float)
    return [float(np.percentile(v, q)) for q in (25, 50, 75)]


def p_wilcoxon(d):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def holm(ps):
    ps = np.asarray(ps, float)
    orden = np.argsort(ps)
    aj = np.empty(len(ps))
    m, previo = len(ps), 0.0
    for k, i in enumerate(orden):
        previo = max(previo, min(1.0, (m - k) * ps[i]))
        aj[i] = previo
    return aj


def dos_grupos(df, c1, c2, col, semillas, etiqueta, hl=True):
    a = df[(df.condicion == c1) & df.semilla.isin(semillas)].sort_values("semilla")
    b = df[(df.condicion == c2) & df.semilla.isin(semillas)].sort_values("semilla")
    x, y = a[col].to_numpy(float), b[col].to_numpy(float)
    p, metodo = p_mwu(x, y)
    lo, hi = cliff_ic(x, y)
    comunes = sorted(set(a.semilla) & set(b.semilla))
    dif = a.set_index("semilla").loc[comunes, col].to_numpy(float) - b.set_index("semilla").loc[comunes, col].to_numpy(float)
    q1, q2 = cuartiles(x), cuartiles(y)
    return {"prueba": etiqueta, "contraste": f"{c1} vs {c2}", "metrica": col, "n1": len(x), "n2": len(y),
            "mediana1": q1[1], "q1_1": q1[0], "q3_1": q1[2], "mediana2": q2[1], "q1_2": q2[0], "q3_2": q2[2],
            "delta_cliff": cliff(x, y), "ic95_inf": lo, "ic95_sup": hi,
            "hodges_lehmann": hodges_lehmann(x, y) if hl else float("nan"),
            "p": p, "metodo_p": metodo, "p_wilcoxon_pareado": p_wilcoxon(dif)}


# ------------------------------------------------------------------ H2
def rayos(df, cond, semillas):
    d = df[(df.condicion == cond) & df.semilla.isin(semillas)].sort_values("semilla")
    ang = np.array([json.loads(s) for s in d.genes_ang_rad])
    alc = np.clip(np.array([json.loads(s) for s in d.genes_alc_W]), 0.5, 8.0)
    return ang, alc


def h2a(ang, alc):
    env = (ang + math.pi) % (2 * math.pi) - math.pi
    out = []
    for a, r in zip(env, alc):
        k = int(np.argmin(np.abs(a)))
        out.append(r[k] - np.median(np.delete(r, k)))
    return np.array(out)             # en W


def h2b(ang, alc, n=100_000, seed=SEMILLA_AZAR):
    d = np.hstack([(ang - ANG_INI) / SIG_ANG, (alc - ALC_INI) / SIG_ALC])     # semillas x 10
    obs = float(np.sqrt((d.mean(axis=0) ** 2).sum()))
    rng = np.random.default_rng(seed)
    signos = rng.choice([-1.0, 1.0], size=(n, len(d)))
    # einsum y no '@': en este ambiente el producto matricial grande de BLAS termina el proceso (0xc06d007f)
    medias = np.einsum("ns,sk->nk", signos, d) / len(d)
    nulo = np.sqrt((medias ** 2).sum(axis=1))
    return obs, (int(np.sum(nulo >= obs - 1e-12)) + 1) / (n + 1), d.mean(axis=0)


def fila_h2(df, cond, semillas, sufijo=""):
    ang, alc = rayos(df, cond, semillas)
    D = h2a(ang, alc)
    q = cuartiles(D)
    fa = {"prueba": "H2a" + sufijo, "contraste": f"campeones {cond}", "metrica": "alcance frontal - mediana otros (W)",
          "n1": len(D), "mediana1": q[1], "q1_1": q[0], "q3_1": q[2], "p": p_wilcoxon(D),
          "metodo_p": "Wilcoxon con signo", "n_positivos": int(np.sum(D > 0))}
    obs, p, media = h2b(ang, alc)
    fb = {"prueba": "H2b" + sufijo, "contraste": f"campeones {cond}", "metrica": "norma del desplazamiento medio (sigmas de mutación)",
          "n1": len(ang), "mediana1": obs, "p": p, "metodo_p": "cambio de signo (100,000)",
          "desplazamiento_medio": json.dumps([round(float(v), 3) for v in media])}
    return fa, fb


def fisher(df, c1, c2, col, semillas, etiqueta):
    a = df[(df.condicion == c1) & df.semilla.isin(semillas)][col].astype(bool)
    b = df[(df.condicion == c2) & df.semilla.isin(semillas)][col].astype(bool)
    t = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
    return {"prueba": etiqueta, "contraste": f"{c1} vs {c2}", "metrica": col, "n1": len(a), "n2": len(b),
            "mediana1": t[0][0], "mediana2": t[1][0], "p": float(stats.fisher_exact(t)[1]), "metodo_p": "Fisher (cuentas en mediana1/2)"}


# ------------------------------------------------------------------ principal
def main():
    df = pd.read_csv(RES / "resumen_redes_v011.csv")
    ch = pd.read_csv(RES / "choques_tempranos.csv")
    assert ch.reproduce_oficial.all(), "hay re-ejecuciones que no reproducen la corrida oficial"
    df = df.merge(ch[["corrida", "frac_choque"]], on="corrida", how="left")
    cuenta = df.groupby("condicion").size().to_dict()
    assert all(cuenta[c] == 20 for c in ("A-GA", "B-GA", "C-GA", "R-GA")) and all(cuenta[c] == 10 for c in ("A-ES", "B-ES", "C-ES")), cuenta
    df["completa_entrenamiento"] = df.mejor_t_vuelta.notna()
    df["t_vuelta_rango"] = df.mejor_t_vuelta.fillna(T_NO_COMPLETA)
    df["ep_primera_rango"] = df.episodios_primera_vuelta.fillna(EP_CENSURA)
    df["ep_95_rango"] = df.episodios_95pct_vuelta.fillna(EP_CENSURA)
    S20, S10, S11 = range(1, 21), range(1, 11), range(11, 21)

    # ---- confirmatorio
    conf = [dos_grupos(df, "B-GA", "A-GA", "mejor_aptitud", S20, "1 H1"),
            dos_grupos(df, "B-GA", "R-GA", "mejor_aptitud", S20, "2 H1 control")]
    fa, fb = fila_h2(df, "B-GA", S20)
    fa["prueba"], fb["prueba"] = "3 H2a", "4 H2b"
    conf += [fa, fb,
             dos_grupos(df, "C-GA", "B-GA", "t_vuelta_rango", S20, "5 H3a", hl=False),
             dos_grupos(df, "C-GA", "B-GA", "frac_choque", S20, "6 H3b")]
    conf = pd.DataFrame(conf)
    conf["p_holm"] = holm(conf.p)
    conf["direccion_predicha"] = ["delta > 0", "delta > 0", "mediana > 0", "(sin signo)", "delta < 0", "delta > 0"]
    ok = [conf.delta_cliff[0] > 0, conf.delta_cliff[1] > 0, conf.mediana1[2] > 0, True, conf.delta_cliff[4] < 0, conf.delta_cliff[5] > 0]
    conf["direccion_observada_ok"] = ok
    conf["apoyada"] = (conf.p_holm < 0.05) & conf.direccion_observada_ok
    conf.to_csv(RES / "contrastes_confirmatorios.csv", index=False)

    # ---- exploratorio
    ex = []
    for nombre, S in (("semillas 1-10", S10), ("semillas 11-20", S11)):
        ex += [dos_grupos(df, "B-GA", "A-GA", "mejor_aptitud", S, f"H1 ({nombre})"),
               dos_grupos(df, "B-GA", "R-GA", "mejor_aptitud", S, f"H1 control ({nombre})"),
               dos_grupos(df, "C-GA", "B-GA", "t_vuelta_rango", S, f"H3a ({nombre})", hl=False),
               dos_grupos(df, "C-GA", "B-GA", "frac_choque", S, f"H3b ({nombre})")]
        ex += list(fila_h2(df, "B-GA", S, f" ({nombre})"))
    ex += [dos_grupos(df, "B-ES", "A-ES", "mejor_aptitud", S10, "H1 réplica ES"),
           dos_grupos(df, "C-ES", "B-ES", "t_vuelta_rango", S10, "H3a réplica ES", hl=False),
           dos_grupos(df, "C-ES", "B-ES", "frac_choque", S10, "H3b réplica ES")]
    ex += list(fila_h2(df, "B-ES", S10, " réplica ES"))
    ex += list(fila_h2(df, "C-GA", S20, " en C-GA"))
    ex += list(fila_h2(df, "C-ES", S10, " en C-ES"))
    ex.append(dos_grupos(df, "A-GA", "R-GA", "mejor_aptitud", S20, "diseño a mano vs azar"))
    for cuerpo in "ABC":
        for col, hl in (("ep_primera_rango", False), ("ep_95_rango", False), ("mejor_aptitud", True)):
            ex.append(dos_grupos(df, f"{cuerpo}-ES", f"{cuerpo}-GA", col, S10, f"H4 cuerpo {cuerpo}", hl=hl))
    for c1, c2, S in (("B-GA", "A-GA", S20), ("B-GA", "R-GA", S20), ("C-GA", "B-GA", S20), ("A-GA", "R-GA", S20),
                      ("A-ES", "A-GA", S10), ("B-ES", "B-GA", S10), ("C-ES", "C-GA", S10)):
        ex.append(fisher(df, c1, c2, "completa_entrenamiento", S, "tasa de vuelta (entrenamiento)"))
        ex.append(fisher(df, c1, c2, "mexico_completa", S, "tasa de vuelta (evaluación final)"))
    ex = pd.DataFrame(ex)
    ex.to_csv(RES / "contrastes_exploratorios.csv", index=False)

    # ---- descriptivos
    desc = []
    for cond, g in df.groupby("condicion"):
        comp = g[g.completa_entrenamiento]
        f = {"condicion": cond, "n": len(g), "completan": len(comp), "fin_convergencia": int((g.fin == "convergencia").sum())}
        for col, sub in (("mejor_aptitud", g), ("mejor_t_vuelta", comp), ("episodios_primera_vuelta", comp),
                         ("episodios", g), ("frac_choque", g.dropna(subset=["frac_choque"])), ("v_max_gen", g)):
            if len(sub):
                q = cuartiles(sub[col])
                f.update({f"{col}_q1": q[0], f"{col}_mediana": q[1], f"{col}_q3": q[2]})
        desc.append(f)
    desc = pd.DataFrame(desc)
    desc.to_csv(RES / "descriptivos.csv", index=False)

    gen = []
    for cond, g in df.groupby("condicion"):
        for p in ("mexico", "spa", "monza", "hungaroring"):
            q = cuartiles(g[f"{p}_fraccion"]); qp = cuartiles(g[f"{p}_pct_perturbadas"])
            gen.append({"condicion": cond, "pista": p, "n": len(g), "completan_salida_estandar": int(g[f"{p}_completa"].astype(bool).sum()),
                        "fraccion_q1": q[0], "fraccion_mediana": q[1], "fraccion_q3": q[2],
                        "pct_perturbadas_q1": qp[0], "pct_perturbadas_mediana": qp[1], "pct_perturbadas_q3": qp[2],
                        "pct_perturbadas_media": float(g[f"{p}_pct_perturbadas"].mean())})
    gen = pd.DataFrame(gen)
    gen.to_csv(RES / "generalizacion.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
    with open(RES / "contrastes_v011.md", "w", encoding="utf-8") as f:
        f.write("# Contrastes de la fase 8 (plan v0.11)\n\nGenerado por scripts/analisis/contrastes.py.\n\n## Confirmatorios (Holm, 6 pruebas)\n\n")
        f.write(conf.round(5).to_string(index=False) + "\n\n## Exploratorios (p sin ajustar)\n\n")
        f.write(ex.round(5).to_string(index=False) + "\n\n## Descriptivos por condición\n\n")
        f.write(desc.round(4).T.to_string() + "\n\n## Generalización y robustez\n\n")
        f.write(gen.round(4).to_string(index=False) + "\n")
    print("listo:", RES / "contrastes_v011.md")


if __name__ == "__main__":
    main()
