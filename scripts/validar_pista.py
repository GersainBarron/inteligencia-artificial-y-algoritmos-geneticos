"""Validación independiente de una pista exportada desde Blender.

Comprueba que `pista.json` (escrito por `blender/exportar_pista.py`):
1. reproduce el trazado de TUM del que dice venir (misma huella SHA-256 y
   desviación máxima de bordes, línea central y checkpoints por debajo de 2 mm,
   que es el margen del redondeo a milímetros más el float32 de Blender);
2. cumple las condiciones geométricas que necesita el simulador (bordes sin
   cruces, línea central dentro de la pista, checkpoints en orden, apoyados en
   los bordes y sin atravesarlos);
y dibuja una figura de control con los checkpoints numerados.

Uso:
    conda activate evocarros
    python scripts/validar_pista.py                       # pistas/mexico/pista.json
    python scripts/validar_pista.py pistas/spa/pista.json

Termina con código 0 si todo pasa y 1 si hay alguna falla.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "blender"))
import pista_geom as geo  # noqa: E402

TOL_M = 2e-3


def validar_archivo(ruta: Path) -> tuple[dict, dict]:
    p = json.loads(ruta.read_text(encoding="utf-8"))
    centro, b_izq, b_der = (np.array(p[k]) for k in ("linea_central", "borde_izquierdo", "borde_derecho"))
    cps = [{**c, "centro": np.array(c["centro"]), "izq": np.array(c["izq"]), "der": np.array(c["der"])} for c in p["checkpoints"]]
    r = geo.validar(centro, b_izq, b_der, cps)

    def exigir(cond, texto):
        if not cond:
            r["fallas"].append(texto)

    exigir(p["esquema"] == geo.ESQUEMA, f"esquema inesperado: {p['esquema']}")
    exigir([c["indice"] for c in p["checkpoints"]] == list(range(len(cps))), "índices de checkpoints no consecutivos")
    exigir(len(cps) == p["n_checkpoints"], "n_checkpoints no coincide")
    exigir(40 <= len(cps) <= 60, "el protocolo pide de 40 a 60 checkpoints")
    exigir(abs(r["longitud_m"] - p["longitud_m"]) < TOL_M * len(centro), "longitud_m no coincide con la geometría")
    exigir(len(p["s_m"]) == len(centro) == len(p["ancho_izq_m"]) == len(p["ancho_der_m"]), "arreglos por punto de distinto tamaño")

    # contraste con la fuente de TUM
    csv = RAIZ / p["fuente_csv"]
    r["sha256_coincide"] = geo.sha256(csv) == p["fuente_sha256"]
    exigir(r["sha256_coincide"], "el CSV de TUM cambió respecto al que se usó para exportar")
    c0, w_der, w_izq = geo.leer_tum(csv)
    i0, d0 = geo.bordes(c0, w_der, w_izq)
    if len(c0) == len(centro):
        r["desv_max_centro_m"] = float(np.abs(c0 - centro).max())
        r["desv_max_bordes_m"] = float(max(np.abs(i0 - b_izq).max(), np.abs(d0 - b_der).max()))
        r["desv_max_anchos_m"] = float(max(np.abs(w_izq - p["ancho_izq_m"]).max(), np.abs(w_der - p["ancho_der_m"]).max()))
        ref = geo.checkpoints(c0, i0, d0, len(cps))
        r["desv_max_checkpoints_m"] = float(max(
            max(np.abs(a[k] - b[k]).max() for k in ("centro", "izq", "der")) for a, b in zip(ref, cps)))
        r["desv_max_s_checkpoints_m"] = float(max(abs(a["s_m"] - b["s_m"]) for a, b in zip(ref, cps)))
        for clave in ("desv_max_centro_m", "desv_max_bordes_m", "desv_max_anchos_m", "desv_max_checkpoints_m", "desv_max_s_checkpoints_m"):
            exigir(r[clave] < TOL_M, f"{clave} = {r[clave]:.4f} m supera {TOL_M} m")
    else:
        r["nota"] = "la pista se editó en Blender (distinto número de puntos que TUM); no se compara punto a punto"
    ancho = np.array(p["ancho_izq_m"]) + np.array(p["ancho_der_m"])
    r["ancho_medio_m"], r["ancho_min_m"], r["ancho_max_m"] = float(ancho.mean()), float(ancho.min()), float(ancho.max())
    return p, r


def figura(p: dict, destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    izq, der = np.array(p["borde_izquierdo"]), np.array(p["borde_derecho"])
    fig, ax = plt.subplots(figsize=(15, 11), dpi=130)
    poli = np.vstack([izq, izq[:1], der[:1], der[::-1]])
    ax.fill(poli[:, 0], poli[:, 1], color="#3a3b40", lw=0)
    for b in (izq, der):
        ax.plot(*np.vstack([b, b[:1]]).T, color="black", lw=0.6)
    for c in p["checkpoints"]:
        (x0, y0), (x1, y1) = c["izq"], c["der"]
        ax.plot([x0, x1], [y0, y1], color="#d62728" if c["indice"] else "#2ca02c", lw=1.6)
        n = np.array([x0 - x1, y0 - y1]); n = n / np.hypot(*n)
        ax.annotate(str(c["indice"]), np.array([x0, y0]) + 16 * n, ha="center", va="center", fontsize=7)
    x, y = p["salida"]["posicion"]; a = p["salida"]["rumbo_rad"]
    ax.annotate("", (x + 90 * np.cos(a), y + 90 * np.sin(a) - 30), (x, y - 30), arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2))
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("x (m, marco de TUM)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{p['nombre']}: bordes y {p['n_checkpoints']} checkpoints (cada {p['paso_checkpoints_m']:.1f} m)\n"
                 f"longitud {p['longitud_m']:.1f} m · ancho {p['ancho_min_m']:.1f}-{p['ancho_max_m']:.1f} m (medio {p['ancho_medio_m']:.2f} m) · "
                 f"sentido {p['sentido']} · verde = salida/meta (checkpoint 0)")
    fig.tight_layout(); fig.savefig(destino); plt.close(fig)


def main() -> int:
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "pistas" / "mexico" / "pista.json"
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    p, r = validar_archivo(ruta)
    for clave, valor in r.items():
        if clave != "fallas":
            print(f"  {clave:32s} {valor:.6g}" if isinstance(valor, float) else f"  {clave:32s} {valor}")
    try:
        figura(p, ruta.with_name("pista_control.png"))
        print(f"  figura: {ruta.with_name('pista_control.png')}")
    except ImportError:
        print("  (sin matplotlib: no se dibujó la figura de control)")
    if r["fallas"]:
        print("FALLAS:"); [print("  -", f) for f in r["fallas"]]
        return 1
    print("OK: la pista pasa todas las comprobaciones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
