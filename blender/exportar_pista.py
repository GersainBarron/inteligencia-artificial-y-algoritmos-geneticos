"""Exporta desde Blender la geometría de la pista para el simulador (`pista.json`).

Se ejecuta DENTRO de Blender. Lee la colección `Pista_Sim` (creada por
`construir_pista.py`, editable a mano): curvas `LineaCentral`, `BordeIzquierdo`,
`BordeDerecho` y objetos `CP_###`. Valida la geometría y solo entonces escribe.

    import sys; sys.path.insert(0, r"<carpeta del proyecto>\\blender")
    import exportar_pista
    exportar_pista.exportar("pistas/mexico/pista.json")

Formato (esquema `evocarros.pista/1`), todo en metros, marco del archivo de TUM:
  linea_central, borde_izquierdo, borde_derecho   polilíneas cerradas [[x, y], ...]
                                                  en el sentido de circulación
  s_m                    abscisa curvilínea de cada punto de la línea central
  ancho_izq_m/ancho_der_m  distancia de cada punto central a cada borde
  checkpoints            [{indice, s_m, centro, izq, der}], en orden; el 0 es la salida/meta
  salida                 posición y rumbo (rad, antihorario desde +x) en s = 0
"""
from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))
import pista_geom as geo  # noqa: E402

COL_SIM = "Pista_Sim"
DECIMALES = 3  # milímetros


def _polilinea(col, nombre) -> np.ndarray:
    ob = col.objects[nombre]
    sp = ob.data.splines[0]
    co = np.empty(len(sp.points) * 4)
    sp.points.foreach_get("co", co)
    xyz = co.reshape(-1, 4)[:, :3]
    m = np.array(ob.matrix_world)
    return (xyz @ m[:3, :3].T + m[:3, 3])[:, :2].astype(float)


def _r(x):
    return np.round(np.asarray(x, dtype=float), DECIMALES).tolist()


def exportar(destino: str = "pistas/mexico/pista.json", forzar: bool = False) -> dict:
    bpy.context.view_layer.update()   # sin esto, los objetos recién creados aún no tienen su matriz de mundo
    col = bpy.data.collections[COL_SIM]
    centro = _polilinea(col, "LineaCentral")
    b_izq = _polilinea(col, "BordeIzquierdo")
    b_der = _polilinea(col, "BordeDerecho")
    s, largo = geo.abscisa(centro)

    cps = []
    for ob in sorted((o for o in col.objects if o.name.startswith("CP_")), key=lambda o: o.name):
        m = np.array(ob.matrix_world)
        v = np.array([list(vt.co) for vt in ob.data.vertices]) @ m[:3, :3].T + m[:3, 3]
        # se ajustan los extremos al borde (corrige el redondeo float32 de Blender;
        # si alguien movió el checkpoint, lo deja apoyado en los bordes)
        izq, der = geo.proyectar(v[0, :2], b_izq), geo.proyectar(v[1, :2], b_der)
        cps.append({"izq": izq, "der": der, "centro": m[:2, 3].astype(float)})  # el origen del objeto es su centro
    # abscisa de cada checkpoint: proyección de su centro sobre la línea central
    for cp in cps:
        ab = np.roll(centro, -1, axis=0) - centro
        u = np.clip(np.einsum("ij,ij->i", cp["centro"] - centro, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        d = np.hypot(*(centro + u[:, None] * ab - cp["centro"]).T)
        i = int(d.argmin())
        cp["s_m"] = float(s[i] + u[i] * np.hypot(*ab[i]))
        cp["centro"] = centro[i] + u[i] * ab[i]
    cps.sort(key=lambda c: c["s_m"])
    for k, cp in enumerate(cps):
        cp["indice"] = k

    informe = geo.validar(centro, b_izq, b_der, cps)
    if informe["fallas"] and not forzar:
        raise RuntimeError(f"No se exporta: la pista no pasa la validación: {informe['fallas']}")

    if len(b_izq) == len(centro) == len(b_der):  # vértices homólogos: ancho de TUM
        a_izq, a_der = np.hypot(*(b_izq - centro).T), np.hypot(*(b_der - centro).T)
    else:  # la pista se editó: distancia mínima a cada borde
        a_izq, a_der = geo.dist_a_polilinea(centro, b_izq), geo.dist_a_polilinea(centro, b_der)
    ancho = a_izq + a_der
    t0 = geo.tangentes(centro)[0]

    datos = {
        "esquema": geo.ESQUEMA,
        "nombre": col.get("nombre", ""),
        "fuente": "TUM racetrack-database (LGPL-3.0), commit e59595d1f3573b30d1ded6a08984935b957688e0",
        "fuente_csv": col.get("fuente_csv", ""),
        "fuente_sha256": col.get("fuente_sha256", ""),
        "generado": {
            "fecha": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "blender": bpy.app.version_string,
            "archivo_blend": Path(bpy.data.filepath).name,
            "script": "blender/exportar_pista.py",
        },
        "marco": "metros del archivo de TUM (x, y); el mismo de entorno.json",
        "unidades": "m",
        "cerrada": True,
        "sentido": informe["sentido"],
        "borde_exterior": informe["borde_exterior"],
        "longitud_m": round(largo, DECIMALES),
        "ancho_medio_m": round(float(ancho.mean()), DECIMALES),
        "ancho_min_m": round(float(ancho.min()), DECIMALES),
        "ancho_max_m": round(float(ancho.max()), DECIMALES),
        "n_puntos": len(centro),
        "n_checkpoints": len(cps),
        "paso_checkpoints_m": round(largo / len(cps), DECIMALES),
        "salida": {"s_m": 0.0, "posicion": _r(centro[0]), "rumbo_rad": round(math.atan2(t0[1], t0[0]), 6)},
        "linea_central": _r(centro),
        "s_m": _r(s),
        "ancho_izq_m": _r(a_izq),
        "ancho_der_m": _r(a_der),
        "borde_izquierdo": _r(b_izq),
        "borde_derecho": _r(b_der),
        "checkpoints": [
            {"indice": c["indice"], "s_m": round(c["s_m"], DECIMALES), "centro": _r(c["centro"]),
             "izq": _r(c["izq"]), "der": _r(c["der"])} for c in cps
        ],
    }

    # una clave por línea (y un checkpoint por línea) para que los cambios se lean en git
    lineas = []
    for clave, valor in datos.items():
        if clave == "checkpoints":
            cuerpo = ",\n".join("  " + json.dumps(c, ensure_ascii=False, separators=(", ", ": ")) for c in valor)
            lineas.append(f' "checkpoints": [\n{cuerpo}\n ]')
        else:
            lineas.append(f" {json.dumps(clave)}: {json.dumps(valor, ensure_ascii=False, separators=(', ', ': '))}")
    ruta = Path(destino)
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("{\n" + ",\n".join(lineas) + "\n}\n", encoding="utf-8")
    informe["archivo"] = str(ruta)
    informe["bytes"] = ruta.stat().st_size
    return informe


if __name__ == "__main__":
    print(exportar())
