"""Base de programas (SQLite) con MAP-Elites por islas (condición D).

Ejes de MAP-Elites (5 x 5 celdas por isla): velocidad media (0-130 km/h) y suavidad del
volante (cambio medio por paso, 0-0.3). Cada celda guarda el mejor programa que cae en ella.
4 islas; cada 50 programas el mejor de cada isla se copia a la siguiente (migración en anillo).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

N_ISLAS, N_BINS = 4, 5
V_MAX_KMH, SUAV_MAX = 130.0, 0.3


def celda(vel_kmh: float, suav: float) -> int:
    i = min(N_BINS - 1, max(0, int(vel_kmh / V_MAX_KMH * N_BINS)))
    j = min(N_BINS - 1, max(0, int(suav / SUAV_MAX * N_BINS)))
    return i * N_BINS + j


class Base:
    def __init__(self, ruta: Path):
        self.c = sqlite3.connect(str(ruta), check_same_thread=False)
        self.c.executescript("""
        CREATE TABLE IF NOT EXISTS programas (
          id INTEGER PRIMARY KEY, padre INTEGER, isla INTEGER, codigo TEXT, valido INTEGER,
          aptitud REAL, etapa INTEGER, vuelta INTEGER, t_vuelta REAL, motivo TEXT,
          velocidad_kmh REAL, suavidad REAL, celda INTEGER, retro TEXT, error TEXT,
          modelo TEXT, episodios INTEGER, segundos REAL, fecha TEXT);
        CREATE TABLE IF NOT EXISTS elites (isla INTEGER, celda INTEGER, id INTEGER, PRIMARY KEY (isla, celda));
        """)
        self.c.commit()

    def agregar(self, padre, isla, codigo, res: dict, modelo: str, segundos: float) -> int:
        ok = bool(res.get("ok"))
        vel, suav = res.get("velocidad_media_kmh", 0.0), res.get("suavidad_volante", 0.0)
        cur = self.c.execute(
            "INSERT INTO programas (padre, isla, codigo, valido, aptitud, etapa, vuelta, t_vuelta, motivo, velocidad_kmh,"
            " suavidad, celda, retro, error, modelo, episodios, segundos, fecha) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (padre, isla, codigo, int(ok), res.get("aptitud", -1.0) if ok else -1.0, res.get("etapa", 0),
             int(bool(res.get("vuelta_completa"))), res.get("t_vuelta"), res.get("motivo"), vel, suav,
             celda(vel, suav) if ok else None, res.get("texto"), res.get("error"), modelo,
             res.get("episodios", 0), segundos, datetime.now().isoformat(timespec="seconds")))
        pid = cur.lastrowid
        if ok:
            self._elite(isla, celda(vel, suav), pid, res["aptitud"])
        self.c.commit()
        return pid

    def _elite(self, isla, cel, pid, apt):
        fila = self.c.execute("SELECT e.id, p.aptitud FROM elites e JOIN programas p ON p.id = e.id"
                              " WHERE e.isla = ? AND e.celda = ?", (isla, cel)).fetchone()
        if fila is None or apt > fila[1]:
            self.c.execute("INSERT OR REPLACE INTO elites (isla, celda, id) VALUES (?,?,?)", (isla, cel, pid))

    def elites(self, isla=None) -> list[dict]:
        q = ("SELECT p.id, p.isla, p.codigo, p.aptitud, p.t_vuelta, p.motivo, p.retro, p.velocidad_kmh, p.suavidad, e.celda"
             " FROM elites e JOIN programas p ON p.id = e.id")
        filas = self.c.execute(q + (" WHERE e.isla = ?" if isla is not None else ""),
                               (() if isla is None else (isla,))).fetchall()
        k = ["id", "isla", "codigo", "aptitud", "t_vuelta", "motivo", "retro", "velocidad_kmh", "suavidad", "celda"]
        return [dict(zip(k, f)) for f in filas]

    def migrar(self):
        """El mejor de cada isla entra a la siguiente (anillo)."""
        mejores = []
        for i in range(N_ISLAS):
            el = self.elites(i)
            if el:
                mejores.append(max(el, key=lambda e: e["aptitud"]))
        for m in mejores:
            destino = (m["isla"] + 1) % N_ISLAS
            self._elite(destino, m["celda"], m["id"], m["aptitud"])
        self.c.commit()

    def mejor(self) -> dict | None:
        f = self.c.execute("SELECT id, codigo, aptitud, t_vuelta, motivo, velocidad_kmh FROM programas"
                           " WHERE valido = 1 ORDER BY aptitud DESC, id ASC LIMIT 1").fetchone()
        return None if f is None else dict(zip(["id", "codigo", "aptitud", "t_vuelta", "motivo", "velocidad_kmh"], f))

    # Fallos de infraestructura (el modelo no respondió) no son programas: no cuentan para el
    # presupuesto ni para el estancamiento (v0.12). Quedan en la tabla para el registro.
    SIN_FALLAS = "(error IS NULL OR error NOT LIKE 'error del modelo%')"

    def contar(self) -> int:
        return self.c.execute(f"SELECT COUNT(*) FROM programas WHERE {self.SIN_FALLAS}").fetchone()[0]

    def desde(self, pid: int) -> int:
        """Programas reales registrados después del programa pid."""
        return self.c.execute(f"SELECT COUNT(*) FROM programas WHERE id > ? AND {self.SIN_FALLAS}", (pid,)).fetchone()[0]

    def ultimos(self, n: int) -> list[dict]:
        filas = self.c.execute("SELECT valido, aptitud, vuelta, t_vuelta, episodios FROM programas ORDER BY id DESC LIMIT ?",
                               (n,)).fetchall()
        return [dict(zip(["valido", "aptitud", "vuelta", "t_vuelta", "episodios"], f)) for f in filas]

    def episodios(self) -> int:
        return self.c.execute("SELECT COALESCE(SUM(episodios), 0) FROM programas").fetchone()[0]

    def celdas_ocupadas(self) -> int:
        return self.c.execute("SELECT COUNT(DISTINCT celda) FROM elites").fetchone()[0]
