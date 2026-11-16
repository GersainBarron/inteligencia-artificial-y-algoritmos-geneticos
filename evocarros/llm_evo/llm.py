"""Mutación de programas con un modelo de lenguaje local (Ollama) para la condición D.

El modelo recibe el programa padre, sus métricas y retroalimentación en texto, más 2 programas
de inspiración de otras celdas, y responde con bloques SEARCH/REPLACE (como AlphaEvolve). Si
no los usa pero devuelve una función completa en un bloque de código, se acepta como reescritura.
Solo puede cambiar lo que está entre las marcas EVOLVE-BLOCK; el resto (el simulador) ni
siquiera está en el programa.
"""
from __future__ import annotations

import ast
import json
import re
import urllib.request

OLLAMA = "http://localhost:11434/api/chat"
INICIO, FIN = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"

PROGRAMA_INICIAL = f'''"""Piloto de la condición D: el modelo de lenguaje reescribe la función manejar()."""
import math

{INICIO}
def manejar(rayos, velocidad, angulo_cp):
    # rayos[0..4]: distancia libre a -60, -30, 0, +30, +60 grados (positivo = izquierda), en [0, 1]
    izquierda = rayos[3] + rayos[4]
    derecha = rayos[0] + rayos[1]
    volante = 1.5 * (izquierda - derecha) + 0.5 * angulo_cp
    frente = rayos[2]
    pedal = 1.0 if frente > 0.5 else -1.0 + 2.0 * frente
    return volante, pedal
{FIN}
'''

SISTEMA = """Eres un ingeniero que mejora el código de un piloto de carreras simulado.
El carro (vista desde arriba, física de bicicleta, 35 m/s de velocidad máxima) corre el Autódromo
Hermanos Rodríguez (4.3 km, 17 curvas, ancho 10-18 m). Muere si toca un borde, si en 3 s no avanza
10 m o si va en sentido contrario. La aptitud es el avance en checkpoints (50 por vuelta) más un
bono si termina la vuelta rápido.

La función manejar(rayos, velocidad, angulo_cp) se llama 30 veces por segundo:
- rayos: 5 distancias libres normalizadas en [0, 1] hacia -60, -30, 0, +30, +60 grados
  (positivo = izquierda; 1 = no hay borde dentro de 49 m).
- velocidad: v / 50 m/s, en [0, 1] (el carro no pasa de 0.7).
- angulo_cp: ángulo hacia el centro del siguiente checkpoint / pi, en [-1, 1] (+ = izquierda).
Devuelve (volante, pedal) en [-1, 1]: volante +1 gira a la izquierda; pedal > 0 acelera
(6 m/s^2), < 0 frena (12 m/s^2). La curvatura se limita a 1.5 g de aceleración lateral.

Reglas: solo Python estándar y el módulo math; nada de archivos, red ni import adicionales.
Puedes usar variables globales para recordar estado entre llamadas. El código debe ser claro.

Responde con uno o más bloques de cambio con este formato exacto:
<<<<<<< SEARCH
(líneas exactas que existen en el programa actual)
=======
(líneas nuevas)
>>>>>>> REPLACE
Antes de los bloques, explica en una sola frase la idea del cambio. Sé breve: no repitas el
programa completo, solo los bloques de cambio."""


def prompt(padre: dict, inspiraciones: list[dict]) -> str:
    partes = ["## Programa actual (a mejorar)", "```python", padre["codigo"], "```",
              f"Aptitud: {padre['aptitud']:.2f}. " + (padre.get("retro") or "")]
    for k, ins in enumerate(inspiraciones, 1):
        partes += [f"## Inspiración {k} (aptitud {ins['aptitud']:.2f}, {ins.get('velocidad_kmh') or 0:.0f} km/h de media)",
                   "```python", extraer_bloque(ins["codigo"]), "```"]
    partes.append("Propón un cambio que aumente la aptitud (más avance sin chocar y, si ya completa "
                  "la vuelta, menor tiempo de vuelta).")
    return "\n".join(partes)


def pedir(modelo: str, texto: str, semilla: int, temperatura: float = 0.8, tiempo: int = 300) -> str:
    cuerpo = {"model": modelo, "stream": False, "think": False,
              "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": texto}],
              "options": {"temperature": temperatura, "seed": semilla, "num_ctx": 8192, "num_predict": 900}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(cuerpo).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=tiempo) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"]


def extraer_bloque(codigo: str) -> str:
    i, j = codigo.find(INICIO), codigo.find(FIN)
    return codigo[i + len(INICIO):j].strip("\n") if 0 <= i < j else codigo


_DIFF = re.compile(r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n?>{5,}\s*REPLACE", re.S)
_CODIGO = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def _buscar_flexible(texto: str, buscar: str) -> tuple[int, int] | None:
    """Ubica `buscar` en `texto` comparando línea por línea sin espacios a los lados
    (los modelos chicos suelen cambiar la sangría o dejar espacios al final).
    Devuelve (inicio, fin) en caracteres del texto original, o None si no hay
    exactamente una coincidencia."""
    lt = texto.splitlines(keepends=True)
    lb = [x.strip() for x in buscar.strip("\n").splitlines()]
    while lb and not lb[0]:
        lb.pop(0)
    while lb and not lb[-1]:
        lb.pop()
    if not lb:
        return None
    hallados = [i for i in range(len(lt) - len(lb) + 1)
                if all(lt[i + k].strip() == lb[k] for k in range(len(lb)))]
    if len(hallados) != 1:
        return None
    i = hallados[0]
    ini = sum(len(x) for x in lt[:i])
    return ini, ini + sum(len(x) for x in lt[i:i + len(lb)])


def _sangria(texto: str, ref: str) -> str:
    """Reindenta el reemplazo para que su primera línea tenga la sangría del texto que sustituye."""
    r = ref.splitlines()[0] if ref.splitlines() else ""
    base = len(r) - len(r.lstrip())
    lineas = texto.strip("\n").splitlines()
    if not lineas:
        return ""
    actual = len(lineas[0]) - len(lineas[0].lstrip())
    delta = base - actual
    if delta == 0:
        return "\n".join(lineas) + "\n"
    out = []
    for x in lineas:
        if delta > 0:
            out.append(" " * delta + x if x.strip() else x)
        else:
            quitar = min(-delta, len(x) - len(x.lstrip()))
            out.append(x[quitar:])
    return "\n".join(out) + "\n"


def aplicar(padre: str, respuesta: str) -> tuple[str | None, str]:
    """Devuelve (programa_nuevo, descripción) o (None, motivo del rechazo)."""
    i, j = padre.find(INICIO), padre.find(FIN)
    cabeza, bloque, cola = padre[:i + len(INICIO)], padre[i + len(INICIO):j], padre[j:]
    cambios = _DIFF.findall(respuesta)
    if cambios:
        nuevo = bloque
        for buscar, reemplazo in cambios:
            if buscar in nuevo:
                nuevo = nuevo.replace(buscar, reemplazo, 1)
                continue
            pos = _buscar_flexible(nuevo, buscar)
            if pos is None:
                return None, "un bloque SEARCH no coincide con el programa"
            ini, fin = pos
            nuevo = nuevo[:ini] + _sangria(reemplazo, nuevo[ini:fin]) + nuevo[fin:]
        bloque = nuevo
        tipo = f"{len(cambios)} cambio(s)"
    else:
        cods = [c for c in _CODIGO.findall(respuesta) if "def manejar" in c]
        if not cods:
            return None, "la respuesta no trae cambios aplicables"
        c = cods[-1]
        c = extraer_bloque(c) if INICIO in c else c
        c = "\n".join(l for l in c.splitlines() if not l.strip().startswith(("import ", "from ")))
        bloque = "\n" + c.strip("\n") + "\n"
        tipo = "reescritura completa"
    if INICIO in bloque or FIN in bloque:
        return None, "el cambio toca las marcas del bloque evolucionable"
    nuevo = cabeza + bloque + cola
    try:
        arbol = ast.parse(nuevo)
    except SyntaxError as e:
        return None, f"error de sintaxis: {e.msg}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "manejar" for n in arbol.body):
        return None, "el programa ya no define manejar()"
    for n in ast.walk(arbol):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            nombres = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            if any(x != "math" for x in nombres):
                return None, "importa módulos no permitidos"
    return nuevo, tipo
