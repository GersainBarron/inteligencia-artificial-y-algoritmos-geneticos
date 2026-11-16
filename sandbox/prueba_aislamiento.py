"""Prueba que el contenedor está aislado: sin red, sin escritura y con numpy disponible.

Imprime una línea JSON con los resultados. Se ejecuta dentro del contenedor.
"""
import json
import os
import socket

resultado = {}

# 1. Red: debe fallar.
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    resultado["red_bloqueada"] = False
except OSError:
    resultado["red_bloqueada"] = True

# 2. Escritura en el sistema de archivos: debe fallar (contenedor de solo lectura).
try:
    with open("/work/intento.txt", "w") as f:
        f.write("x")
    resultado["escritura_bloqueada"] = False
except OSError:
    resultado["escritura_bloqueada"] = True

# 3. Usuario sin privilegios.
resultado["usuario_no_root"] = os.getuid() != 0

# 4. NumPy disponible para el simulador.
try:
    import numpy as np

    resultado["numpy"] = np.__version__
except ImportError:
    resultado["numpy"] = None

print(json.dumps(resultado))
