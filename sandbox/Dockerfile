# Contenedor aislado para evaluar código generado por el modelo de lenguaje (condición D).
# Se ejecuta siempre con: --network none, --read-only, límites de CPU/memoria/procesos y usuario sin privilegios.
FROM python:3.12-slim

RUN pip install --no-cache-dir "numpy>=2,<3" \
 && useradd --create-home --uid 1000 piloto

USER piloto
WORKDIR /work
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
