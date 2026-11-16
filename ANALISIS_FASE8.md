# Entrega para la fase 8: análisis de las 60 corridas de red (15 nov 2026, 20:10)

Este documento es el punto de partida de la sesión de análisis. Leer también `ESTADO.md`
(historia completa y decisiones) y el protocolo vigente **v0.10** (documento "Protocolo v0.10:
neuroevolución de carros con sensores evolutivos" en el proyecto de Claude "Redes neuronales
basicas", https://claude.ai/artifact/PcjEwN3yZa2bj6NtusEiD4).

Convención obligatoria: redactar como investigación de nivel doctorado; cada decisión justificada;
suposiciones marcadas; resultados negativos reportados; todo cambio al protocolo sube de versión
con fecha y razón en el "Historial de versiones".

## Qué hay listo

- **60 de 60 corridas de red terminadas**, cada una con evaluación final (6 condiciones × 10 semillas:
  A-GA, B-GA, C-GA, A-ES, B-ES, C-ES; semillas 1-10).
- Tabla consolidada: `resultados/resumen_redes.csv` (carpeta del proyecto), generada con
  `python scripts/consolidar_resultados.py` (solo lectura). Una fila por corrida.
- Corridas completas en `D:\redes neuronales\runs\<CONDICION>_sNN\`:
  - `config.json` (parámetros, protocolo, motor), `estado.json` (estado final, fin = convergencia o presupuesto)
  - `fitness.csv`: una fila por generación (`generacion, episodios, aptitud_max, aptitud_media,
    aptitud_mediana, pct_vuelta, mejor_t_vuelta, diversidad, t_limite, segundos`)
  - `champions/gen_XXXX.json`: genoma del campeón cada 10 generaciones (+ final)
  - `checkpoint.pkl`: población final exacta (`pob[0]` = campeón con precisión completa)
  - `replays/gen_XXXX.json.gz`: 20 mejores carros por generación guardada (formato `evocarros.repeticion/1`)
  - `eval_final.json` / `eval_final.csv`: campeón en México, Spa, Monza y Hungaroring; salida
    estándar + 20 salidas perturbadas (±1.5 m, ±3°, semilla 12345); T_max = 2 × vuelta ideal de cada pista.

Columnas de `resumen_redes.csv`: corrida, condicion, cuerpo (A/B/C), algoritmo (GA/ES), semilla,
protocolo, motor, estado, fin, generaciones, episodios, mejor_aptitud, pct_vuelta_final,
mejor_t_vuelta, episodios_primera_vuelta, episodios_95pct_vuelta, v_max_campeon,
angulos_grados, alcances_m, y por pista (mexico, spa, monza, hungaroring): _completa, _t_vuelta,
_fraccion, _pct_perturbadas, _fraccion_media_pert.

Conteo descriptivo (sin analizar todavía): corridas que completan la vuelta en México —
A-GA 7/10, B-GA 8/10, C-GA 10/10, A-ES 5/10, B-ES 4/10, C-ES 5/10.

## Qué pide el protocolo (fase 8)

- H1: B (rayos evolutivos) vs A (rayos fijos), con el mismo presupuesto de episodios. Control
  adicional previsto: configuraciones de rayos fijas al azar con su red optimizada (no corrido aún).
- H2: los rayos evolucionados convergen a configuraciones no uniformes (más alcance al frente),
  parecidas entre semillas → diagrama polar de ángulos y alcances de los campeones B y C.
- H3: velocidad como gen (C) → vueltas más rápidas a costa de más choques tempranos.
- H4 (exploratoria): ES de OpenAI vs GA (μ+λ): misma aptitud en menos episodios.
- Prueba: Mann-Whitney U, α = 0.05, 10 semillas por condición; reportar tamaños de efecto
  (p. ej., delta de Cliff o correlación rango-biserial) y medianas con rango intercuartil.
- Resultados a reportar: curvas de aptitud (mediana y rango intercuartil por condición, contra
  episodios), diagrama polar de rayos, tabla de generalización, robustez (salidas perturbadas).
- **Ablación (pendiente de implementar):** en el campeón de cada corrida B y C, apagar un rayo a la
  vez (lectura fija en 1) y medir la caída de aptitud en 20 episodios.

## Decisiones que hay que tomar y registrar ANTES de mirar los contrastes

1. Métrica principal de H1: el protocolo dice "aptitud final". Varias corridas no completan la
   vuelta; la aptitud las incluye a todas (avance + bono). Las comparaciones de tiempo de vuelta
   solo aplican a las que completan (reportar n). La tasa de vuelta completa se puede comparar con
   prueba exacta de Fisher.
2. Los tiempos de A y B tienen techo distinto que C: A y B están limitados a 35 m/s (vuelta ideal
   por la línea central 143.8 s); C evoluciona su velocidad (15-50 m/s; ideal con 50 m/s: 125.4 s).
3. Corrección por comparaciones múltiples (son varias pruebas): decidir y registrar (p. ej., Holm).

## Advertencias para la interpretación

- **v0.10:** ES arranca del mejor de los 50 genomas iniciales del GA de la misma semilla. Las 9 ES
  hechas antes (s01-s03, regla anterior) están archivadas en `runs\_archivo_ES_v09` y **no** entran.
- `A-GA_s01_piloto_v05` (reglas v0.5) es piloto: **no** entra (el consolidado ya lo excluye).
- A-GA_s01 corrió 608 generaciones con el motor de referencia y el resto con Numba (equivalentes:
  misma vuelta al 1e-12 m).
- B-ES_s05 se interrumpió por un error de escritura de Windows en la generación 512 y se reanudó
  desde su checkpoint (estado del generador aleatorio incluido).
- Las evaluaciones finales usan el genoma exacto del checkpoint; antes el JSON se redondeaba a 8
  decimales y eso cambiaba carros frágiles (corregido y reevaluado).
- Hallazgos ya observados (resultados, no fallas): mesetas en la curva 1 (A-GA_s04 nunca salió);
  carros C "miedosos" que bajan su velocidad máxima a ~18.5 m/s y convergen con vueltas de ~231-234 s;
  todos los campeones chocan en el mismo punto de Monza (~16% de la vuelta); la semilla 4 resultó
  difícil para ambos algoritmos (misma población inicial desde v0.10).
- El protocolo sigue en v0.10; cualquier criterio de análisis nuevo se registra como v0.11.

## Lo que sigue corriendo (no detener)

- **Condición D (mini-AlphaEvolve)** en la GPU: `scripts/campana_d.py`, semillas 1-5, bitácora
  `D:\redes neuronales\runs\campana_d.log`. D-LLM_s01 pasó a modo "grande" (qwen3.5 9B) tras 500
  programas sin mejora; ningún programa ha pasado la curva 1 (~1,900 programas). Su análisis (H5)
  va cuando termine.
- Guardián (`scripts/guardian.py`) y panel (http://localhost:8080) en segundo plano.
- Extensión Gran Premio de campeones (fase 10, H6) pendiente, después del análisis.

## Entorno

- AORUS, conda `evocarros` (Python 3.12, numpy, scipy, pandas, matplotlib, numba). Usar el python
  de `C:\Users\gersa\anaconda3\envs\evocarros\python.exe`.
- Carpeta del proyecto: `C:\Users\gersa\OneDrive\Documentos\redes neuronales basicas`.
- Sugerencia: scripts de análisis en `scripts/analisis/`, figuras en `resultados/figuras/`,
  y un documento de resultados nuevo en el proyecto de Claude.
