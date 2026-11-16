# Inteligencia artificial y algoritmos genéticos

**Carros que aprenden a manejar solos por evolución, sin que nadie les enseñe.**

🏁 **Ver el simulador en el navegador:** https://gersainbarron.github.io/inteligencia-artificial-y-algoritmos-geneticos/

Autor: Gersain Barron Velazquez · [gersadesign.com](https://gersadesign.com)

---

## ¿Qué es?

Un experimento de neuroevolución planteado como investigación doctoral. Un carro visto desde arriba recorre réplicas de circuitos reales de Fórmula 1. Su "piloto" empieza sin saber nada. Generación tras generación sobreviven y se reproducen los que llegan más lejos, hasta que aprenden a dar la vuelta completa.

La pregunta central: **¿qué conviene dejar que la evolución decida?** Solo el "cerebro" del piloto, o también sus "sentidos" (hacia dónde y qué tan lejos mira) y su temperamento (qué tan rápido se atreve a ir). Como contraste se agrega un piloto que no es una red neuronal sino **código escrito y mejorado por un modelo de lenguaje local**, al estilo AlphaEvolve.

## Qué se simula

### El carro
- **Física:** modelo cinemático de bicicleta con paso de 1/30 s, distancia entre ejes de 2.6 m y volante máximo de 30°.
- **Desempeño:** velocidad tope de 50 m/s (180 km/h), aceleración de 6 m/s² y frenado de 12 m/s².
- **Agarre:** el agarre lateral se limita a 1.5 g; si se rebasa, el carro subvira (se abre en la curva) en lugar de derrapar.
- **Carrocería:** un rectángulo de 4.0 × 1.8 m. Si toca el borde de la pista, queda fuera.
- **Sentidos:** 5 rayos que miden la distancia al borde de la pista, más su propia velocidad y el ángulo hacia el siguiente punto de control.

### El piloto (red neuronal 7:8:2)
- **Entradas (7):** los 5 rayos, la velocidad y el ángulo hacia el punto de control.
- **Capa oculta:** 8 neuronas.
- **Salidas (2):** volante y acelerador/freno.

El genoma tiene 93 números:
- 82 pesos de la red.
- 5 ángulos y 5 alcances de los rayos.
- La velocidad máxima que el piloto se permite.

### Las condiciones del experimento
| Condición | Qué evoluciona |
|---|---|
| **A** | Solo la red; rayos fijos en un abanico de 5 |
| **B** | La red **y** los rayos (ángulo y alcance) |
| **C** | Lo de B **más** la velocidad máxima (15–50 m/s) |
| **R** | Control: rayos fijos puestos al azar |
| **D** | Mini-AlphaEvolve: un LLM local escribe y muta el código del piloto en Python |

A, B, C y R se entrenan con dos algoritmos:
- **Algoritmo genético (μ+λ):** 50 padres y 50 hijos por generación, con mutación gaussiana.
- **Estrategias evolutivas de OpenAI:** 50 pares antitéticos, rangos centrados y Adam.

Ambos tienen el mismo presupuesto: 100,000 episodios por corrida, con 10 a 20 semillas por condición.

### Cómo se califica (aptitud)
- **Avance:** puntos de control cruzados, más la fracción del tramo actual.
- **Premio por rapidez:** se suma solo si el carro termina la vuelta.
- **Muerte:** el carro muere si choca, se estanca (no avanza 10 m en 3 s) o va en sentido contrario. Al morir conserva lo que avanzó.
- **Fantasmas:** los carros no chocan entre sí; cada uno corre como si estuviera solo.

### Pistas
- **Entrenamiento:** Autódromo Hermanos Rodríguez (Ciudad de México).
- **Prueba:** Spa-Francorchamps, Monza y Hungaroring. Son pistas que el piloto nunca vio; miden si aprendió a manejar o solo memorizó una pista.

### Condición D (el piloto escrito por un LLM)
- **Modelos:** qwen2.5-coder:7b y qwen3.5:9b en Ollama, en local. No se usa ninguna API externa.
- **Cómo cambia el código:** el modelo propone cambios en formato SEARCH/REPLACE.
- **Dónde corre:** cada programa se ejecuta en un contenedor Docker sin red, de solo lectura, con 1 CPU y 512 MB.
- **Evaluación en cascada:** primero 45 s de prueba (debe recorrer 1,200 m); si pasa, se evalúa la vuelta completa.
- **Diversidad:** MAP-Elites con 4 islas de 5 × 5 celdas (velocidad media × suavidad del volante).

## Resultados principales (redes)
- **H1:** los rayos evolutivos (B) **no** superaron a un abanico fijo bien elegido (A).
- **Contra el control:** B sí superó con claridad a rayos fijos puestos al azar (R). Evolucionar los sentidos ayuda, pero el diseño fijo que elegimos ya era bueno.
- **H2:** los rayos evolucionados **alargan el alcance hacia el frente**, en 18 de 20 semillas.
- **Algoritmos:** el algoritmo genético superó a las estrategias de OpenAI en este problema.

El análisis completo está en [`ANALISIS_FASE8.md`](ANALISIS_FASE8.md) y [`resultados/`](resultados/).

## El visor 3D

Hecho con Three.js. Reproduce carreras reales grabadas durante la evolución: los 20 mejores carros de cada generación grabada, corriendo como fantasmas.

| Control | Qué hace |
|---|---|
| **Corrida** | Elige condición y algoritmo, por ejemplo `B-GA_s01` |
| **Generación** | Salta a otra etapa del aprendizaje: en la 0 chocan todos; en las últimas dan la vuelta completa |
| **▶ / ⏸** | Reproducir o pausar |
| **⏮** | Volver al arranque de la carrera |
| **Velocidad** | Cámara lenta o rápida |
| **Cámara** | *Libre* (giras con el ratón), *Seguir* (persigue al carro elegido) o *A bordo* (desde el carro, con los rayos visibles) |
| **Fantasma** | Superpone otra generación para compararlas en la misma pista |
| **Entorno / Puntos de control / Muertos** | Muestra u oculta las gradas y el paisaje, las líneas rojas de control y los carros que ya quedaron fuera |

**Clic en un carro** para abrir su ficha:
- Su velocidad, avance, aptitud y genes.
- Sus rayos dibujados en la pista; una esfera roja marca dónde cada rayo toca el borde.
- La pestaña **Red**: la red neuronal con cada neurona iluminándose según su activación en ese instante.
- La pestaña **Código**: la misma red escrita como código, con los valores vivos.

La versión web (carpeta `docs/`) trae 7 corridas con 6 generaciones cada una, en la pista de entrenamiento.

## Cómo montarlo

### Solo ver el visor en tu computadora
```bash
cd docs
python -m http.server 8000
# abre http://localhost:8000
```
Hace falta un servidor local porque el navegador no carga los modelos 3D desde `file://`. Sirve cualquier navegador moderno: Chrome, Edge, Firefox o Safari.

### Correr el experimento
Requisitos: Python 3.12 con numpy, numba, scipy, pandas, matplotlib y requests.
```bash
conda create -n evocarros python=3.12
conda activate evocarros
pip install numpy numba scipy pandas matplotlib requests

python -m evocarros.servidor.servidor           # panel y visor en http://localhost:8080
python scripts/campana.py --semillas 1-10       # condiciones A, B y C con GA y ES
python scripts/manejar_teclado.py               # manejar tú mismo con el teclado
```
- **Dónde se guarda:** las corridas se escriben en `runs/`, o en la carpeta que indique `EVOCARROS_RUNS`. Cada corrida guarda su avance y se puede interrumpir y reanudar.
- **Condición D:** necesita además [Ollama](https://ollama.com) con `qwen2.5-coder:7b` y `qwen3.5:9b`, y Docker. Se lanza con `python scripts/campana_d.py --semillas 1-5`.
- **Pistas:** se regeneran con Blender a partir de los scripts en `blender/`.

### Publicar la versión web en GitHub Pages
1. Sube este repositorio a GitHub con el nombre `inteligencia-artificial-y-algoritmos-geneticos`.
2. En *Settings → Pages*, elige la rama `main` y la carpeta `/docs`.
3. Cambia `USUARIO` en el enlace de arriba por tu usuario de GitHub.

## Estructura
| Carpeta | Contenido |
|---|---|
| `evocarros/sim` | Física, rayos, choques, aptitud y repeticiones (con versión acelerada en Numba) |
| `evocarros/neuro` | Red neuronal, algoritmo genético y estrategias evolutivas |
| `evocarros/llm_evo` | Mini-AlphaEvolve (condición D) |
| `evocarros/servidor` | Panel local |
| `sandbox` | Evaluador aislado que corre dentro de Docker |
| `visor` | Visor original, conectado al panel |
| `docs` | Visor web estático para GitHub Pages |
| `pistas` | Las cuatro pistas en JSON |
| `blender` | Scripts para construir pistas y carro |
| `scripts` | Campañas, evaluación, análisis y utilidades |
| `resultados` | Tablas y contrastes estadísticos |

## Licencia
El código de este proyecto se publica con licencia MIT (ver [LICENSE](LICENSE)). Los datos y bibliotecas de terceros conservan sus propias licencias:

## Créditos y licencias de terceros
- **Trazados de pista:** [TUM racetrack-database](https://github.com/TUMFTM/racetrack-database), con licencia LGPL-3.0. Ver `pistas/fuente_tum/`.
- **Entorno de las pistas:** datos de © colaboradores de OpenStreetMap, con licencia ODbL.
- **Three.js:** licencia MIT, incluido en `visor/vendor/three/`.
