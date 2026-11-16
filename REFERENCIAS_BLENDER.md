# Referencias para modelar el Autódromo Hermanos Rodríguez en Blender

Recopiladas el 14 nov 2026. Solo se usan como **referencia visual**: no se copian fotos ni imágenes satelitales como texturas (tienen derechos). Los datos geométricos sí son libres: trazado TUM (LGPL-3.0) y entorno OpenStreetMap (ODbL), ya en `pistas/`.

## Datos del circuito (trazado actual, 2015-hoy)

- 4.304 km, 17 curvas, rediseño de Hermann Tilke en 2015. Sentido horario. Altitud ~2,240-2,285 m.
- Recta principal: ~1.29 km desde la salida de la curva 17 hasta la curva 1; la más larga del calendario de F1.
- Nombres de curvas:
  - 1, 2 y 3: Curva Moisés Solana (frenada fuerte, casi ángulo recto en la 1).
  - 4: Ese del Lago.
  - 6: Horquilla.
  - 7, 8 y 9: las Eses (izquierda-derecha-izquierda, rápidas; la sección con más fuerza G).
  - 13, 14 y 15 (16): sección del Foro Sol / Estadio GNP Seguros, lenta, rodeada de tribunas por tres lados.
  - 17: Peraltada / Curva Nigel Mansell, la última, rápida y **ligeramente peraltada**; desemboca en la recta.
- Pit building: 33 fosos, torre de control, centro de medios y Paddock Club en la recta principal (lado del Viaducto Río de la Piedad).
- Asfalto poco abrasivo; zonas de escape asfaltadas en curvas (remodelación 2015).

## Edificios clave para modelar

| Elemento | Qué es | Cómo modelarlo |
| --- | --- | --- |
| Estadio GNP Seguros (antes Foro Sol) | Estadio de 1993, remodelado en 2024 por Moyao Arquitectos; 65,000 personas; techo nuevo de 13,800 m² sobre las gradas | La "U" de tribunas alrededor de las curvas 13-16; gradas con techo ligero; es la pieza principal del visor |
| Tribunas de la recta y curvas 1-3 | Tribunas metálicas grandes | Extruir los polígonos `tribuna` de `entorno.json`, ~12 m de altura, con gradas escalonadas |
| Pit building | Edificio largo de la recta con fosos y torre de control | Volumen largo de 2-3 pisos frente a la recta, fachada de vidrio |
| Estadio Alfredo Harp Helú | Estadio de béisbol nuevo, dentro del área del autódromo (zona este) | Usar el polígono `estadio` de OSM; gradas en forma de abanico |
| Palacio de los Deportes ("Domo de cobre") | Arena olímpica de 1968 de Félix Candela, Antonio Peyrí y Enrique Castañeda; cúpula geodésica forrada de cobre sobre arcos de acero, contrafuertes de concreto y muros de ladrillo en zigzag | Cúpula facetada con material cobre (verde-café oxidado), al poniente del circuito; hito visual |
| Velódromo Olímpico Agustín Melgar | Velódromo cercano | Volumen ovalado bajo |
| Ciudad Deportiva Magdalena Mixhuca | Parque con canchas, árboles y lagos alrededor | Pasto, canchas (`cancha`), árboles low-poly desde los 1,781 puntos `arbol` |

## Paleta y ambiente

- Asfalto gris oscuro; pianos rojo y blanco (confirmar con fotos propias del recinto si se quieren colores exactos de cada curva).
- Pasto verde con zonas de escape grises.
- Tribunas: estructura gris metálica, asientos de color.
- Cielo de la Ciudad de México: luz de tarde cálida, algo de bruma; volcanes al fondo opcionales.
- Cúpula del Palacio de los Deportes: cobre envejecido.

## Enlaces de referencia (abrir, no copiar)

- Datos y curvas: https://oversteer48.com/autodromo-hermanos-rodriguez-circuit-layouts/
- Ficha del circuito: https://www.f1-fansite.com/f1%20circuits/autodromo-hermanos-rodriguez-layout-records/
- Vistas 360° desde las tribunas: https://3ddigitalvenue.com/3dmap/clients/f1/autodromo-hermanos-rodriguez/
- Remodelación 2015 (pit building, 33 fosos): https://www.mexicogp.mx/noticia/comienza-el-asfaltado-de-la-primera-capa-en-el-autodromo
- Estadio GNP Seguros remodelado (fotos interiores, techo): https://www.chilango.com/noticias/inauguran-el-estadio-gnp-antes-foro-sol-asi-luce-remodelado-fotos/
- Estadio GNP Seguros (Pollstar, techo y renders): https://news.pollstar.com/?p=1063150
- Palacio de los Deportes (estructura de la cúpula): https://www.redalyc.org/journal/7737/773782254013/773782254013.epub
- Palacio de los Deportes (ArchDaily): https://www.archdaily.com/es/tag/palacio-de-los-deportes
- Wikipedia del autódromo (historia, Peraltada, Foro Sol): https://www.Wikipedia.com/wiki/Autodromo_Hermanos_Rodriguez

## Segunda búsqueda: detalles finos

### Pista y bordes
- Escapatorias de **asfalto y pasto**. La curva 1 tiene escapatoria amplia por fuera.
- Pianos ("lavaderos"): ~3 km en total en 2015, homologados para F1, turismos y MotoGP. La curva 12 tiene marcador de frenado a 150 m.
- Asfalto de 2015 con materiales locales (obsidiana, arena, grava) en tres capas; tono gris oscuro con brillo fino. En 2026 se colocó una capa nueva de 6.5 cm y se renovaron kerbs, U-drains y la fachada del pit building.
- Barreras: **Tecpro** (bloques absorbentes, normalmente en colores de patrocinio) en zonas de impacto; mallas de seguridad y dovelas (muros de concreto) en el resto, incluido el Foro Sol.
- Orientación: 7 curvas a la izquierda, 10 a la derecha; la salida está a ~1,181 m de la curva 1; recta total ~1,314 m.

### Puntos oficiales (documento FIA 2025, para checkpoints y señalización)
- Sector 1: 136 m antes de la curva 4. Sector 2: 242 m antes de la curva 12. Trampa de velocidad: 234 m antes de la curva 1.
- DRS: detección 1 a 70 m después de la curva 9 y activación 1 a 80 m después de la 11; detección 2 a la salida de la 15; activación 2 a 240 m después de la 17; activación 3 a 115 m después de la 3.
- Pit lane: garajes en la recta principal; el documento trae el dibujo del pit lane con entrada, salida y líneas SC1/SC2. PDF: https://www.fia.com/system/files/decision-document/2025_mexico_city_grand_prix_-_event_notes_-_circuit_map_pit_lane_drawing_emergency_exits_map_and_quarantine_zone_area.pdf

### Tribunas (mapa oficial del GP)
| Tribuna | Ubicación | Zona |
| --- | --- | --- |
| Main 1, 1A, 2 | Recta principal, frente a los pits; semicubierta | Verde (puertas 6 y 7) |
| 2A | Recta principal | Naranja (puerta 8) |
| 3, 3A, 4, 5, 5A, 6, 6A, 6B | Frenada y curvas 1 a 3 (Moisés Solana) | Azul |
| 7, 8, 9, 10, 11 | Sección media: Ese del Lago, horquilla, eses; la 10 al final de la Ese del Lago | Amarilla y otras |
| 14 (Foro Sol Sur) | Dentro del Estadio GNP Seguros, parcialmente cubierta | Gris |
| 15 (Foro Sol Norte) | Dentro del estadio, sobre el **podio**, entre los edificios del Paddock Club cerca de la curva 13 | Café |
- El podio está en la sección del Foro Sol; los autos entran al estadio por la curva 12.
- Acceso al Foro Sol por la puerta 1, junto al Palacio de los Deportes (confirma la cercanía para el modelo).
- Pantallas gigantes junto al Paddock Club en el estadio.
- Mapas: https://www.mexico.gp/es/plano-de-las-tribunas-10 y https://oversteer48.com/grandstand-15-f1-mexico/

### Vegetación
- Ciudad Deportiva Magdalena Mixhuca: ~292 ha, con lagos artificiales y miles de árboles; hay árboles dentro de las curvas 3 y 4 del óvalo (zona interior).
- No encontré un inventario de especies del lugar. Como aproximación, las especies urbanas más comunes en la Ciudad de México son fresno (Fraxinus uhdei), casuarina, eucalipto, trueno (Ligustrum) y jacaranda. Para el modelo bastan 3 o 4 árboles low-poly de esos tipos.

## Orden sugerido en Blender

1. Pista desde `pistas/fuente_tum/MexicoCity.csv` (bordes con ancho real), pianos en el exterior de las curvas, pasto y escapatorias. Exportar `pista.json` (bordes + 40-60 checkpoints) para el simulador.
2. Peralte suave en la curva 17 (solo visual; el simulador es 2D).
3. Tribunas y Estadio GNP Seguros desde `pistas/mexico/entorno.json`.
4. Pit building, Palacio de los Deportes, Harp Helú, velódromo.
5. Árboles instanciados, edificios cercanos como volúmenes grises, cielo.
6. Exportar a glTF optimizado para Three.js (pocos polígonos, texturas pequeñas).

Créditos al publicar: TUM racetrack-database (LGPL-3.0) y © colaboradores de OpenStreetMap (ODbL).
