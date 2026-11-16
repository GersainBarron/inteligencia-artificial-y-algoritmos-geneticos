# Fuente de las pistas

Trazados reales tomados de la base de datos **racetrack-database** de la Universidad Técnica de Múnich (TUM, Institute of Automotive Technology).

- Repositorio: https://github.com/TUMFTM/racetrack-database
- Versión usada: commit `e59595d1f3573b30d1ded6a08984935b957688e0` (2021-09-18)
- Licencia: GNU LGPL v3 (ver `LICENSE` en esta carpeta). Los datos se usan sin modificar; las pistas derivadas para el simulador se generan con scripts propios a partir de estos archivos.

Formato de cada CSV: `x_m, y_m, w_tr_right_m, w_tr_left_m` (línea central en metros y ancho hacia cada lado).

| Archivo | Circuito | Uso | Largo | Ancho |
| --- | --- | --- | --- | --- |
| `Spa.csv` | Spa-Francorchamps, Bélgica | Prueba | 7.00 km | 8-16 m |
| `Monza.csv` | Monza, Italia | Prueba | 5.79 km | 8-12 m |
| `Budapest.csv` | Hungaroring, Hungría | Prueba | 4.38 km | 8-16 m |
| `MexicoCity.csv` | Autódromo Hermanos Rodríguez, México | Entrenamiento | 4.30 km | 10-18 m |
