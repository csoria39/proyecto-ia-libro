# Repintado del 9-11-2026 (revisión completa)

Correcciones tras revisar los 19 diagramas enteros. Cada script lee la imagen anterior y escribe la corregida:
`python3 <script> entrada.png salida.png`. Usan `pintar.py` (texto con supermuestreo; fuentes Fira, Latin Modern,
DejaVu y FreeSans) y `tilde.py`.

Orden aplicado: primero `f_tildes.py` en 3.1, 7.2, 7.4 y 9.1; después el script de cada diagrama
(`f3_1`, `f7_4_l2` → `f7_4_p5` → `f7_4_t`, `f9_1`); los demás sobre la imagen del repositorio
(`f2_1`, `f4_1`, `f4_2`, `f7_3` con argumentos `fira600 13`, `f10_1`, `f10_2`, `f11_1`, `f13_1`, `f17_1`, `f20_1`,
`f20_2`, `fG_1`). `instalar.py` guarda las versiones anteriores en `originales/` (`_antes_2026-11-09b`) y genera
las imágenes del libro (giradas 90° salvo 3.1, 4.1, 4.2 y 7.4).

Las rutas de entrada y salida de algunos scripts apuntan a la carpeta de trabajo de la sesión; se cambian en la
línea de comandos.
