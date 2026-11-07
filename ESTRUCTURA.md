# Estructura del repositorio

Mapa de la ubicación de cada archivo, rama `revision-v18`, 7 de noviembre de 2026.
La lista completa de programas y ejercicios, con el capítulo de cada uno, está en el [README](README.md).

```
proyecto-ia-libro/
├── README.md                     · presentación, instalación y tabla de archivos
├── ESTRUCTURA.md                 · este mapa
├── ERRATAS.md
├── LICENSE                       · licencia MIT (código)
├── LICENSE-LIBRO.md              · licencia CC BY-NC-ND 4.0 (libro)
├── requirements.txt              · versiones mínimas
├── requirements-probado.txt      · versiones exactas con las que se obtuvieron las salidas
├── instalar_windows.bat · instalar_mac.sh · instalar_linux.sh · desinstalar_windows.bat
├── .gitignore · .gitattributes
├── .github/
│   └── FUNDING.yml
│
├── libro/                        ← EDICIÓN FINAL
│   ├── README.md                 · qué es cada archivo, ISBN, comprobaciones y SHA-256
│   ├── Libro_v18_interior_KDP.pdf                       · 377 páginas
│   ├── Libro_v18_sin_recortes_limpio.docx
│   ├── Libro_v18_sin_recortes_con_control_de_cambios.docx
│   ├── Cubierta_KDP_377p_color_contraportada_larga.pdf
│   └── Cubierta_KDP_377p_color_contraportada_corta.pdf
│
├── docs/                         ← GUÍAS COMPLEMENTARIAS
│   ├── Guia_entorno_desarrollo.docx                     · instalación y organización del entorno
│   ├── Guia_entorno_desarrollo_con_control_de_cambios.docx
│   ├── Guia_profesor.md                                 · plan docente, prácticas y rúbrica (se lee en GitHub)
│   └── Guia_profesor.docx
│
├── programas/                    ← 117 archivos .py, idénticos a los listados del libro
│   ├── programa_1a_activaciones.py … programa_9b_cnn.py     (programas numerados 1–9, con el 8, el 8 documentado y el 8bis)
│   ├── programa_O1 … O4          · optimizadores (Cap. 5)
│   ├── programa_R1 … R4          · sobreajuste y regularización (Cap. 7)
│   ├── programa_BPE1, BPE2       · tokenizador (Cap. 9)
│   ├── programa_RNN1, RNN2       · redes recurrentes (Cap. 10)
│   ├── programa_S1, S2           · muestreo y generación (Cap. 14)
│   ├── programa_RZ1 … RZ4        · razonamiento (Cap. 15)
│   ├── programa_G1 … G4          · GPT-2 real (Cap. 16)
│   ├── programa_EVAL1, EVAL2     · evaluación (Cap. 18)
│   ├── programa_E1, E2           · ética y sesgos (Cap. 19)
│   ├── programa_PT1 … PT7        · PyTorch (Apéndice D)
│   ├── figura_7_atencion_nano_llm.py · fragmento_9q3_bits.py · fragmento_9q4_repeticion.py
│   ├── ejercicio_1_1.py … ejercicio_20_2.py      · 65 soluciones, ejercicio_<capítulo>_<n>
│   └── ejercicio_G_1.py … ejercicio_G_3.py       · 3 soluciones del Apéndice G
│   (gpt2/  ← no está en el repositorio: se descarga aparte, unos 500 MB; ver README)
│
├── salidas/                      ← 161 archivos
│   ├── salida_<programa>.txt     · 123 salidas de referencia, una por ejecución (algunos programas tienen varias)
│   └── experimentos_capitulo_15/ · 38 ejecuciones de los experimentos del Cap. 15
│       ├── ej152_*.txt  (6)      · Ejercicio 15.2 con las semillas 1–5 y 42
│       ├── rz2_*.txt  (16)       · RZ-2 con distintas semillas y tamaños de entrenamiento
│       ├── rz2x75_*.txt  (7)
│       ├── rz4_*.txt  (5)
│       └── sinprev_*.txt  (4)
│
├── diagramas/                    ← 36 archivos
│   ├── Diagrama_2.1 · 3.1 · 4.1 · 4.2 · 7.1 · 7.2 · 7.3 · 7.4 · 9.1 · 10.1 · 10.2
│   ├── Diagrama_11.1 · 11.2 · 12.1 · 13.1 · 17.1 · 20.1 · 20.2 · G.1          (19 en total, .png)
│   ├── originales/  (8)          · versiones antes de los retoques (13.1, 17.1, 5.1, G.1…)
│   └── scripts/  (9)             · los scripts que hicieron cada retoque (fix_d7_4_g1.py, fix_d13_1_d17_1.py…)
│
└── verificaciones/               ← 8 archivos
    ├── LEEME.md
    ├── verificar_entorno.py                      · versiones, BLAS, hilos y SHA-256 de GPT-2
    ├── verificar_8bis_contra_transformers.py     + su salida .txt
    ├── verificar_gradientes_float64.py           + su salida .txt
    └── verificar_gradientes_RZ.py                + su salida .txt
```
