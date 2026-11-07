# Guía para el profesor — De la Neurona al LLM

Carlos Soria Martínez · revisada el 7 de noviembre de 2026

## El libro en el aula

El libro sirve para que un alumno de primero de Informática construya, solo con Python y NumPy, un GPT-2 completo y lo ejecute después con los pesos reales de OpenAI. Cada pieza se explica, se programa y se comprueba con una ejecución real.

- **Requisitos:** Python básico (funciones, bucles, listas) y matemáticas de bachillerato (derivadas, producto de matrices); el libro repasa NumPy en 1.5 y la derivada en 4.3. No hace falta saber de IA.
- **Lo que aprende el alumno:** neurona y retropropagación, optimizadores, MLP y sobreajuste, embeddings y BPE, atención y Transformer, un LLM entrenado desde cero, muestreo, memorización frente a razonamiento, y el GPT-2 real cargado en su propio código.
- **Material:** 377 páginas (ISBN 9798193023975) con 20 capítulos, 7 apéndices, 45 programas, 68 ejercicios resueltos, glosario de 141 términos e índice alfabético de materias. Todo el código y sus salidas están en el repositorio del libro, que trae también el libro en PDF (carpeta `libro/`) y esta guía (carpeta `docs/`).
- **Licencia:** el texto es CC BY-NC-ND 4.0 y el código, MIT, así que puede usarse libremente en clase y repartirse a los alumnos.

## Asignaturas en las que encaja

El libro completo cubre una asignatura cuatrimestral de introducción al aprendizaje profundo; sus partes sirven como lectura complementaria en otras.

| Asignatura | Curso | Capítulos | Uso |
| --- | --- | --- | --- |
| Introducción a la IA / al aprendizaje profundo | 2.º-3.º de grado | 1-16 (17-19 como lectura) | Texto base de la asignatura |
| Programación (proyecto final) | 1.º de grado | 2-6 | Proyecto: una red que aprende XOR desde cero |
| Álgebra lineal o Cálculo | 1.º de grado | 4, 6, 11 y 13.4 | Aplicación: matrices, regla de la cadena, gradientes comprobados |
| Procesamiento del lenguaje natural | 3.º-4.º de grado | 8, 9, 11-16 | Prácticas: BPE, atención, Nano-LLM y GPT-2 real |
| Ética y aspectos legales de la informática | Cualquiera | 17-19 | Lectura: sesgos, evaluación, AI Act |
| Máster o curso de especialización (nivelación) | Posgrado | 11-16 y Apéndices D y F | Puesta al día rápida |

Las rutas del prólogo (curioso, programador, completa) ya separan los capítulos de código de los de lectura.

## Plan de un cuatrimestre (15 semanas)

Pensado para 2 horas de teoría y 2 de laboratorio por semana. Los ejercicios en negrita son los que propongo evaluar (sección de prácticas). Con 3 horas semanales, el Capítulo 10 y la segunda mitad del 15 pasan a lectura (ver abajo) y los programas largos quedan como tarea en casa; con 6 horas, el plan cabe en 10 semanas y quedan 5 para el proyecto final.

| Semana | Capítulos | Programas | Ejercicios en clase |
| --- | --- | --- | --- |
| 1 | 1. IA, ML y LLM; entorno | — | 1.1, 1.2 |
| 2 | 2-3. La neurona | 1A, 1B | 2.1, 3.2 |
| 3 | 4. Gradientes y retropropagación | 2 | **4.1**, 4.2 |
| 4 | 5. Optimizadores | O-1 a O-4 | 5.2 |
| 5 | 6. MLP y XOR | 3, 4 | **6.1**, 6.2 |
| 6 | 7. Sobreajuste y regularización | R-1 a R-4 | 7.3 |
| 7 | 8-9. Embeddings y BPE | 5, BPE-1, BPE-2 | 8.3, **9.1** |
| 8 | 10. Redes recurrentes (lectura) y repaso | RNN-1, RNN-2 | 10.1 |
| 9 | 11. Atención | 6 | **11.4**, 11.2 |
| 10 | 12. El bloque Transformer | 7 | **12.1**, 12.2 |
| 11 | 13.1-13.3. Nano-LLM: datos y entrenamiento | 8 | 13.2, 13.3 |
| 12 | 13.4-13.6. El backward completo | 8 (comentado) | **13.6**, 13.7 |
| 13 | 14-15. Muestreo; ¿memoriza o razona? | S-1, S-2, RZ-1, RZ-2 | 14.2, **15.1** |
| 14 | 16. Tu código mueve GPT-2 | 8 bis, G-1 a G-3 | **16.2**, 16.1 |
| 15 | 17-19 (lectura) y presentación de proyectos | — | 17.1, 18.1 |

Si el grupo va justo de tiempo, el Capítulo 10 y la segunda mitad del 15 (RZ-3 y RZ-4) pueden quedar como lectura sin perder el hilo.

## Formatos cortos

**Seminario de 6 sesiones de 2 horas** (alumnos que ya han visto redes neuronales):

1. Atención a mano y en NumPy: 11.2-11.4, Programa 6, Ejercicio 11.4.
2. El bloque Transformer: Capítulo 12, Programa 7, Ejercicio 12.1.
3. El Nano-LLM: 13.1-13.3, ejecutar el Programa 8 y cambiar la temperatura (Ejercicio 13.2).
4. El backward comprobado: 13.4 y Ejercicio 13.6 (diferencias finitas).
5. ¿Memoriza o razona?: 15.1-15.3, Programas RZ-1 y RZ-2.
6. GPT-2 real: 16.1-16.6, Programas 8 bis y G-1.

**Taller de un día (6 horas, divulgación o bachillerato)**: Capítulos 1 y 2 sin código; demostración del Programa 8 generando texto; el banco y el río (11.6) a mano; GPT-2 real (Programa 8 bis) en directo; y una hora de debate con el Capítulo 19 (sesgos y AI Act). El profesor ejecuta; los asistentes solo leen y discuten las salidas.

## Prácticas evaluables y rúbrica

Cinco prácticas (60 % de la nota) y un proyecto final (40 %). Las soluciones están en el libro, así que cada práctica pide una variación que la solución no resuelve.

| Práctica | Ejercicio de partida | Variación que se pide | Peso |
| --- | --- | --- | --- |
| P1. Una neurona que aprende | 4.1 | Entrenar XOR con una sola neurona, explicar por qué no converge y medir cuántos pasos tarda en estancarse | 10 % |
| P2. MLP desde cero | 6.1 | Comparar 2, 4 y 8 neuronas ocultas con 5 semillas y presentar media y desviación | 10 % |
| P3. Atención a mano | 11.4 | Repetir el cálculo con una frase propia de 4 palabras, a mano y con NumPy | 10 % |
| P4. Verifica tu backward | 13.6 | Introducir a propósito un error en la derivada de LayerNorm y mostrar cómo lo detecta la comprobación | 15 % |
| P5. ¿Memoriza? | 15.1 | Entrenar el Nano-LLM con un corpus propio de unas 100 frases y medir validación y copia literal | 15 % |
| Proyecto final | 15.3 o 16.2 | Sumar números de tres cifras con RZ-2 (la ampliación que la solución de 15.3 deja abierta), o aplicar la lente de logits a frases en español y en inglés, con informe de 4 páginas | 40 % |

**Rúbrica común (sobre 10):**

- **El programa funciona (3):** se ejecuta tal cual y produce las salidas que se piden.
- **Las cifras son reales (3):** todas salen de una ejecución entregada, con semilla fijada, y se repiten con varias semillas cuando el resultado depende de ella.
- **Explicación (3):** el alumno explica por qué ocurre lo que mide, no solo qué ocurre, y lo relaciona con el capítulo.
- **Honestidad del resultado (1):** si algo no funciona, lo dice y propone una causa. Es la misma regla que sigue el libro.

## Recursos y requisitos del aula

Un ordenador de aula normal basta: todo corre en CPU con Python 3.11 o posterior y NumPy 2.3 o posterior, sin GPU. Los tiempos son los del Apéndice E, medidos con un solo núcleo.

| Programa | Tiempo aproximado | Memoria | Nota para la clase |
| --- | --- | --- | --- |
| Programa 8 (Nano-LLM, 12 000 pasos) | 2.5 minutos | — | Se puede lanzar al empezar la clase |
| RZ-1 (tras el Programa 8) | 2 minutos | — | Necesita los pesos que guarda el Programa 8 |
| RZ-2 (sumas) | 3 minutos | — |  |
| RZ-3 (deducción con letras) | 8 minutos | — | Mejor como tarea en casa |
| RZ-4 (deducción con frases) | 35 minutos | — | Solo como tarea en casa |
| Programa 8 bis (GPT-2 real) | 2 minutos | 1.1 GB | Descargar antes los 548 MB de pesos |
| G-1 (lente de logits) | 25 segundos | 1.1 GB |  |
| PT-7 (PyTorch, CPU) | 2 minutos | — | Solo si el aula tiene PyTorch; con `OMP_NUM_THREADS=1` si comparte la CPU con otros programas |

- **Repositorio:** todos los programas, sus salidas de referencia y un `requirements.txt` (Apéndice E). `requirements-probado.txt` fija las versiones exactas con las que se generaron las salidas, y `verificaciones/verificar_entorno.py` comprueba un ordenador (versiones, hilos y huellas de los pesos de GPT-2). Cada ejercicio se llama `ejercicio_<capítulo>_<número>.py`, y `ESTRUCTURA.md` muestra dónde está cada archivo. Conviene que el profesor ejecute antes los programas en los ordenadores del aula y compare con las salidas de `salidas/`.
- **Pesos de GPT-2:** 548 MB desde huggingface.co/openai-community/gpt2; mejor descargarlos una vez en una carpeta compartida que dejar que lo haga cada alumno.
- **Ilustraciones:** los diagramas del libro, numerados por capítulo, están en el repositorio en PNG y pueden proyectarse en clase.

## Dificultades habituales

| Dificultad | Dónde aparece | Cómo abordarla |
| --- | --- | --- |
| Las formas de las matrices (T × d, B × T × D) | Caps. 11-13 | Que el alumno escriba la forma junto a cada línea, como hace 11.3 («Las formas, paso a paso») |
| La regla de la cadena con muchas capas | Caps. 4, 6 y 13.4 | Empezar por el ejemplo con números de 4.4 y comprobar siempre con diferencias finitas (Ejercicio 13.6) |
| Confundir puntuaciones y pesos de atención | 11.4 | Insistir en que −∞ solo existe antes del softmax; después es 0 (cuadro «Las formas, paso a paso» de 11.3) |
| Pensar que una pérdida baja es un buen modelo | 13.6 y 15.2 | Mostrar el 0.21 de entrenamiento frente al 4.18 de validación (Programa RZ-1) |
| Resultados que cambian con la semilla | Caps. 7 y 15 | Pedir siempre varias semillas: en 15.3 una cabeza aprende con 4 de 6 y dos con 6 de 6 |
| Esperar que el Nano-LLM «entienda» | Caps. 13-16 | Usar el Capítulo 16 como respuesta: lo que faltaba era preentrenamiento, no otra arquitectura |
| Instalación en Windows y rutas de archivos | Apéndice E | Dedicar la primera semana de laboratorio a dejar el entorno funcionando, con la guía del entorno de desarrollo (carpeta `docs/`) |

## Curso piloto

Un grupo de 15 a 25 alumnos de primero o segundo durante el seminario de 6 sesiones basta para saber si el libro funciona en el aula y dónde se atascan.

1. **Antes de empezar:** un cuestionario de 10 preguntas sobre los conceptos del seminario (atención, softmax, pérdida, validación) y su nivel de Python.
2. **En cada sesión:** registrar cuánto tarda cada alumno en el ejercicio, qué preguntas hace y en qué página; recoger las erratas en `ERRATAS.md`.
3. **Al terminar:** el mismo cuestionario y una encuesta breve: qué capítulo se entendió peor, qué programa no funcionó y qué cambiarían.
4. **Resultado:** un informe de dos páginas con la mejora media en el cuestionario, las tres secciones más difíciles y las erratas, que alimenta la siguiente edición.

Si el piloto lo dirige un profesor distinto del autor, su informe vale además como primera opinión externa del libro.
