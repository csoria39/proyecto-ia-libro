# Prompts de los diagramas

Un prompt por cada uno de los 19 diagramas del libro, escrito el 8-11-2026 a partir de la imagen publicada y del
texto de su capítulo. Las cifras salen de las ejecuciones reales de los programas.

## Cómo está hecho cada prompt

Cada archivo `Prompt_Diagrama_<n>.txt` tiene dos partes:

1. **Cabecera, en español:**
   - dónde va el diagrama en el libro (capítulo, apartado y página del PDF de KDP);
   - qué imagen del repositorio es su referencia;
   - de dónde sale cada cifra;
   - en qué se aparta de la imagen publicada.
2. **El prompt, en inglés.** Es la lengua que mejor siguen los generadores de imágenes. Los textos que deben
   aparecer en la imagen van en español y entre comillas, tal cual. Se describen el tamaño exacto en píxeles, la
   composición, los colores y cada texto, panel por panel, y al final hay una lista de comprobación.

**Sobre la exactitud.** Un generador de imágenes no reproduce una imagen píxel a píxel: con el mismo prompt, dos
ejecuciones dan dibujos distintos. Estos prompts fijan todo lo que se puede fijar: contenido, textos, cifras,
disposición, colores y tamaño. La imagen exacta del libro está en la carpeta `diagramas/`. Para rehacer un diagrama
lo más fiel posible, conviene dar al generador las dos cosas: el prompt y la imagen de referencia.

## Correcciones que llevan los prompts

Al transcribir cada imagen y cotejarla con el capítulo y con las salidas de los programas, han aparecido errores en
16 de los 19 diagramas publicados. Cada prompt describe la imagen **corregida** y explica en su cabecera qué cambia y
por qué. Si se quiere la imagen tal como está publicada, basta con no aplicar esas correcciones.

El 8-11-2026 se corrigieron en el libro los ocho diagramas que cambian de contenido (4.2, 7.3, 9.1, 10.2, 11.1, 20.1,
20.2 y G.1). Se repintaron por código las zonas afectadas sobre la imagen original, con los scripts de
`diagramas/scripts/repintado_2026-11-08/`; las versiones anteriores están en `diagramas/originales/`. Sus prompts
describen exactamente la imagen corregida. Los otros diez con correcciones (2.1, 3.1, 4.1, 7.1, 7.2, 7.4, 10.1, 11.2, 12.1 y 13.1) siguen como estaban
en el libro; sus correcciones pueden aplicarse igual, con el mismo método.

| Diagrama | Pág. | Correcciones | Las más importantes | Estado |
|---|---|---|---|---|
| 2.1 La neurona artificial | 19 | 2 | Barras de softmax que no suman 1 | Pendiente |
| 3.1 Funciones de activación | 27 | 1 | Barras de softmax que no suman 1 | Pendiente |
| 4.1 Gradiente descendente | 36 | 3 | −∇L no es la opuesta de ∇L | Pendiente |
| 4.2 Backpropagation | 37 | 2 | Ejemplo 1-2-1 con cifras incoherentes (mezcla sigmoide y tanh) | Corregido en el libro (8-11-2026) |
| 7.1 Red multicapa (MLP) | 66 | 1 | Dropout clásico en vez del invertido del libro | Pendiente |
| 7.2 Train / Validación / Test | 71 | 2 | 0.001 / 0.847 inventados; ahora 0.032 / 0.091 del Programa R-1 | Pendiente |
| 7.3 Dropout | 76 | 4 | Dropout clásico; resultados que no son los del Programa R-3 | Corregido en el libro (8-11-2026) |
| 7.4 Overfitting y regularización | 84 | 4 | Curva de prueba por debajo de la de entrenamiento | Pendiente |
| 9.1 Tokenización | 98 | 4 | Fusiones BPE imposibles; fertilidad y vocabularios inexactos | Corregido en el libro (8-11-2026) |
| 10.1 Redes recurrentes | 111 | 3 | h₃ repetido; fórmula del gradiente sin la derivada de tanh | Pendiente |
| 10.2 LSTM | 118 | 6 | «Cuatro puertas» (son tres y un candidato); «solo sumas» | Corregido en el libro (8-11-2026) |
| 11.1 Self-attention | 127 | 5 | Matriz causal incoherente con la bidireccional | Corregido en el libro (8-11-2026) |
| 11.2 Self-attention: referencia | 135 | 2 | Coma decimal; matiz de las cabezas en el Capítulo 15 | Pendiente |
| 12.1 El bloque Transformer | 144 | 1 | Separador de miles y coma decimal | Pendiente |
| 13.1 El Nano-LLM | 181 | 1 | Curva de aprendizaje: ahora reproducible | Pendiente |
| 17.1 De nano a gigante | 285 | 0 | — | — |
| 20.1 RAG | 313 | 4 | «Privacidad total»; gráfico y tabla no coinciden | Corregido en el libro (8-11-2026) |
| 20.2 Multimodales | 315 | 4 | El mapa del libro omite los Capítulos 13-16 | Corregido en el libro (8-11-2026) |
| G.1 CNN | 369 | 3 | Ejemplo de convolución incoherente; filtros «aprendidos» que son clásicos | Corregido en el libro (8-11-2026) |

## Cálculos

La carpeta `calculos/` tiene los scripts que dan las cifras que no estaban ya impresas en el libro, y su salida.
Todos usan funciones de los propios programas del libro:

| Script | Diagrama | Qué calcula |
|---|---|---|
| `diagrama_4_2_ejemplo_1_2_1.py` | 4.2 | Forward y backward de la red 1-2-1, comprobados con diferencias finitas |
| `diagrama_9_1_bpe_ejemplo.py` | 9.1 | Las fusiones BPE del corpus {bajo: 6, bajar: 3, caja: 4}, con el Programa BPE-1 |
| `diagrama_9_1_fertilidad.py` | 9.1 | Tokens de GPT-2 por palabra, con el tokenizador del Programa 8 bis |
| `diagrama_13_1_nano_llm.py` | 13.1 | Curva de aprendizaje, temperaturas y top-p del Programa 8 (semilla 42) |
| `diagrama_G_1_convolucion.py` | G.1 | Convolución con Sobel-X, con la función del Programa 9A |

Se ejecutan desde la carpeta `programas/`. Los de 9.1 (fertilidad) necesitan además la carpeta `gpt2/`.
`perdidas_programa_8.txt` guarda la pérdida de cada uno de los 12 000 pasos del Programa 8.

Los prompts antiguos (`Prompt_Diagrama_13`, `14`, `15`, `16`, `13.1` y `17.1`, con la numeración anterior) quedan
sustituidos por estos.
