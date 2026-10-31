# Proyecto IA — De la Neurona al LLM

[![Apoya el libro en Ko-fi](https://img.shields.io/badge/Ko--fi-Apoya%20el%20libro-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/delaneuronaalllm)

Repositorio oficial del libro de **Carlos Soria Martínez**. Contiene todos los programas y ejercicios resueltos del libro, sus salidas reales y las comprobaciones que respaldan sus cifras.

El libro lleva a un estudiante de primero de Informática, a un autodidacta o a un curioso con Python básico desde una sola neurona hasta un GPT-2 completo, programado con NumPy. Al final, con ese mismo código, se ejecuta el GPT-2 real de OpenAI.

## Instalación

Necesitas Python 3.10 o posterior. El libro explica la instalación paso a paso en la sección 1.4 y en el Apéndice E. Resumen:

```bash
python -m venv entorno_ia
source entorno_ia/bin/activate          # Windows: entorno_ia\Scripts\activate
pip install -r requirements.txt
cd programas
python programa_1a_activaciones.py
```

Solo NumPy es imprescindible. Las gráficas usan matplotlib, y el Apéndice D y las verificaciones usan PyTorch y transformers.

## Estructura

| Carpeta | Contenido |
|---|---|
| `programas/` | Los 117 programas y soluciones de ejercicios del libro, en una sola carpeta. Así funcionan las dependencias entre ellos: por ejemplo, RZ-1 lee `programa_8_nano_llm.py` y varios ejercicios ejecutan el programa de su capítulo. |
| `salidas/` | La salida real de cada programa, tal como aparece en el libro. `experimentos_capitulo_15/` contiene todas las ejecuciones de las tablas del Capítulo 15. |
| `verificaciones/` | Comprobaciones que no van en el libro: el gradiente de los programas con diferencias finitas, y el Programa 8 bis frente a `transformers` (mismos logits y mismo tokenizador). |
| `diagramas/` | Las 19 ilustraciones del libro, con la numeración por capítulo (Diagrama 2.1 … 20.2 y G.1); en `scripts/`, los programas que corrigieron sus erratas y su numeración, y en `originales/`, las versiones del autor antes de corregirlas. |

Los nombres de archivo son los que cita el libro. Algunos conservan la numeración de los capítulos de una versión anterior; por ejemplo, `ejercicio_9t_2.py` es la solución del Ejercicio 15.2. La tabla siguiente indica a qué capítulo pertenece cada uno.

Cada archivo es idéntico al impreso en el libro, salvo la primera línea, que indica su licencia (`# SPDX-License-Identifier: MIT`).

## Orden de lectura

Sigue el orden del libro. El **Programa 8** (`programa_8_nano_llm.py`, Capítulo 13) guarda `nano_llm_pesos.npz`, y los Capítulos 14, 15 y 18 lo reutilizan: ejecútalo antes que ellos.

| Capítulo | Programas y soluciones |
|---|---|
| Capítulo 1 | `ejercicio_1_1.py`, `ejercicio_1_2.py`, `ejercicio_1_3.py` |
| Capítulo 2 | `ejercicio_2_1.py`, `ejercicio_2_2.py`, `ejercicio_2_3.py` |
| Capítulo 3 | `programa_1a_activaciones.py`, `programa_1b_neurona.py`, `ejercicio_3_1.py`, `ejercicio_3_2.py`, `ejercicio_3_3.py` |
| Capítulo 4 | `programa_2_neurona_aprende.py`, `ejercicio_4_1.py`, `ejercicio_4_2.py`, `ejercicio_4_3.py` |
| Capítulo 5 | `programa_O1_momentum.py`, `programa_O2_optimizadores.py`, `programa_O3_comparativa.py`, `programa_O4_lr_scheduling.py`, `ejercicio_4b_1.py`, `ejercicio_4b_2.py`, `ejercicio_4b_3.py` |
| Capítulo 6 | `programa_3_mlp.py`, `programa_4_xor.py`, `ejercicio_5_1.py`, `ejercicio_5_2.py`, `ejercicio_5_3.py` |
| Capítulo 7 | `programa_R1_overfitting.py`, `programa_R2_regularizacion.py`, `programa_R3_dropout.py`, `programa_R4_early_stopping.py`, `ejercicio_5b_1.py`, `ejercicio_5b_2.py`, `ejercicio_5b_3.py` |
| Capítulo 8 | `programa_5_embeddings.py`, `ejercicio_6_1.py`, `ejercicio_6_2.py`, `ejercicio_6_3.py` |
| Capítulo 9 | `programa_BPE1_entrenador.py`, `programa_BPE2_codificador.py`, `ejercicio_6b_1.py`, `ejercicio_6b_2.py`, `ejercicio_6b_3.py` |
| Capítulo 10 | `programa_RNN1_rnn_basico.py`, `programa_RNN2_vanishing_gradient.py`, `ejercicio_6t_1.py`, `ejercicio_6t_2.py`, `ejercicio_6t_3.py` |
| Capítulo 11 | `programa_6_atencion.py`, `ejercicio_7_1.py`, `ejercicio_7_2.py`, `ejercicio_7_3.py`, `ejercicio_7_4.py`, `figura_7_atencion_nano_llm.py` |
| Capítulo 12 | `programa_7_transformer_bloque.py`, `ejercicio_8_1.py`, `ejercicio_8_2.py`, `ejercicio_8_3.py` |
| Capítulo 13 | `programa_8_nano_llm.py`, `programa_8_nano_llm_documentado.py`, `ejercicio_9_1.py`, `ejercicio_9_2.py`, `ejercicio_9_3.py`, `ejercicio_9_4.py`, `ejercicio_9_5.py`, `ejercicio_9_6.py`, `ejercicio_9_7.py` |
| Capítulo 14 | `programa_S1_visualizacion.py`, `programa_S2_generacion_completa.py`, `ejercicio_9b_1.py`, `ejercicio_9b_2.py`, `ejercicio_9b_3.py` |
| Capítulo 15 | `programa_RZ1_memoriza.py`, `programa_RZ2_suma.py`, `programa_RZ3_deduce_letras.py`, `programa_RZ4_deduce_frases.py`, `ejercicio_9t_1.py`, `ejercicio_9t_2.py`, `ejercicio_9t_3.py`, `ejercicio_9t_4.py`, `ejercicio_9t_5.py` |
| Capítulo 16 | `programa_8bis_gpt2_real.py`, `programa_G1_lente_logits.py`, `programa_G2_induccion.py`, `programa_G3_kvcache.py`, `programa_G4_ajuste_fino.py`, `ejercicio_9q_1.py`, `ejercicio_9q_2.py`, `ejercicio_9q_3.py`, `ejercicio_9q_4.py`, `ejercicio_9q_5.py`, `fragmento_9q3_bits.py`, `fragmento_9q4_repeticion.py` |
| Capítulo 17 | `ejercicio_10_1.py`, `ejercicio_10_2.py`, `ejercicio_10_3.py` |
| Capítulo 18 | `programa_EVAL1_perplexity.py`, `programa_EVAL2_bleu_benchmark.py`, `ejercicio_10b_1.py`, `ejercicio_10b_2.py`, `ejercicio_10b_3.py` |
| Capítulo 19 | `programa_E1_sesgo.py`, `programa_E2_fairness.py` |
| Capítulo 20 | `ejercicio_11b_1.py`, `ejercicio_11b_2.py` |
| Apéndice D | `programa_PT1_tensores_activaciones.py`, `programa_PT2_neurona_and.py`, `programa_PT3_mlp_xor.py`, `programa_PT4_embeddings.py`, `programa_PT5_self_attention.py`, `programa_PT6_transformer_bloque.py`, `programa_PT7_nano_llm.py` |
| Apéndice G | `programa_9a_convolucion.py`, `programa_9b_cnn.py`, `ejercicio_11_1.py`, `ejercicio_11_2.py`, `ejercicio_11_3.py` |

## Los pesos de GPT-2 (Capítulo 16)

Los programas del Capítulo 16 y la carpeta `verificaciones/` necesitan los pesos reales de GPT-2 small, que **no** están en este repositorio. Se descargan una sola vez (548 MB):

- **Con el navegador:** abre <https://huggingface.co/openai-community/gpt2>, pestaña «Files and versions», y descarga `model.safetensors`, `vocab.json`, `merges.txt` y `config.json`. Guárdalos en una carpeta `gpt2/` dentro de `programas/`.
- **Con Python** (`pip install huggingface_hub`):

  ```python
  from huggingface_hub import hf_hub_download
  for f in ['model.safetensors', 'vocab.json', 'merges.txt', 'config.json']:
      hf_hub_download('openai-community/gpt2', f, local_dir='programas/gpt2')
  ```

OpenAI publicó GPT-2 con licencia MIT.

## Licencias

- **Código** (`programas/`, `verificaciones/`, scripts de `diagramas/`): licencia MIT, © 2026 Carlos Soria Martínez. Archivo [`LICENSE`](LICENSE).
- **Libro** (texto, ilustraciones y diagramas, PDF y DOCX): Creative Commons Atribución–NoComercial–SinDerivadas 4.0 Internacional (CC BY-NC-ND 4.0). Archivo [`LICENSE-LIBRO.md`](LICENSE-LIBRO.md).
- **Pesos de GPT-2:** de OpenAI, con licencia MIT. No se incluyen.
- **Código de terceros:** en `programas/programa_8bis_gpt2_real.py` (y en `ejercicio_9q_1.py`, que lo reutiliza), la función `bytes_a_unicode()` y la expresión regular `TROCEO` reproducen las del tokenizador de GPT-2, `src/encoder.py` del repositorio [openai/gpt-2](https://github.com/openai/gpt-2) (© 2019 OpenAI, licencia MIT): tienen que ser idénticas para obtener los mismos tokens que el modelo.

## Apoya el proyecto

El libro y todo su código son gratuitos y lo seguirán siendo. Detrás hay muchas horas de escritura, de programas ejecutados uno a uno para que cada cifra del libro sea real y de revisión para que un estudiante de primer curso pueda seguirlo sin tropiezos.

Si te ha servido para entender por fin cómo funciona un LLM, para preparar una asignatura o simplemente te ha gustado, puedes invitar al autor a un café en Ko-fi: es un pago único, del importe que quieras, con tarjeta y sin registrarte.

**[ko-fi.com/delaneuronaalllm](https://ko-fi.com/delaneuronaalllm)**

Cualquier aportación ayuda a mantener el repositorio, corregir erratas y preparar nuevas ediciones. Y si no puedes aportar, también ayudas mucho recomendándolo, dejando una estrella ⭐ en el repositorio o enviando una errata.

## Erratas y contacto

Las erratas conocidas y su corrección están en [`ERRATAS.md`](ERRATAS.md).

Las erratas y sugerencias pueden enviarse a de.la.neurona.al.llm@gmail.com o abrirse como *issue* en este repositorio.

## Comprobación

Todos los programas se ejecutaron desde una copia limpia de este repositorio, con un entorno virtual nuevo de Python 3.11 y NumPy 2.4.6. La salida de cada uno coincide con la de `salidas/`. Solo cambian los tiempos que imprimen algunos programas y la de `programa_PT4_embeddings.py`, que no fija la semilla aleatoria; el libro no imprime su salida.
