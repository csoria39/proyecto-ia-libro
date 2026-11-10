# Proyecto IA — De la Neurona al LLM

[![Apoya el libro en Ko-fi](https://img.shields.io/badge/Ko--fi-Apoya%20el%20libro-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/delaneuronaalllm)

Repositorio oficial del libro de **Carlos Soria Martínez**. Contiene todos los programas y ejercicios resueltos del libro, sus salidas reales y las comprobaciones que respaldan sus cifras.

El libro lleva a un estudiante de primero de Informática, a un autodidacta o a un curioso con Python básico desde una sola neurona hasta un GPT-2 completo, programado con NumPy. Al final, con ese mismo código, se ejecuta el GPT-2 real de OpenAI.

## Empieza por aquí

| Qué buscas | Dónde está |
|---|---|
| **El libro** (edición final) | [descarga el PDF](https://github.com/csoria39/proyecto-ia-libro/releases/download/v18/De_la_Neurona_al_LLM_interior_KDP.pdf) (377 páginas) o el [.docx](https://github.com/csoria39/proyecto-ia-libro/releases/download/v18/De_la_Neurona_al_LLM_sin_recortes_limpio.docx), desde la [versión v18](https://github.com/csoria39/proyecto-ia-libro/releases/tag/v18). Copia en [`libro/`](libro/), junto con la [cubierta](libro/Cubierta_KDP_377p_color_contraportada_larga.pdf). Qué es cada archivo, en [`libro/README.md`](libro/README.md). |
| **El mapa del repositorio** | [`ESTRUCTURA.md`](ESTRUCTURA.md): dónde está cada archivo, carpeta por carpeta. |
| **Los instaladores** | En esta misma carpeta: [`instalar_windows.bat`](instalar_windows.bat), [`instalar_mac.sh`](instalar_mac.sh), [`instalar_linux.sh`](instalar_linux.sh) y [`desinstalar_windows.bat`](desinstalar_windows.bat). Cómo usarlos, en [Instalación](#instalación). |
| **La arquitectura del entorno de desarrollo** | [`docs/Guia_entorno_desarrollo.docx`](docs/Guia_entorno_desarrollo.docx): qué se instala, dónde y cómo se organiza, paso a paso en Windows, macOS y Linux. |
| **La guía para el profesor** | [`docs/Guia_profesor.md`](docs/Guia_profesor.md) (se lee aquí en GitHub; también en [.docx](docs/Guia_profesor.docx)): plan de un cuatrimestre, prácticas evaluables y rúbrica. |
| **Las plantillas para lectores de prueba** | [`docs/Plantillas_lectores.md`](docs/Plantillas_lectores.md) (también en [.docx](docs/Plantillas_lectores.docx)): cuestionarios para un lector principiante y otro experto en ML. |
| **La ejecución completa de los programas** | [`verificaciones/Ejecucion_completa_programas.md`](verificaciones/Ejecucion_completa_programas.md) (también en [.txt](verificaciones/Ejecucion_completa_programas.txt)): los 117 listados del libro ejecutados en orden, cada uno con su salida completa y la comparación con lo que imprime el libro. |
| **Las dependencias de Python** | [`requirements.txt`](requirements.txt): versiones mínimas de los paquetes (`pip install -r requirements.txt`). [`requirements-probado.txt`](requirements-probado.txt): versiones exactas con las que se obtuvieron las salidas del libro. |

## Instalación

La forma más sencilla es el script de instalación de tu sistema operativo (sección 1.4 del libro). Descárgalo de esta página y ejecútalo:

| Sistema | Script | Cómo se lanza |
|---|---|---|
| Windows 10/11 | [`instalar_windows.bat`](instalar_windows.bat) | Doble clic (o clic derecho → «Ejecutar como administrador») |
| Linux (Ubuntu, Debian, Mint) | [`instalar_linux.sh`](instalar_linux.sh) | `chmod +x instalar_linux.sh && ./instalar_linux.sh` |
| macOS 12 o posterior | [`instalar_mac.sh`](instalar_mac.sh) | `chmod +x instalar_mac.sh && ./instalar_mac.sh` |

Cada script instala Python (si no tienes la 3.11 o posterior), crea el entorno virtual `entorno_ia` en tu carpeta de usuario, instala en él NumPy (2.3 o posterior), matplotlib, PyTorch (CPU; en Mac con chip Apple, con aceleración MPS), transformers y requests, e instala Visual Studio Code con su extensión de Python. En Windows, [`desinstalar_windows.bat`](desinstalar_windows.bat) deshace la instalación (sin argumentos solo muestra lo que haría). En Linux y macOS basta con borrar la carpeta `~/entorno_ia`.

**Mac con procesador Intel:** la última versión de PyTorch para ese chip es la 2.2, que necesita NumPy 1 y Python 3.12 o anterior (por eso, en un Mac Intel el script instala Python 3.12). Para no cambiar las cifras del libro, el script deja NumPy 2 en `entorno_ia` e instala PyTorch en un segundo entorno, `~/entorno_ia_pytorch`, que se activa solo para los programas del Apéndice D.

Si prefieres hacerlo a mano (Apéndice E), necesitas Python 3.11 o posterior:

```bash
python3 -m venv ~/entorno_ia            # Windows (CMD): python -m venv %USERPROFILE%\entorno_ia
source ~/entorno_ia/bin/activate        # Windows (CMD): %USERPROFILE%\entorno_ia\Scripts\activate.bat
# Solo en Linux: PyTorch en versión CPU antes (si no, pip instala la de CUDA, de varios GB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd programas
python programa_1a_activaciones.py
```

Solo NumPy es imprescindible. Las gráficas usan matplotlib, y el Apéndice D y las verificaciones usan PyTorch y transformers.

## Estructura

El mapa completo, archivo por archivo, está en [ESTRUCTURA.md](ESTRUCTURA.md).

| Carpeta | Contenido |
|---|---|
| `programas/` | Los 117 programas y soluciones de ejercicios del libro, en una sola carpeta. Así funcionan las dependencias entre ellos: por ejemplo, RZ-1 lee `programa_8_nano_llm.py` y varios ejercicios ejecutan el programa de su capítulo. |
| `salidas/` | La salida real de cada programa, tal como aparece en el libro. `experimentos_capitulo_15/` contiene todas las ejecuciones de las tablas del Capítulo 15. |
| `verificaciones/` | La ejecución completa de los 117 listados del libro, en orden y con su salida (`Ejecucion_completa_programas`, en .md y .txt), y comprobaciones que no van en el libro: el gradiente de los programas con diferencias finitas, y el Programa 8 bis frente a `transformers` (mismos logits y mismo tokenizador). |
| `libro/` | La edición final del libro: interior para Amazon KDP (377 páginas, PDF), el .docx con y sin control de cambios, y las cubiertas. |
| `docs/` | Guías complementarias: la guía del entorno de desarrollo (instalación paso a paso en Windows, macOS y Linux, en .docx limpio y con control de cambios) la guía para el profesor (plan de un cuatrimestre, prácticas evaluables y rúbrica) y las plantillas para lectores de prueba (un lector principiante y otro experto en ML), estas dos en .md y .docx. |
| `diagramas/` | Las 19 ilustraciones del libro, con la numeración por capítulo (Diagrama 2.1 … 20.2 y G.1); en `scripts/`, los programas que corrigieron sus erratas y su numeración (con las fuentes Fira que usan, en `scripts/fuentes/`), en `originales/`, cada versión anterior de cada diagrama, y en `prompts/`, un prompt por diagrama que describe la imagen final y enumera sus correcciones, con las cifras comprobadas (ver `prompts/LEEME.md`). |
| `scripts/` | Scripts de maquetación del libro: `paginacion_2026-11-09/` evita títulos solos al pie y páginas en blanco, y corrige el índice alfabético. |

Los nombres de archivo son los que cita el libro. Cada ejercicio se llama `ejercicio_<capítulo>_<número>.py`: por ejemplo, `ejercicio_15_2.py` es la solución del Ejercicio 15.2, y `ejercicio_G_1.py`, la del Ejercicio G.1. La tabla siguiente indica a qué capítulo pertenece cada archivo.

Cada archivo es idéntico al impreso en el libro, salvo la primera línea, que indica su licencia (`# SPDX-License-Identifier: MIT`).

## Orden de lectura

Sigue el orden del libro. El **Programa 8** (`programa_8_nano_llm.py`, Capítulo 13) guarda `nano_llm_pesos.npz`, y los Capítulos 14, 15 y 18 lo reutilizan: ejecútalo antes que ellos.

| Capítulo | Programas y soluciones |
|---|---|
| Capítulo 1 | `ejercicio_1_1.py`, `ejercicio_1_2.py`, `ejercicio_1_3.py` |
| Capítulo 2 | `ejercicio_2_1.py`, `ejercicio_2_2.py`, `ejercicio_2_3.py` |
| Capítulo 3 | `programa_1a_activaciones.py`, `programa_1b_neurona.py`, `ejercicio_3_1.py`, `ejercicio_3_2.py`, `ejercicio_3_3.py` |
| Capítulo 4 | `programa_2_neurona_aprende.py`, `ejercicio_4_1.py`, `ejercicio_4_2.py`, `ejercicio_4_3.py` |
| Capítulo 5 | `programa_O1_momentum.py`, `programa_O2_optimizadores.py`, `programa_O3_comparativa.py`, `programa_O4_lr_scheduling.py`, `ejercicio_5_1.py`, `ejercicio_5_2.py`, `ejercicio_5_3.py` |
| Capítulo 6 | `programa_3_mlp.py`, `programa_4_xor.py`, `ejercicio_6_1.py`, `ejercicio_6_2.py`, `ejercicio_6_3.py` |
| Capítulo 7 | `programa_R1_overfitting.py`, `programa_R2_regularizacion.py`, `programa_R3_dropout.py`, `programa_R4_early_stopping.py`, `ejercicio_7_1.py`, `ejercicio_7_2.py`, `ejercicio_7_3.py` |
| Capítulo 8 | `programa_5_embeddings.py`, `ejercicio_8_1.py`, `ejercicio_8_2.py`, `ejercicio_8_3.py` |
| Capítulo 9 | `programa_BPE1_entrenador.py`, `programa_BPE2_codificador.py`, `ejercicio_9_1.py`, `ejercicio_9_2.py`, `ejercicio_9_3.py` |
| Capítulo 10 | `programa_RNN1_rnn_basico.py`, `programa_RNN2_vanishing_gradient.py`, `ejercicio_10_1.py`, `ejercicio_10_2.py`, `ejercicio_10_3.py` |
| Capítulo 11 | `programa_6_atencion.py`, `ejercicio_11_1.py`, `ejercicio_11_2.py`, `ejercicio_11_3.py`, `ejercicio_11_4.py`, `figura_7_atencion_nano_llm.py` |
| Capítulo 12 | `programa_7_transformer_bloque.py`, `ejercicio_12_1.py`, `ejercicio_12_2.py`, `ejercicio_12_3.py` |
| Capítulo 13 | `programa_8_nano_llm.py`, `programa_8_nano_llm_documentado.py`, `ejercicio_13_1.py`, `ejercicio_13_2.py`, `ejercicio_13_3.py`, `ejercicio_13_4.py`, `ejercicio_13_5.py`, `ejercicio_13_6.py`, `ejercicio_13_7.py` |
| Capítulo 14 | `programa_S1_visualizacion.py`, `programa_S2_generacion_completa.py`, `ejercicio_14_1.py`, `ejercicio_14_2.py`, `ejercicio_14_3.py` |
| Capítulo 15 | `programa_RZ1_memoriza.py`, `programa_RZ2_suma.py`, `programa_RZ3_deduce_letras.py`, `programa_RZ4_deduce_frases.py`, `ejercicio_15_1.py`, `ejercicio_15_2.py`, `ejercicio_15_3.py`, `ejercicio_15_4.py`, `ejercicio_15_5.py` |
| Capítulo 16 | `programa_8bis_gpt2_real.py`, `programa_G1_lente_logits.py`, `programa_G2_induccion.py`, `programa_G3_kvcache.py`, `programa_G4_ajuste_fino.py`, `ejercicio_16_1.py`, `ejercicio_16_2.py`, `ejercicio_16_3.py`, `ejercicio_16_4.py`, `ejercicio_16_5.py`, `fragmento_9q3_bits.py`, `fragmento_9q4_repeticion.py` |
| Capítulo 17 | `ejercicio_17_1.py`, `ejercicio_17_2.py`, `ejercicio_17_3.py` |
| Capítulo 18 | `programa_EVAL1_perplexity.py`, `programa_EVAL2_bleu_benchmark.py`, `ejercicio_18_1.py`, `ejercicio_18_2.py`, `ejercicio_18_3.py` |
| Capítulo 19 | `programa_E1_sesgo.py`, `programa_E2_fairness.py` |
| Capítulo 20 | `ejercicio_20_1.py`, `ejercicio_20_2.py` |
| Apéndice D | `programa_PT1_tensores_activaciones.py`, `programa_PT2_neurona_and.py`, `programa_PT3_mlp_xor.py`, `programa_PT4_embeddings.py`, `programa_PT5_self_attention.py`, `programa_PT6_transformer_bloque.py`, `programa_PT7_nano_llm.py` |
| Apéndice G | `programa_9a_convolucion.py`, `programa_9b_cnn.py`, `ejercicio_G_1.py`, `ejercicio_G_2.py`, `ejercicio_G_3.py` |

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

- **Código** (`programas/`, `verificaciones/`, `scripts/` y scripts de `diagramas/`): licencia MIT, © 2026 Carlos Soria Martínez. Archivo [`LICENSE`](LICENSE).
- **Libro** (texto, ilustraciones y diagramas, PDF y DOCX): Creative Commons Atribución–NoComercial–SinDerivadas 4.0 Internacional (CC BY-NC-ND 4.0). Archivo [`LICENSE-LIBRO.md`](LICENSE-LIBRO.md).
- **Fuentes tipográficas** (`diagramas/scripts/fuentes/`): Fira Sans, Fira Sans Condensed y Fira Mono, SIL Open Font License 1.1 (licencias junto a las fuentes).
- **Pesos de GPT-2:** de OpenAI, con licencia MIT. No se incluyen.
- **Código de terceros:** en `programas/programa_8bis_gpt2_real.py` (y en `ejercicio_16_1.py`, que lo reutiliza), la función `bytes_a_unicode()` y la expresión regular `TROCEO` reproducen las del tokenizador de GPT-2, `src/encoder.py` del repositorio [openai/gpt-2](https://github.com/openai/gpt-2) (© 2019 OpenAI, licencia MIT): tienen que ser idénticas para obtener los mismos tokens que el modelo.

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

Hace falta NumPy 2.3 o posterior. Con NumPy 2.2 los programas funcionan, pero los entrenamientos largos (el Programa 8 y los que parten de él) se separan en los decimales a partir de unos miles de pasos, y las pérdidas finales y el texto generado ya no coinciden con los del libro. Se ha comprobado con el Programa 8: con NumPy 2.3.5 y 2.4.6 la salida es idéntica; con 2.2.6 cambia desde el paso 4000, tanto en Python 3.10 como en 3.11. Como Python 3.10 no admite NumPy 2.3, el libro y los scripts piden Python 3.11 o posterior.
