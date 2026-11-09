# Proyecto IA — De la Neurona al LLM · versión v18 (edición final)

Versión elegida por el autor el 7 de noviembre de 2026 para la edición impresa.

| Archivo | Qué es |
|---|---|
| `De_la_Neurona_al_LLM_interior_KDP.pdf` | Interior para Amazon KDP: A4, color, 377 páginas |
| `De_la_Neurona_al_LLM_sin_recortes_limpio.docx` | El mismo libro en Word, sin marcas de revisión |
| `De_la_Neurona_al_LLM_sin_recortes_con_control_de_cambios.docx` | El mismo, con todos los cambios de la revisión v18 marcados |
| `Cubierta_KDP_377p_color_contraportada_larga.pdf` | Cubierta completa (contraportada, lomo y portada) para 377 páginas en papel Premium Color |
| `Cubierta_KDP_377p_color_contraportada_corta.pdf` | La misma cubierta, con la contraportada corta |

- ISBN de la edición impresa: 9798193023975.
- El libro se publica con licencia CC BY-NC-ND 4.0 (ver `LICENSE-LIBRO.md`); el código, con licencia MIT (ver `LICENSE`).
- Los 117 listados del libro son idénticos a los archivos de `programas/`, sin contar la línea de licencia SPDX.

Comprobaciones de esta versión:
- los dos .docx pasan la validación;
- el índice general y el índice alfabético coinciden con las páginas del PDF;
- no hay páginas en blanco;
- las 118 ejecuciones de los programas y ejercicios terminan sin errores.

## Diagramas repintados (8-11-2026)

Ocho diagramas cambian de contenido para corregir errores detectados al escribir sus prompts (ver
`diagramas/prompts/LEEME.md`): 4.2, 7.3, 9.1, 10.2, 11.1, 20.1, 20.2 y G.1. Se repintaron por código solo las
zonas afectadas (`diagramas/scripts/repintado_2026-11-08/`). El texto del libro no cambia: el PDF tiene las mismas
377 páginas y el mismo texto, página a página, que la versión anterior.

## Diagramas verificados (9-11-2026)

Se incorporan las correcciones de los otros diez diagramas (2.1, 3.1, 4.1, 7.1, 7.2, 7.4, 10.1, 11.2, 12.1 y 13.1),
revisadas una a una, y la fila de LLaMA 3 en la tabla del 9.1. En 4.1, 7.4 y 13.1 hubo que retocar la versión
verificada (scripts en `diagramas/scripts/repintado_2026-11-09/`). Los 19 diagramas siguen en su capítulo y página;
el PDF mantiene 377 páginas y el mismo texto.

## Revisión completa de los diagramas (9-11-2026, tarde)

Se revisaron los 19 diagramas enteros, no solo las zonas cambiadas, y se corrigieron 16: barras de 2.1 y 17.1 (escala
logarítmica desplazada 2-3 décadas), ReLU, tanh, rangos y árbol de 3.1, curva de pérdida de 4.1, δ y b¹ de 4.2,
tildes de los títulos (3.1, 7.2, 7.4, 9.1), código PyTorch y textos de 7.3, L2 y sesgo/varianza de 7.4, BPE, OOV e
implicación de 9.1, h₄ e y₁-y₃ de 10.1, eje y notación de 10.2, caja multi-head y d_k de 11.1, pesos en disco de 13.1
(0.49 MB: el Programa 8 guarda float64), tildes de 20.1, cuatro detalles de 20.2 y tabla, lema y fórmula de G.1.
Scripts en `diagramas/scripts/repintado_2026-11-09b/`; versiones anteriores en
`diagramas/originales/Diagrama_<n>_antes_2026-11-09b.png`. El PDF mantiene 377 páginas y el mismo texto.

En el Capítulo 16, la tabla Nano-LLM / GPT-2 decía «Archivo de pesos 0.24 MB (float32)»; el Programa 8 guarda
los 60 000 pesos en float64 (`nano_llm_pesos.npz` ocupa 488 646 B), así que ahora dice «0.49 MB (float64)». Es el
único cambio de texto del PDF.

Después se limpiaron dos defectos menores: dos trazos sueltos bajo el título del 10.2 y uno sobre la tabla del panel 3
del 11.2 (`f10_2_titulo.py` y `f11_2.py`; versiones anteriores en `diagramas/originales/Diagrama_<n>_antes_2026-11-09c.png`).

## Paginación (9-11-2026)

Revisión página a página del PDF: números de página (pie de las 377), índice general, índice alfabético
(143 entradas, 844 referencias), saltos de página, títulos al pie, páginas en blanco y posición de los 19 diagramas.
Se corrigió:

- 13 títulos que quedaban solos al pie de página (por ejemplo «7.7 Dropout», «16.5 Mirar dentro (I): la lente de
  logits», «19.3 Casos reales», «Softmax» en 2.6) y 8 títulos de recuadro de código («📄 …») que quedaban al pie sin su
  código. Los estilos Título 1-4, las barras de título de los programas y los títulos «📄» ahora se conservan con lo
  que sigue (`scripts/paginacion_2026-11-09/conservar_con_siguiente.py`).
- Los saltos de página manuales hechos con un párrafo vacío pasan a ser «salto de página anterior» del párrafo
  siguiente: así nunca dejan una página en blanco (con los cambios anteriores habría aparecido una en el Capítulo 9).
- Índice alfabético: 8 referencias que se mueven una página con lo anterior (COMPAS, Dropout, Embeddings, KV-cache,
  Lente de logits, Línea base, Safetensors, Softmax) (`indice_alfabetico.py`).

El texto del libro no cambia; el PDF sigue con 377 páginas, todos los capítulos, partes y apéndices empiezan en la
misma página que indica el índice, y los diagramas siguen en sus páginas.

## Índice alfabético (9-11-2026, tarde)

Comprobación completa de las 143 entradas: orden alfabético, letra de cada grupo, páginas ordenadas y sin repetir,
como mucho ocho por término, rangos válidos y, para cada una de las 844+ referencias, que el término aparece en esa
página. Además se buscaron los títulos de sección que tratan un término cuya página no estaba en su entrada. Cambios
(`scripts/paginacion_2026-11-09/indice_alfabetico_2.py`):

- Forward pass: se añade la 58 (6.3 Forward propagation en una red) y sale la 158 (solo nombra la función forward()).
- Gradiente que desaparece: se añade la 117 (10.5 LSTM, con la medida del gradiente que decae).
- Evaluación de modelos: 296 → 296-297 (18.9 Guía práctica para evaluar tu propio modelo).
- Función de pérdida: 340 → 342 (donde se define la pérdida en PyTorch, nn.BCEWithLogitsLoss; en la 340 solo
  aparece loss.backward()).

Ninguna otra referencia cambia, ni el texto ni la paginación (377 páginas).

## SHA-256

```
097ef3f2b4b7cf11327092e3c08eb98cddf159bb95635114a827832ba2e91931  Cubierta_KDP_377p_color_contraportada_corta.pdf
9d9b52a0400c8b490590c7807a4e02d2f1d9167951bbe839aaeefc67532d8bc2  Cubierta_KDP_377p_color_contraportada_larga.pdf
a6e6869bbaffa411045381c564ce363aa593af2dea2e28bbe0d7fcecc032ff54  De_la_Neurona_al_LLM_interior_KDP.pdf
128e257280c5effe95d63564ba204737b3b19c0bd4f73676a2a8b28efb8b5cc5  De_la_Neurona_al_LLM_sin_recortes_con_control_de_cambios.docx
c9c2d2fbe91fdf8635d1eee66be700a4dfd332ce58dfe87acbb53fd81d808c39  De_la_Neurona_al_LLM_sin_recortes_limpio.docx
```
