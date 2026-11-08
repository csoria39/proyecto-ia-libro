# Proyecto IA — De la Neurona al LLM · versión v18 (edición final)

Versión elegida por el autor el 7 de noviembre de 2026 para la edición impresa.

| Archivo | Qué es |
|---|---|
| `Libro_v18_interior_KDP.pdf` | Interior para Amazon KDP: A4, color, 377 páginas |
| `Libro_v18_sin_recortes_limpio.docx` | El mismo libro en Word, sin marcas de revisión |
| `Libro_v18_sin_recortes_con_control_de_cambios.docx` | El mismo, con todos los cambios de la revisión v18 marcados |
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

## SHA-256

```
097ef3f2b4b7cf11327092e3c08eb98cddf159bb95635114a827832ba2e91931  Cubierta_KDP_377p_color_contraportada_corta.pdf
9d9b52a0400c8b490590c7807a4e02d2f1d9167951bbe839aaeefc67532d8bc2  Cubierta_KDP_377p_color_contraportada_larga.pdf
3134370217b7d0816a016c32c6fb50ee57f434450ffe5d9e586ae1d74c3e46c2  Libro_v18_interior_KDP.pdf
c3ee5744a81909d09009331e0c1039f7ea35cda88963d22cdaeaa9e39cdfd496  Libro_v18_sin_recortes_con_control_de_cambios.docx
aec9c0e92e61a41ac41fbfc85984d00b667d2f671a8c622e1962675e45f2cefc  Libro_v18_sin_recortes_limpio.docx
```
