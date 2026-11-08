# Repintado de diagramas (8-11-2026)

Estos scripts corrigen ocho diagramas (4.2, 7.3, 9.1, 10.2, 11.1, 20.1, 20.2 y G.1) a partir de la versión anterior,
guardada en `diagramas/originales/Diagrama_<n>_antes_repintado_2026-11-08.png`. Borran solo las zonas que cambian
y escriben encima el texto, las cifras o los gráficos corregidos. Lo que se corrige en cada diagrama, y por qué,
está en la cabecera de su prompt (`diagramas/prompts/`).

- `pintar.py`: escribe texto con supermuestreo y mide el tamaño y la línea base del texto original para que el
  nuevo ocupe el mismo sitio.
- `repintar_<n>.py`: un script por diagrama. Sin argumentos, sobrescribe `diagramas/Diagrama_<n>.png`.

Fuentes: Fira Sans y Fira Sans Condensed (licencia SIL OFL), de los paquetes npm `@fontsource/fira-sans` y
`@fontsource/fira-sans-condensed` (archivos `files/*.woff`), en una carpeta `fuentes/` junto a los scripts o en la
indicada por `FIRA_DIR`; Latin Modern y DejaVu, del sistema (paquetes `fonts-lmodern` y `fonts-dejavu`).

Las imágenes del libro (`word/media` del .docx) son estas mismas, giradas 90° salvo la del 4.2. 11.1, 20.1, 20.2 y G.1 se
guardan con paleta de 256 colores, como las anteriores.
