# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Revisión técnica externa (04-11-2026).

1. El antiguo Diagrama 5.1 (overfitting y regularización) estaba en el capítulo de optimizadores y no ilustraba
   ningún optimizador (hallazgo H42). Pasa al final del Capítulo 7 como Diagrama 7.4: se repinta «5.1» → «7.4»
   en el título.
2. Diagrama G.1 (CNN): el recuadro de parámetros decía 36 + 1.617 = 1.653; el Programa 9B tiene 36 pesos de filtro,
   4 sesgos convolucionales y 1633 parámetros densos: 1673 en total (hallazgo H29). Se repintan las tres líneas.

Uso:  python fix_d7_4_g1.py   (desde la carpeta diagramas/, con Pillow instalado)"""
from PIL import Image, ImageDraw, ImageFont

CONDENSADA = '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'
MONO = '/usr/share/texmf/fonts/opentype/public/lm/lmmonolt10-bold.otf'   # Latin Modern Mono Light Bold


def glifo(f, ch, w, h):
    m = Image.new('L', (300, 300), 0)
    ImageDraw.Draw(m).text((20, 0), ch, font=f, fill=255)
    return m.crop(m.getbbox()).resize((w, h), Image.LANCZOS)


def diagrama_7_4(src='originales/d5_1_orig.png', dst='Diagrama_7.4.png'):
    im = Image.open(src).convert('RGB')
    f = ImageFont.truetype(CONDENSADA, 200)
    d = ImageDraw.Draw(im)
    blanco, tinta = (254, 254, 254), (11, 6, 49)
    d.rectangle((330, 14, 349, 46), fill=blanco)     # el «5»
    d.rectangle((360, 14, 380, 46), fill=blanco)     # el «1» (el punto se conserva)
    color = Image.new('RGB', (40, 40), tinta)
    im.paste(color.crop((0, 0, 15, 28)), (333, 16), glifo(f, '7', 15, 28))
    im.paste(color.crop((0, 0, 16, 28)), (361, 16), glifo(f, '4', 16, 28))
    im.save(dst)


def diagrama_g1(src='originales/dG_1_orig.png', dst='Diagrama_G.1.png'):
    im = Image.open(src).convert('RGB')
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(MONO, 15)
    fondo, tinta, cw, W = (254, 253, 243), (20, 20, 20), 8.1, 36

    def linea(y, izq, der):
        txt = izq + ' ' * (W - len(izq) - len(der)) + der
        d.rectangle((38, y - 1, 345, y + 13), fill=fondo)
        for k, ch in enumerate(txt):
            d.text((42 + k * cw, y - 5), ch, font=f, fill=tinta)

    linea(721, 'filtros 4×3×3 + 4 sesgos:', '40 params')
    linea(736, 'capa densa:', '1633 params')
    linea(765, 'TOTAL CNN:', '1673 params')
    im.save(dst)


if __name__ == '__main__':
    diagrama_7_4()
    diagrama_g1()
