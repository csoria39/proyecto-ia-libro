# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Diagramas 6, 13, 14, 17, 18 y 19: referencias a capítulos con la numeración antigua y erratas
(verificación final, 26-10-2026). Se trabaja en orientación de lectura (r*.png) y se guarda girado en
corr_image*.png, listo para media_opt. Los originales quedan en orig_image*.png.

Cada cambio: (caja holgada que contiene el texto viejo, texto viejo, texto nuevo, fuente, alineación,
tinta, fondo, ancho máximo). La talla de letra se calcula para que el texto viejo, con esa fuente,
mida lo mismo que en la imagen, y el nuevo se escribe con la misma talla y la misma línea base
(si no cabe en el ancho máximo, se estrecha)."""
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

CB = '/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf'
CR = '/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf'
MB = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
FM = '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'
Z = 4


def fondo_de(im, c):
    x0, y0, x1, y1 = c
    px = [im.getpixel((x, y0)) for x in range(x0, x1)] + [im.getpixel((x, y1)) for x in range(x0, x1)] + \
         [im.getpixel((x0, y)) for y in range(y0, y1)] + [im.getpixel((x1, y)) for y in range(y0, y1)]
    return Counter(px).most_common(1)[0][0]


def tinta_bbox(im, c, fondo, umbral=90):
    a = np.asarray(im.crop(c)).astype(int)
    dist = np.abs(a - np.array(fondo)).sum(axis=2)
    ys, xs = np.where(dist > umbral)
    bb = (c[0] + xs.min(), c[1] + ys.min(), c[0] + xs.max() + 1, c[1] + ys.max() + 1)
    sel = a[dist > umbral]
    # tinta: el color más alejado del fondo (media del 15 % más alejado)
    dd = np.abs(sel - np.array(fondo)).sum(axis=1)
    tinta = tuple(int(v) for v in sel[dd >= np.percentile(dd, 85)].mean(axis=0))
    return bb, tinta


def cambiar(im, caja, viejo, nuevo, fuente, alinear='izq', tinta=None, fondo=None, ancho_max=None, borrar=None):
    d = ImageDraw.Draw(im)
    fondo = fondo or fondo_de(im, caja)
    bb, t_auto = tinta_bbox(im, caja, fondo)
    tinta = tinta or t_auto
    ancho = bb[2] - bb[0]
    # talla tal que «viejo» mida «ancho»
    t = 10.0
    for _ in range(30):
        f = ImageFont.truetype(fuente, int(t * Z))
        w = d.textbbox((0, 0), viejo, font=f)
        w = (w[2] - w[0]) / Z
        t *= ancho / w
    f = ImageFont.truetype(fuente, int(t * Z))
    bv = d.textbbox((0, 0), viejo, font=f)
    bn = d.textbbox((0, 0), nuevo, font=f)
    capa = Image.new('L', (bn[2] - bn[0] + 8 * Z, (max(bn[3], bv[3]) - min(bn[1], bv[1])) + 8 * Z), 0)
    top = min(bn[1], bv[1])
    ImageDraw.Draw(capa).text((4 * Z - bn[0], 4 * Z - top), nuevo, font=f, fill=255)
    wn = (bn[2] - bn[0]) / Z
    amax = ancho_max or max(ancho, wn)
    escala_x = min(1.0, amax / wn)
    capa = capa.resize((max(1, round(capa.width / Z * escala_x)), round(capa.height / Z)), Image.LANCZOS)
    # y: la línea base del texto viejo  (bb[1] corresponde a bv[1])
    y = bb[1] - (bv[1] - top) / Z - 4
    wn *= escala_x
    if alinear == 'izq':
        x = bb[0] - 4 * escala_x
    else:
        x = (bb[0] + bb[2]) / 2 - wn / 2 - 4 * escala_x
    zona = borrar or (bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2)
    d.rectangle(zona, fill=fondo)
    im.paste(Image.new('RGB', capa.size, tinta), (round(x), round(y)), capa)
    return round(t, 1), round(escala_x, 2)


def celda(im, caja, s, fuente, talla, tinta, fondo):
    d = ImageDraw.Draw(im)
    d.rectangle(caja, fill=fondo)
    f = ImageFont.truetype(fuente, talla * Z)
    b = d.textbbox((0, 0), s, font=f)
    capa = Image.new('L', (b[2] - b[0] + 8 * Z, b[3] - b[1] + 8 * Z), 0)
    ImageDraw.Draw(capa).text((4 * Z - b[0], 4 * Z - b[1]), s, font=f, fill=255)
    capa = capa.resize((round(capa.width / Z), round(capa.height / Z)), Image.LANCZOS)
    cx, cy = (caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2
    im.paste(Image.new('RGB', capa.size, tinta), (round(cx - capa.width / 2), round(cy - capa.height / 2)), capa)


CAMBIOS = {
    12: [((828, 699, 1030, 719), '# Python / NumPy (Capítulo 7)', '# Python / NumPy (Capítulo 11)', MB, 'izq', None, None, 215)],
    13: [((300, 46, 1240, 69), 'Resumen visual del Capitulo 7 — fórmulas, matrices y multi-head',
          'Resumen visual del Capítulo 11 — fórmulas, matrices y multi-head', CB, 'centro'),
         ((440, 772, 790, 797), 'solo el pasado — tokens futuros bloqueados',
          'solo el pasado: −∞ antes del softmax → 0 después', CB, 'centro', None, None, 345),
         ((640, 505, 706, 523), 'ASIENTO', 'ASIENTO', CB, 'izq', None, None, None, (629, 504, 708, 524))],
    14: [((300, 46, 1240, 71), 'Resumen visual del Capitulo 8 — la arquitectura que lo cambió todo',
          'Resumen visual del Capítulo 12 — la arquitectura que lo cambió todo', CB, 'centro'),
         ((462, 493, 686, 515), 'x = LayerNorm(x + MHA(x))', 'x = x + MHA(LayerNorm(x))', FM, 'centro'),
         ((462, 703, 690, 727), 'x = LayerNorm(x + FFN(x))', 'x = x + FFN(LayerNorm(x))', FM, 'centro'),
         ((28, 900, 270, 922), 'x = LayerNorm(x + MHA(x))', 'x = x + MHA(LayerNorm(x))', FM, 'izq'),
         ((28, 931, 270, 953), 'x = LayerNorm(x + FFN(x))', 'x = x + FFN(LayerNorm(x))', FM, 'izq'),
         ((28, 610, 252, 629), 'Conexión residual: x + sublayer(x)', 'Pre-LN: x + sublayer(LayerNorm(x))', CB, 'izq', None, None, 262),
         ((1448, 455, 1500, 475), '117M', '124M', CB, 'centro'),
         ((1292, 530, 1334, 551), '2-4', '2', CB, 'centro'),
         ((1370, 530, 1420, 551), '64-128', '48', CB, 'centro'),
         ((1450, 530, 1492, 551), '<1M', '60.000', CB, 'centro', None, None, 48)],
    17: [((528, 48, 1108, 72), 'Resumen visual del Capítulo 11 — Redes Convolucionales',
          'Resumen visual del Apéndice G — Redes Convolucionales', CB, 'centro'),
         ((1293, 772, 1516, 790), 'Con los Capítulos 9 y 11 tienes las bases',
          'Con el Capítulo 12 y el Apéndice G tienes las bases', CB, 'izq', None, None, 285)],
    18: [((430, 45, 1121, 72), 'Resumen visual del Capítulo 11 bis — El ecosistema práctico',
          'Resumen visual del Capítulo 20 — El ecosistema práctico', CB, 'centro'),
         ((840, 163, 1084, 180), 'La similitud coseno del Capítulo 6 en acción',
          'La similitud coseno del Capítulo 8 en acción', CR, 'centro'),
         ((810, 212, 986, 228), 'En el Capítulo 6 aprendiste que:', 'En el Capítulo 8 aprendiste que:', CB, 'izq'),
         ((1268, 639, 1436, 657), 'Con Caps. 9, 11 y 11bis tienes', 'Con los Caps. 12 y 20 y el Ap. G tienes',
          CB, 'centro', None, None, 270)],
    19: [((403, 45, 1116, 72), 'Resumen visual del Capítulo 11 bis — Cuando CNN y Transformer se unen',
          'Resumen visual del Capítulo 20 — Cuando CNN y Transformer se unen', CB, 'centro'),
         ((680, 745, 766, 763), 'del Capítulo 8', 'del Capítulo 12', CR, 'centro'),
         ((1288, 193, 1430, 211), 'Cap. 3-4: LA NEURONA', 'Caps. 2-3: LA NEURONA', CB, 'centro'),
         ((1312, 280, 1392, 298), 'Cap. 5: MLP', 'Cap. 6: MLP', CB, 'centro'),
         ((1318, 375, 1407, 393), 'Cap. 11: CNN', 'Apéndice G: CNN', CB, 'centro', None, None, 140),
         ((1241, 481, 1512, 501), 'Caps. 6ter-7-8: RNN → ATTENTION → TRANSFORMER', 'Caps. 10-11-12: RNN → ATTENTION → TRANSFORMER', CB, 'izq', None, None, 262),
         ((1262, 588, 1453, 610), 'Cap. 11bis: MULTIMODAL', 'Cap. 20: MULTIMODAL', CB, 'centro'),
         ((22, 587, 212, 603), 'Cap. 6 → embeddings (vectores)', 'Cap. 8 → embeddings (vectores)', CR, 'izq'),
         ((22, 607, 272, 623), 'Cap. 8 → Transformer (procesa vectores)', 'Cap. 12 → Transformer (procesa vectores)', CR, 'izq'),
         ((22, 625, 245, 643), 'Cap. 11 → CNN (imagen → vectores)', 'Ap. G → CNN (imagen → vectores)', CR, 'izq')],
}

# Diagrama 13: las seis casillas «−∞» de la tabla causal muestran pesos tras el softmax → «0»
CELDAS13 = [(630, 625, 711, 656), (712, 625, 793, 656), (548, 626, 629, 657),
            (631, 658, 711, 691), (712, 658, 793, 690), (712, 692, 793, 725)]

if __name__ == '__main__':
    import sys
    solo = [int(a) for a in sys.argv[1:]] or list(CAMBIOS)
    for n in solo:
        im = Image.open(f'r{n}.png').convert('RGB')
        if n == 13:
            for c in CELDAS13:
                rojo = im.getpixel((c[0] + 4, c[1] + 4))
                celda(im, c, '0', CB, 22, (255, 255, 255), rojo)
        for ch in CAMBIOS[n]:
            caja, viejo, nuevo, fuente, alin = ch[:5]
            extra = list(ch[5:]) + [None] * (4 - len(ch[5:]))
            print(n, nuevo[:40], cambiar(im, caja, viejo, nuevo, fuente, alin, *extra))
        im.save(f'c{n}.png')
        im.rotate(90, expand=True).save(f'corr_image{n}.png')
