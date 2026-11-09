# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pintar import *
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SRC = os.path.join(BASE, 'originales', 'Diagrama_G.1_antes_repintado_2026-11-08.png')
im = Image.open(SRC).convert('RGB')
d = ImageDraw.Draw(im)
tc = lambda b: tinta_de(im, b)
NEG = (15, 15, 15)
BL = (254, 254, 254)

# ------------------------------------------------------------------ panel 2: ejemplo de convolución
REG = (418, 184, 862, 397)
d.rectangle(REG, fill=BL)
AZ, VENT, NAR, NAR_F, TEAL, NAVY = (60, 95, 155), (15, 104, 106), (240, 125, 60), (255, 246, 238), (40, 120, 125), (20, 40, 90)
TIT_AZ, TIT_NAR, TIT_TEAL = (25, 75, 165), (226, 80, 20), (0, 84, 78)


def rejilla(x0, y0, cw, ch, valores, linea, fondo=None, size=15, color=NEG, fam='fira500'):
    filas, cols = len(valores), len(valores[0])
    if fondo:
        d.rectangle((x0, y0, x0 + cols * cw, y0 + filas * ch), fill=fondo)
    for i in range(filas + 1):
        d.line((x0, y0 + i * ch, x0 + cols * cw, y0 + i * ch), fill=linea, width=2)
    for j in range(cols + 1):
        d.line((x0 + j * cw, y0, x0 + j * cw, y0 + filas * ch), fill=linea, width=2)
    for i, fila in enumerate(valores):
        for j, v in enumerate(fila):
            escribir(d, x0 + j * cw + cw / 2, y0 + i * ch + ch / 2 + size * 0.36, [R(str(v), fam, color)], size, anchor='m')


IMG = [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]]
SOB = [['−1', '0', '+1'], ['−2', '0', '+2'], ['−1', '0', '+1']]
MAPA = [[4, 0, '−4', 0], [4, '−1', '−4', 0], [4, '−2', '−4', '−1'], [3, '−1', '−3', '−2']]
C = 26
ix, iy = 424, 216
escribir(d, ix + 3 * C, 210, [R('Imagen (6×6)', 'fira600', TIT_AZ)], 15, anchor='m')
rejilla(ix, iy, C, C, IMG, AZ, size=15)
d.rectangle((ix, iy, ix + 3 * C, iy + 3 * C), outline=VENT, width=3)
# flecha 1
fx = ix + 6 * C + 8
d.polygon([(fx, iy + 74), (fx + 12, iy + 74), (fx + 12, iy + 69), (fx + 22, iy + 78), (fx + 12, iy + 87), (fx + 12, iy + 82), (fx, iy + 82)], fill=NAVY)
# filtro
sx = fx + 30
escribir(d, sx + 1.5 * 29, 237, [R('Filtro Sobel-X', 'fira600', TIT_NAR)], 15, anchor='m')
rejilla(sx, 247, 29, 29, SOB, NAR, fondo=NAR_F, size=15)
escribir(d, sx + 1.5 * 29, 352, [R('detector de', 'fira500', TIT_NAR)], 13.5, anchor='m')
escribir(d, sx + 1.5 * 29, 368, [R('bordes verticales', 'fira500', TIT_NAR)], 13.5, anchor='m')
# flecha 2
gx = sx + 87 + 9
d.polygon([(gx, iy + 74), (gx + 12, iy + 74), (gx + 12, iy + 69), (gx + 22, iy + 78), (gx + 12, iy + 87), (gx + 12, iy + 82), (gx, iy + 82)], fill=(2, 104, 94))
# mapa
mx = gx + 30
escribir(d, mx + 2 * C, 211, [R('Mapa de', 'fira600', TIT_TEAL)], 13.5, anchor='m')
escribir(d, mx + 2 * C, 227, [R('características (4×4)', 'fira600', TIT_TEAL)], 12.5, anchor='m')
rejilla(mx, 236, C, C, MAPA, TEAL, size=14)
for k, t in enumerate(['sin relleno: 6×6 → 4×4', '(con relleno=1, como el', 'Programa 9A, sería 6×6)']):
    escribir(d, mx + 2 * C, 357 + k * 14, [R(t, 'fira500', TIT_TEAL)], 12, anchor='m')
assert mx + 4 * C < 860, mx

# ------------------------------------------------------------------ panel 2: cálculo en posición (1,1)
f = fondo_de(im, (440, 444, 690, 535))
viejo = [R('suma = Σ imagen[i,j] × filtro[i,j]', 'mono', NEG)]
bb = bbox_tinta(im, (436, 426, 700, 444), f)
s = 30.0
while bbox_render(viejo, s)[2] - bbox_render(viejo, s)[0] > bb[2] - bb[0]:
    s -= 0.25
d.rectangle((436, 445, 692, 535), fill=f)
lineas = [(474, 460, '= 0×(−1) + 0×0 + 1×(+1)'), (490, 482, '+ 0×(−2) + 0×0 + 1×(+2)'),
          (490, 504, '+ 0×(−1) + 0×0 + 1×(+1) = 4')]
s2 = min(s, ajustar([R(lineas[2][2], 'mono')], s, 688 - 490))
for x, y, l in lineas:
    escribir(d, x, y, [R(l, 'mono', NEG)], s2)
escribir(d, 440, 529, [R('resultado = 4 → borde vertical detectado', 'mono', NEG)], ajustar([R('resultado = 4 → borde vertical detectado', 'mono')], s2, 248))

# ------------------------------------------------------------------ panel 3: filtros clásicos
reemplazar(im, (900, 128, 1180, 154), (900, 128, 1180, 154), [R('QUÉ APRENDEN LOS FILTROS', 'fira700')],
           [R('LOS FILTROS CLÁSICOS (PROGRAMA 9A)', 'fira700')], align='m', color=(255, 255, 255),
           fondo=im.getpixel((905, 140)), maxw=300)
c = tc((891, 162, 1192, 175))
reemplazar(im, (885, 159, 1200, 178), (885, 159, 1200, 178),
           [R('La CNN descubrió detectores de bordes sin supervisión', 'fira500', c)],
           [R('Diseñados a mano; una CNN aprende detectores parecidos', 'fira500', c)], align='m')

tarjetas = [  # título viejo, nuevo, rejilla, (xs, ys), caja título, caption vieja, nueva, caja caption
    ('Filtro 1 — Detector de barra HORIZONTAL', 'Filtro 1 — Bordes HORIZONTALES (Sobel-Y)',
     [['−1', '−2', '−1'], ['0', '0', '0'], ['+1', '+2', '+1']], ([887, 917, 948, 979], [210, 237, 263, 289]),
     (888, 186, 1190, 207), 'activa fuerte cuando hay línea horizontal', 'activa en cambios de arriba a abajo', (962, 289, 1190, 304)),
    ('Filtro 2 — Detector de barra VERTICAL', 'Filtro 2 — Bordes VERTICALES (Sobel-X)',
     [['−1', '0', '+1'], ['−2', '0', '+2'], ['−1', '0', '+1']], ([887, 917, 948, 979], [343, 369, 395, 421]),
     (888, 320, 1190, 338), 'activa fuerte cuando hay columna brillante', 'activa en cambios de izquierda a derecha', (978, 423, 1200, 439)),
    ('Filtro 3 — Detector de ESQUINA', 'Filtro 3 — Detector de ESQUINAS (Laplaciano)',
     None, None, (888, 452, 1190, 470), None, None, None),
    ('Filtro 4 — SUAVIZADO (Gaussiano)', 'Filtro 4 — SUAVIZADO (Gaussiano 1/16)',
     [['1/16', '2/16', '1/16'], ['2/16', '4/16', '2/16'], ['1/16', '2/16', '1/16']], ([884, 917, 950, 984], [584, 607, 631, 655]),
     (888, 562, 1190, 583), None, None, None),
]
for tv, tn, rej, lin, ctit, cv, cn, ccap in tarjetas:
    col = tc(ctit)
    reemplazar(im, ctit, ctit, [R(tv, 'fira700', col)], [R(tn, 'fira700', col)], align='l', maxw=300,
               fondo=im.getpixel((1120, ctit[1] + 2)))
    if rej:
        xs, ys = lin
        for i in range(3):
            for j in range(3):
                caja = (xs[j] + 3, ys[i] + 3, xs[j + 1] - 2, ys[i + 1] - 2)
                f = fondo_de(im, caja)
                d.rectangle(caja, fill=f)
                v = rej[i][j]
                sz = 14 if len(v) <= 2 else 11.5
                escribir(d, (xs[j] + xs[j + 1]) / 2, (ys[i] + ys[i + 1]) / 2 + sz * 0.36, [R(v, 'fira600', NEG)], sz, anchor='m')
    if cv:
        col = tc(ccap)
        reemplazar(im, ccap, ccap, [R(cv, 'fira500', col)], [R(cn, 'fira500', col)], align='m', maxw=ccap[2] - ccap[0])

# tarjeta «¿Cómo los aprende la CNN?»
c = tc((975, 688, 1135, 707))
reemplazar(im, (925, 686, 1150, 708), (925, 686, 1150, 708), [R('¿Cómo los aprende la CNN?', 'fira700', c)],
           [R('¿Y qué aprende la CNN?', 'fira700', c)], align='m')
d.rectangle((914, 692, 926, 708), fill=fondo_de(im, (914, 692, 926, 708)))
c = tc((886, 743, 1164, 756))
reemplazar(im, (883, 741, 1180, 758), (883, 741, 1180, 758),
           [R('El backpropagation ajusta los 9 valores del filtro', 'fira500', c)],
           [R('El backpropagation ajusta los 9 valores de cada filtro', 'fira500', c)], align='l', maxw=292)
reemplazar(im, (883, 758, 1180, 775), (883, 758, 1180, 775),
           [R('hasta que maximizan la señal correcta.', 'fira500', c)],
           [R('hasta que separan las dos clases.', 'fira500', c)], align='l')
c = tc((893, 780, 1174, 796))
reemplazar(im, (885, 778, 1185, 798), (885, 778, 1185, 798),
           [R('La CNN redescubrió el filtro Sobel de 1968.', 'fira700', c)],
           [R('En redes grandes: detectores parecidos a Sobel.', 'fira700', c)], align='m', maxw=292)

# flecha «qué aprenden los filtros» → «los filtros clásicos»
fnav = im.getpixel((822, 490))
caja = (818, 492, 868, 565)
bb = bbox_tinta(im, caja, fnav, 200)
d.rectangle((bb[0] - 1, bb[1] - 1, bb[2] + 1, bb[3] + 1), fill=fnav)
cy = (bb[1] + bb[3]) / 2
for k, t in enumerate(['los', 'filtros', 'clásicos']):
    escribir(d, (bb[0] + bb[2]) / 2, cy - 12 + k * 15, [R(t, 'fira600', (255, 255, 255))], 12.5, anchor='m')

# ------------------------------------------------------------------ panel 1: predicción
fv = im.getpixel((92, 680))
reemplazar(im, (90, 672, 260, 690), (90, 672, 260, 690), [R('sigmoid → P(clase)', 'fira600')], [R('sigmoid → P(vertical)', 'fira600')],
           align='m', color=(255, 255, 255), fondo=fv)
reemplazar(im, (60, 689, 290, 707), (95, 689, 250, 707), [R('Ej: P(horizontal) = 0.97', 'fira600')],
           [R('Ej: barra vertical → P(vertical) = 1.0000', 'fira600')], align='m', color=(255, 255, 255), fondo=fv, maxw=225)

im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'Diagrama_G.1.png'))
print('ok')
