# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pintar import *
from PIL import Image, ImageDraw
import math

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SRC = os.path.join(BASE, 'originales', 'Diagrama_10.2_antes_repintado_2026-11-08.png')
im = Image.open(SRC).convert('RGB')


def M(t, col, sub=None, fam='mathbi'):
    out = [R(t, fam, col)]
    if sub:
        out.append(R(sub, fam, col, sub=True))
    return out


def col_de(box):
    return tinta_de(im, box)


# 1. subtítulo
c = col_de((440, 68, 1090, 95))
reemplazar(im, (432, 67, 1098, 94), (432, 67, 1098, 94),
           [R('Cuatro puertas · Estado de celda ', 'fira700', c)] + M('C', c, 't') + [R(' · Memoria a largo y corto plazo', 'fira700', c)],
           [R('Tres puertas y un candidato · Estado de celda ', 'fira700', c)] + M('C', c, 't') + [R(' · Memoria a largo y corto plazo', 'fira700', c)],
           align='m', maxw=760)
# 2. etiqueta de flecha
reemplazar(im, (325, 96, 390, 111), (325, 96, 390, 111), [R('las cuatro', 'fira600')], [R('las tres', 'fira600')],
           align='m', color=(255, 255, 255), fondo=im.getpixel((323, 113)))
# 3. cabecera del panel 2
reemplazar(im, (441, 112, 752, 133), (441, 112, 752, 133), [R('LAS CUATRO PUERTAS DE LA LSTM', 'fira600')],
           [R('TRES PUERTAS Y UN CANDIDATO', 'fira600')], align='l', color=(255, 255, 255), fondo=im.getpixel((735, 122)))
# 4. subcabecera
c = col_de((410, 142, 745, 159))
reemplazar(im, (405, 141, 750, 160), (405, 141, 750, 160),
           [R('Cada puerta es una red pequeña con activación σ (sigmoid)', 'fira500', c)],
           [R('Las puertas usan σ (sigmoid); el candidato ', 'fira500', c)] + M('g', c, 't') + [R(' usa tanh', 'fira500', c)],
           align='m', maxw=335)
# 5. épocas → pasos
c = col_de((26, 319, 334, 337))
reemplazar(im, (26, 319, 336, 337), (26, 319, 336, 337),
           [R('→ Vanishing gradient: gradiente → 0 en pocas épocas', 'fira600', c)],
           [R('→ Vanishing gradient: gradiente → 0 en pocos pasos', 'fira600', c)], align='l')
# 6. ② título
c = col_de((418, 368, 672, 390))
reemplazar(im, (416, 367, 690, 391), (416, 367, 690, 391), [R('PUERTA DE ENTRADA (Input Gate)', 'fira700', c)],
           [R('PUERTA DE ENTRADA Y CANDIDATO', 'fira700', c)], align='l')
# 7. tira amarilla
c = col_de((385, 688, 715, 707))
reemplazar(im, (378, 687, 722, 708), (378, 687, 722, 708),
           [R('Solo SUMAS en ', 'fira600', c)] + M('C', c, 't') + [R(' → gradiente fluye sin atenuar', 'fira600', c)],
           M('C', c, 't−1') + [R(' solo se multiplica por ', 'fira600', c)] + M('f', c, 't') + [R(' → el gradiente decae menos', 'fira600', c)],
           align='m', maxw=338)
# 8. «Las 4 puertas»
c = col_de((32, 721, 190, 739))
reemplazar(im, (30, 720, 195, 740), (30, 720, 195, 740), [R('Las 4 puertas deciden:', 'fira600', c)],
           [R('Las 3 puertas deciden:', 'fira600', c)], align='l')
# 9. tarjeta verde, dos líneas
c = col_de((32, 803, 300, 837))
s, s2, b1 = reemplazar(im, (30, 801, 316, 820), (30, 801, 316, 820),
                       M('C', c, 't') + [R(' se actualiza con SUMAS, no productos.', 'fira600', c)],
                       M('C', c, 't−1') + [R(' solo se multiplica por ', 'fira600', c)] + M('f', c, 't') + [R(', sin pasar', 'fira600', c)],
                       align='l', maxw=282)
reemplazar(im, (30, 820, 316, 840), (30, 820, 316, 840),
           [R('El gradiente fluye por ', 'fira600', c)] + M('C', c, 't') + [R(' sin atenuarse.', 'fira600', c)],
           [R('por pesos ni tanh: el gradiente decae menos.', 'fira600', c)], align='l', maxw=282, size=s2)
# 10. comparación del panel 3
c = col_de((962, 820, 1152, 838))
reemplazar(im, (960, 819, 1154, 839), (960, 819, 1154, 839),
           [R('Gradiente: fluye libremente por ', 'fira600', c)] + M('C', c, 't'),
           [R('Gradiente: fluye por ', 'fira600', c)] + M('C', c, 't') + [R(' (solo × ', 'fira600', c)] + M('f', c, 't') + [R(')', 'fira600', c)],
           align='l', maxw=188)
# 11. LSTM estándar
c = col_de((1309, 391, 1503, 409))
reemplazar(im, (1307, 390, 1506, 410), (1307, 390, 1506, 410), [R('4 puertas, la arquitectura original.', 'fira600', c)],
           [R('3 puertas y un candidato.', 'fira600', c)], align='l')
# 12. Peephole
c = col_de((1309, 477, 1418, 495))
reemplazar(im, (1307, 476, 1460, 496), (1307, 476, 1418, 496), M('C', c, 't') + [R(' directamente.', 'fira600', c)],
           M('C', c, 't−1') + [R(' directamente.', 'fira600', c)], align='l')

# 13. curva de la RNN (Programa RNN-2, «W_hh moderada»)
X0, X1 = 1240.0, 1347.0          # pasos 0 y 20
Y1, DEC = 205.5, 39.5            # y de 1 y píxeles por década
datos = [(0, 1.0), (1, 0.454545), (2, 0.209258), (5, 0.018165), (10, 0.000482), (20, 1e-12)]


def y_de(v):
    return Y1 + (-math.log10(max(v, 1e-3))) * DEC


def v_de(p):
    """Interpolación monótona (PCHIP) en escala logarítmica entre las medidas de RNN-2."""
    xs = [a for a, _ in datos]; ys = [math.log10(v) for _, v in datos]
    n = len(xs); h = [xs[i + 1] - xs[i] for i in range(n - 1)]; dl = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [dl[0]] + [0.0] * (n - 2) + [dl[-1]]
    for i in range(1, n - 1):
        if dl[i - 1] * dl[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i])
    for i in range(n - 1):
        if xs[i] <= p <= xs[i + 1]:
            t = (p - xs[i]) / h[i]
            h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
            return 10 ** (h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1])


INT = (1241, 207, 1346, 323)
E = 4
w, h = (INT[2] - INT[0]) * E, (INT[3] - INT[1]) * E
capa = Image.new('RGB', (w, h), (255, 255, 255))
dc = ImageDraw.Draw(capa)
pts = []
for i in range(0, 2001):
    p = 20 * i / 2000
    x = (X0 + (X1 - X0) * p / 20 - INT[0]) * E
    y = (y_de(v_de(p)) - INT[1]) * E
    pts.append((x, y))
# relleno degradado bajo la curva
for i in range(len(pts) - 1):
    x, y = pts[i]
    for yy in range(int(y), h):
        t = (yy - y) / max(1, h - y)
        r = (253, int(232 - 40 * t), int(230 - 40 * t))
        dc.point((x, yy), fill=r)
for p in (2, 5, 9):              # guías punteadas
    x = (X0 + (X1 - X0) * p / 20 - INT[0]) * E
    y = (y_de(v_de(p)) - INT[1]) * E
    for yy in range(int(y), h, 12):
        dc.line((x, yy, x, yy + 6), fill=(240, 120, 110), width=3)
dc.line(pts, fill=(228, 20, 20), width=10, joint='curve')
capa = capa.resize((INT[2] - INT[0], INT[3] - INT[1]), Image.LANCZOS)
im.paste(capa, INT[:2])
d = ImageDraw.Draw(im)
rojo = (225, 25, 25)
escribir(d, 1322, 275, [R('zona', 'fira600', rojo)], 15, anchor='m')
escribir(d, 1322, 293, [R('ciega', 'fira600', rojo)], 15, anchor='m')

im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'Diagrama_10.2.png'))
print('ok')
