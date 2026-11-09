# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pintar import *
from PIL import Image, ImageDraw
import math

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SRC = os.path.join(BASE, 'originales', 'Diagrama_7.3_antes_repintado_2026-11-08.png')
im = Image.open(SRC).convert('RGB')
d = ImageDraw.Draw(im)
tc = lambda b: tinta_de(im, b)


def M(t, col, sub=None, fam='mathbi'):
    return [R(t, fam, col)] + ([R(sub, fam, col, sub=True)] if sub else [])


# 1. subtítulo
c = tc((335, 69, 1280, 87))
reemplazar(im, (325, 66, 1290, 90), (325, 66, 1290, 90),
           [R('Entrenamiento: neuronas desactivadas aleatoriamente  |  Inferencia: todas activas, pesos escalados', 'fira700', c)],
           [R('Entrenamiento: neuronas desactivadas aleatoriamente  |  Inferencia: todas activas, sin escalar', 'fira700', c)],
           align='m')

# 2. tarjeta «Corrección de escala»: línea 4, solución y fórmula
c = tc((882, 455, 1110, 466))
reemplazar(im, (878, 470, 1172, 489), (878, 470, 1172, 489),
           [R('Sin corrección, la salida sería ', 'fira500', c), R('p', 'mathi', c), R(' veces mayor.', 'fira500', c)],
           [R('Sin corregir, la salida sería 1/(1−p) veces mayor.', 'fira500', c)], align='l', maxw=290)
c = tc((886, 505, 1165, 520))
reemplazar(im, (880, 502, 1172, 523), (880, 502, 1172, 523),
           [R('SOLUCIÓN: multiplicar cada peso por (1−p)', 'fira600', c)],
           [R('SOLUCIÓN (dropout invertido): escalar AL ENTRENAR', 'fira600', c)], align='l', maxw=288)
# interior del recuadro azul
fondo = fondo_de(im, (902, 541, 1168, 586))
azul = tc((905, 545, 1165, 585))
d.rectangle((880, 535, 1184, 585), fill=fondo)
s = 17
l1 = [R('h', 'mathbi', azul), R('train', 'mono', azul, sub=True), R(' = h · máscara / (1 − p)', 'mono', azul)]
l2 = [R('p = 0.5: activas × 2 · inferencia: h sin tocar', 'mono', azul)]
s1 = ajustar(l1, 19, 290); s2 = ajustar(l2, 19, 290)
escribir(d, 1032, 557, l1, min(s1, s2), anchor='m')
escribir(d, 1032, 579, l2, min(s1, s2), anchor='m')
c = tc((892, 600, 1175, 613))
reemplazar(im, (886, 597, 1180, 615), (886, 597, 1180, 615),
           [R('Implementado automáticamente en PyTorch/Keras', 'fira500', c)],
           [R('Así lo hacen nn.Dropout (PyTorch) y el Programa R-3', 'fira500', c)], align='m')

# 3. tabla «Comportamiento por fase»: fila Pesos
c = tc((873, 716, 905, 729))
reemplazar(im, (868, 712, 945, 732), (868, 712, 945, 732), [R('Pesos', 'fira500', c)], [R('Escalado', 'fira500', c)], align='l', size=13.5)
reemplazar(im, (962, 712, 1088, 732), (962, 712, 1088, 732), [R('sin escalar', 'fira500', c)],
           [R('activas × 1/(1−p)', 'fira500', c)], align='m', maxw=118, size=13.5, dy=3)
reemplazar(im, (1094, 712, 1176, 732), (1094, 712, 1176, 732), [R('× (1−p)', 'fira500', c)],
           [R('ninguno', 'fira500', c)], align='m', size=13.5)

# 4. leyenda inferior
reemplazar(im, (1440, 921, 1600, 939), (1440, 921, 1600, 939), [R('= Inferencia / pesos escalados', 'firac500')],
           [R('= Inferencia / sin escalar', 'firac500')], align='l', color=(255, 255, 255),
           fondo=im.getpixel((1602, 912)))

# 5. etiquetas de las neuronas en los tres pasos
def nodo(cx, cy, r, texto, sub, color_txt, fam='mathbi'):
    relleno = im.getpixel((int(cx) + r - 3, int(cy)))
    E = 4
    capa = Image.new('RGB', ((2 * r) * E, (2 * r) * E), relleno)
    m = Image.new('L', capa.size, 0)
    ImageDraw.Draw(m).ellipse((0, 0, 2 * r * E - 1, 2 * r * E - 1), fill=255)
    m = m.resize((2 * r, 2 * r), Image.LANCZOS)
    im.paste(capa.resize((2 * r, 2 * r)), (int(cx - r), int(cy - r)), m)
    escribir(d, cx, cy + 5, [R(texto, fam, color_txt), R(sub, fam, color_txt, sub=True)], 17, anchor='m')


negro, blanco = (20, 20, 20), (255, 255, 255)
nodo(416.5, 371, 12, 'h', '3', blanco)        # Paso 1: la tercera oculta es h3
for y, k in ((284, '1'), (339, '2'), (397.5, '3')):
    nodo(525, y, 12, 'x', k, negro, 'fira500i')            # Paso 2: entradas x1 x2 x3
nodo(594, 422.5, 12, 'h', '4', blanco)         # Paso 2: cuarta oculta
nodo(700, 285, 12, 'x', '1', negro, 'fira500i')            # Paso 3: primera entrada
nodo(769, 267.5, 12, 'h', '1', blanco)         # Paso 3: primera oculta
nodo(769, 320, 12, 'h', '2', blanco)           # Paso 3: segunda oculta

# 6. resultados (Programa R-3, 256 neuronas): ejes de 0 a 0.2
AZ, NA = (0, 36, 168), (252, 54, 6)


def grafica(x0e, x100, curvas, leyenda_box):
    leyenda = im.crop(leyenda_box)
    INT = (x0e, 208, x100 + 4, 310)
    fondo = fondo_de(im, (INT[0], INT[1], INT[2], INT[1] + 4))
    E = 4
    W, H = (INT[2] - INT[0]) * E, (INT[3] - INT[1]) * E
    capa = Image.new('RGB', (W, H), fondo)
    dc = ImageDraw.Draw(capa)
    for f, col in curvas:
        pts = []
        for i in range(0, 401):
            e = 100 * i / 400
            v = f(e)
            if v > 0.205:
                continue
            pts.append(((x0e + e * (x100 - x0e) / 100 - INT[0]) * E, (309.5 - min(v, 0.215) / 0.2 * 91 - INT[1]) * E))
        dc.line(pts, fill=col, width=9, joint='curve')
    im.paste(capa.resize((INT[2] - INT[0], INT[3] - INT[1]), Image.LANCZOS), INT[:2])
    im.paste(leyenda, leyenda_box[:2])
    d.line((x0e - 1, 212, x0e - 1, 310), fill=(46, 48, 61), width=1)
    d.line((x0e - 1, 310, x100 + 3, 310), fill=(60, 60, 75), width=1)


grafica(1275, 1396, [(lambda e: 0.0324 + 0.6 * math.exp(-e / 6), AZ),
                     (lambda e: 0.0745 + 0.6 * math.exp(-e / 5) + 0.0166 * (e / 100) ** 1.5, NA)], (1303, 222, 1378, 258))
grafica(1469, 1589, [(lambda e: 0.0554 + 0.6 * math.exp(-e / 8), AZ),
                     (lambda e: 0.0856 + 0.6 * math.exp(-e / 7) + 0.002 * (e / 100), NA)], (1512, 222, 1582, 258))
# etiquetas del eje: 1.0 → 0.2, 0.5 → 0.1
for x0, x1 in ((1248, 1273), (1442, 1467)):
    reemplazar(im, (x0, 211, x1, 226), (x0, 211, x1, 226), [R('1.0', 'fira500')], [R('0.2', 'fira500')], align='r', color=(0, 0, 0), dx=-2)
    reemplazar(im, (x0, 257, x1, 272), (x0, 257, x1, 272), [R('0.5', 'fira500')], [R('0.1', 'fira500')], align='r', color=(0, 0, 0), dx=-2)
# títulos y cifras
c = tc((1468, 186, 1597, 203))
reemplazar(im, (1464, 184, 1600, 205), (1464, 184, 1600, 205), [R('Con Dropout (p=0.5)', 'fira600', c)],
           [R('Con Dropout (p=0.3)', 'fira600', c)], align='l')
c = tc((1266, 354, 1377, 369))
reemplazar(im, (1250, 351, 1395, 372), (1264, 351, 1380, 372), [R('Overfitting severo', 'fira600', c)],
           [R('Overfitting (val = 2.8 × train)', 'fira600', c)], align='m', maxw=140)
c = tc((1454, 354, 1582, 370))
reemplazar(im, (1440, 351, 1596, 372), (1452, 351, 1584, 372), [R('Buena generalización', 'fira600', c)],
           [R('Mejor validación (1.6 × train)', 'fira600', c)], align='m', maxw=148)
for caja in ((1372, 394, 1396, 415), (1569, 394, 1593, 415)):      # desplazar ✗ y ✓ 9 px
    marca = im.crop(caja); f = fondo_de(im, caja)
    d.rectangle(caja, fill=f); im.paste(marca, (caja[0] + 9, caja[1]))
for (x0, y0, x1, y1), viejo, nuevo in [((1336, 375, 1371, 393), '0.02', '0.032'), ((1336, 396, 1371, 414), '0.89', '0.091'),
                                         ((1531, 375, 1568, 393), '0.15', '0.055'), ((1531, 396, 1568, 414), '0.20', '0.088')]:
    reemplazar(im, (x0, y0, x1, y1), (x0, y0, x1, y1), [R(viejo, 'fira500')], [R(nuevo, 'fira500')], align='l', color=(0, 0, 0))

im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'Diagrama_7.3.png'))
print('ok')
