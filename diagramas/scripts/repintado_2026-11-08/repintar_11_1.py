# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pintar import *
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SRC = os.path.join(BASE, 'originales', 'Diagrama_11.1_antes_repintado_2026-11-08.png')
im = Image.open(SRC).convert('RGB')
d = ImageDraw.Draw(im)
NEG = (15, 15, 15)

# 1. tilde suelta sobre la C del título
d.rectangle((1295, 0, 1309, 8), fill=fondo_de(im, (1290, 0, 1315, 9)))
d.rectangle((1333, 41, 1349, 52), fill=fondo_de(im, (1330, 40, 1352, 53)))

# 2. matriz causal coherente con la bidireccional
celdas = [((852, 552, 898, 574), '0.42', '0.25'), ((902, 552, 948, 574), '0.58', '0.75'),
          ((852, 582, 898, 604), '0.32', '0.21'), ((902, 582, 948, 604), '0.29', '0.35'),
          ((954, 582, 998, 604), '0.39', '0.44')]
for caja, viejo, nuevo in celdas:
    reemplazar(im, caja, caja, [R(viejo, 'fira600', NEG)], [R(nuevo, 'fira600', NEG)], align='m',
               fondo=im.getpixel((caja[0] + 3, caja[1] + 3)))

# 3. viñeta del escalado
c = tinta_de(im, (470, 682, 720, 696))
reemplazar(im, (468, 679, 726, 699), (468, 679, 726, 699),
           [R('Con escalar: varianza ÷ ', 'firac600', c), R('d', 'mathbi', c), R('k', 'mathbi', c, sub=True),
            R('  →  gradientes estables', 'firac600', c)],
           [R('Con escalar: varianza ≈ 1  →  gradientes estables', 'firac600', c)], align='l')

# 4. fórmula de head_i
c = tinta_de(im, (1248, 688, 1550, 704))
f = fondo_de(im, (1244, 686, 1562, 706))
bb = bbox_tinta(im, (1244, 686, 1562, 706), f)
viejo = [R('head', 'mono', c), R('i', 'mono', c, sub=True), R(' = Attention(Q·W', 'mono', c), R('iQ', 'mono', c, sub=True),
         R(', K·W', 'mono', c), R('iK', 'mono', c, sub=True), R(', V·W', 'mono', c), R('iV', 'mono', c, sub=True), R(')', 'mono', c)]
s = 30.0
while s > 6 and bbox_render(viejo, s)[2] - bbox_render(viejo, s)[0] > bb[2] - bb[0]:
    s -= 0.25
rb = bbox_render(viejo, s)
base = bb[3] - rb[3] - 1
d.rectangle((1244, 686, 1562, 706), fill=f)
nuevo = [R('head', 'mono', c), R('i', 'mono', c, sub=True), R(' = Attention(Q·W', 'mono', c), R('i', 'mono', c, sub=True), R('Q', 'mono', c, sup=True),
         R(', K·W', 'mono', c), R('i', 'mono', c, sub=True), R('K', 'mono', c, sup=True),
         R(', V·W', 'mono', c), R('i', 'mono', c, sub=True), R('V', 'mono', c, sup=True), R(')', 'mono', c)]
escribir(d, bb[0] - rb[0], base + 1, nuevo, ajustar(nuevo, s, 1558 - bb[0]))

# 5. «Pesos causal» → «Pesos (causal)»
c = tinta_de(im, (1250, 270, 1293, 286))
reemplazar(im, (1247, 268, 1297, 288), (1247, 268, 1297, 288), [R('causal', 'fira700', c)], [R('(causal)', 'fira700', c)],
           align='m', maxw=48)

im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'Diagrama_11.1.png'))
print('ok')
