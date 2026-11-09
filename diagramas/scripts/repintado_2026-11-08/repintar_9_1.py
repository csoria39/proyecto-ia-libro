# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pintar import *
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SRC = os.path.join(BASE, 'originales', 'Diagrama_9.1_antes_repintado_2026-11-08.png')
im = Image.open(SRC).convert('RGB')
d = ImageDraw.Draw(im)
tc = lambda b: tinta_de(im, b)
NEG = (20, 20, 20)

# 1. «Resultado tras N fusiones» y sus filas
c = tc((548, 707, 732, 721))
reemplazar(im, (545, 705, 735, 723), (545, 705, 735, 723), [R('Resultado tras N fusiones:', 'fira700', c)],
           [R('Resultado tras 5 fusiones:', 'fira700', c)], align='l')
ROJO = tc((560, 729, 585, 744))
fondo = fondo_de(im, (552, 745, 830, 784))
d.rectangle((552, 745, 832, 784), fill=fondo)
d.rectangle((552, 727, 690, 744), fill=fondo)        # se reescribe también la fila de 'bajo'
s = 15.5
filas = [(742, [R('[', 'monob', NEG), R('bajo', 'monob', ROJO), R('</w>]', 'monob', NEG)], '← 1 token'),
         (760, [R('[', 'monob', NEG), R('baj', 'monob', ROJO), R('][a][r][</w>]', 'monob', NEG)], '← 4 tokens'),
         (778, [R('[', 'monob', NEG), R('caj', 'monob', ROJO), R('][a][</w>]', 'monob', NEG)], '← 3 tokens')]
w = max(ancho(f[1], s) for f in filas)
while w > 134:
    s -= 0.25; w = max(ancho(f[1], s) for f in filas)
sf = 30.0
bb = bbox_tinta(im, (690, 728, 775, 745), fondo)
while sf > 6 and bbox_render([R('← 1 token', 'monob')], sf)[2] - bbox_render([R('← 1 token', 'monob')], sf)[0] > bb[2] - bb[0]:
    sf -= 0.25
for y, runs, t in filas:
    escribir(d, 556, y, runs, s)
    if t != '← 1 token':
        escribir(d, bb[0] - bbox_render([R(t, 'monob')], sf)[0], y, [R(t, 'monob', NEG)], sf)

# 2. fertilidad (tokenizador de GPT-2)
c = tc((920, 565, 1180, 580))
reemplazar(im, (915, 563, 1185, 583), (915, 563, 1185, 583), [R('Los LLMs se entrenaron mayoritariamente en inglés', 'fira500', c)],
           [R('Tokenizador de GPT-2 (entrenado casi todo en inglés)', 'fira500', c)], align='m')
for caja, viejo, nuevo in [((1054, 624, 1127, 644), '→ 2-3 tokens', '→ 3 tokens'),
                           ((1054, 654, 1127, 672), '→ 5 tokens', '→ 9 tokens'),
                           ((1054, 684, 1127, 702), '→ 6-8 tokens', '→ 5 tokens')]:
    reemplazar(im, caja, caja, [R(viejo, 'fira500', NEG)], [R(nuevo, 'fira500', NEG)], align='l')
reemplazar(im, (948, 683, 1012, 702), (948, 683, 1012, 702), [R('compuestos', 'fira500', NEG)],
           [R("'evlerinizden'", 'fira500', NEG)], align='l', maxw=95)
POR_TOKEN = 10.5
for (y0, y1), n, col in [((596, 612), 1, (6, 104, 6)), ((625, 642), 3, (248, 104, 7)),
                         ((655, 670), 9, (200, 8, 6)), ((685, 700), 5, (200, 8, 6))]:
    f = fondo_de(im, (1133, y0 - 2, 1238, y1 + 2))
    d.rectangle((1133, y0 - 1, 1238, y1 + 1), fill=f)
    d.rectangle((1135, y0, 1135 + round(n * POR_TOKEN), y1), fill=col)
c = tc((910, 716, 1195, 732))
reemplazar(im, (905, 714, 1205, 734), (905, 714, 1205, 734),
           [R("Inglés es ~30% más 'barato' que español en tokens", 'fira500i', c)],
           [R('La misma frase: 12 tokens en inglés, 24 en español (Ej. 16.1)', 'fira500i', c)], align='m', maxw=296)

# 3. OOV con BPE (dos líneas)
c = tc((1066, 448, 1230, 462))
f = fondo_de(im, (1060, 445, 1236, 476))
d.rectangle((1060, 445, 1236, 476), fill=f)
l1 = [R("['super','cal','if','rag',", 'monob', c)]
l2 = [R("'il','istic','o']", 'monob', c)]
s3 = ajustar(l1, 13, 172)
escribir(d, 1066, 457, l1, s3)
escribir(d, 1066, 471, l2, s3)

# 4. vocabularios
c = tc((1096, 254, 1232, 266))
reemplazar(im, (1092, 252, 1236, 268), (1092, 252, 1236, 268), [R('★ GPT-4 usa 100.257', 'fira600', c)],
           [R('★ GPT-4 usa 100.277', 'fira600', c)], align='m')
for caja, viejo, nuevo, al in [((1282, 239, 1356, 258), 'GPT-3/4', 'GPT-4', 'l'),
                               ((1536, 239, 1600, 258), '100.257', '100.277', 'm'),
                               ((1282, 269, 1356, 288), 'LLaMA 2/3', 'LLaMA 2', 'l'),
                               ((1272, 329, 1356, 349), 'T5/mT5', 'T5', 'l')]:
    reemplazar(im, caja, caja, [R(viejo, 'fira500', NEG)], [R(nuevo, 'fira500', NEG)], align=al)

im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'Diagrama_9.1.png'))
print('ok')
