# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pintar import *
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SRC = os.path.join(BASE, 'originales', 'Diagrama_20.2_antes_repintado_2026-11-08.png')
im = Image.open(SRC).convert('RGB')
d = ImageDraw.Draw(im)
tc = lambda b: tinta_de(im, b)

# 1. la caja de los Caps. 10-12 pasa a cubrir 10-16 (Nano-LLM y GPT-2 real)
c = tc((1247, 482, 1510, 496))
reemplazar(im, (1246, 480, 1512, 499), (1246, 480, 1512, 499),
           [R('Caps. 10-11-12: RNN → ATTENTION → TRANSFORMER', 'fira700', c)],
           [R('Caps. 10-16: RNN → TRANSFORMER → NANO-LLM → GPT-2', 'fira700', c)], align='m', maxw=262)
c = tc((1252, 507, 1470, 523))
reemplazar(im, (1249, 501, 1505, 519), (1249, 501, 1505, 519),
           [R('Memoria secuencial → atención global', 'fira500', c)],
           [R('Atención global → tu LLM y GPT-2 con tu código', 'fira500', c)], align='l', maxw=252)

# 2. CLIP: similitudes ilustrativas
escribir(d, 1104, 592, [R('(similitudes ilustrativas)', 'fira500i', (90, 90, 90))], 10, anchor='r')

# 3. tarjeta final
f = im.getpixel((1225, 860))
d.rectangle((1228, 853, 1494, 893), fill=f)
l2 = [R('Con lo que sabes aquí, tienes las bases', 'fira500', (255, 255, 255))]
l3 = [R('para leer los artículos de IA de la última década.', 'fira500', (255, 255, 255))]
s = ajustar(l3, 17, 255)
escribir(d, 1360, 869, l2, s, anchor='m')
escribir(d, 1360, 887, l3, s, anchor='m')

# 4. «Claude Visión» → «Claude»; subrayado suelto bajo la N
c = tc((1256, 628, 1446, 642))
reemplazar(im, (1252, 625, 1452, 645), (1252, 625, 1452, 645), [R('= GPT-4V, Gemini, Claude Visión', 'fira500', c)],
           [R('= GPT-4V, Gemini, Claude', 'fira500', c)], align='m')
d.rectangle((481, 324, 495, 328), fill=fondo_de(im, (478, 322, 498, 328)))

im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'Diagrama_20.2.png'))
print('ok')
