# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'repintado_2026-11-08'))
from pintar import *
from PIL import Image, ImageDraw
im = Image.open(sys.argv[1]).convert('RGB'); d = ImageDraw.Draw(im)
c = tinta_de(im, (584, 794, 609, 802))
for caja, v, n in [((580, 791, 614, 805), '4 000', '4000'), ((672, 791, 706, 805), '8 000', '8000')]:
    reemplazar(im, caja, caja, [R(v, 'fira500', c)], [R(n, 'fira500', c)], align='m')
c = tinta_de(im, (478, 820, 809, 831))
reemplazar(im, (474, 817, 815, 833), (474, 817, 815, 833),
           [R('Media en el propio texto de entrenamiento: el modelo lo memoriza (13.6)', 'fira500', c)],
           [R('Medida en el propio texto de entrenamiento: el modelo lo memoriza (13.6)', 'fira500', c)], align='m')
# muestras de la leyenda
AZ, RO, VE = (20, 30, 230), (200, 30, 30), (40, 120, 50)
for y, col, tipo in [(723, AZ, 'p'), (735, RO, 'd'), (747, VE, 'd')]:
    if tipo == 'p':
        d.line((588, y, 603, y), fill=col, width=2); d.ellipse((593, y - 2.5, 598, y + 2.5), fill=col)
    else:
        for x in (588, 595):
            d.line((x, y, x + 4, y), fill=col, width=2)
# rótulo del eje
capa = Image.new('RGBA', (60, 16), (0, 0, 0, 0)); dc = ImageDraw.Draw(capa)
escribir(dc, 30, 12, [R('Pérdida', 'fira500', (40, 40, 40))], 11, anchor='m')
capa = capa.rotate(90, expand=True)
im.paste(capa, (466, 715), capa)
im.save(sys.argv[2]); print('ok')
