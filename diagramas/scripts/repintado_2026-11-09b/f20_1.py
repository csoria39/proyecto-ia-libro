import sys; sys.path.insert(0, '.')
from pintar import *
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); d = ImageDraw.Draw(im)
GRIS = (40, 40, 40); GR2 = (60, 60, 60); FONDO = (253, 253, 253)
# cadenas del gráfico de vectores, con tildes (como en la tabla Top-3)
d.rectangle((898, 402, 1104, 415), fill=FONDO)
escribir(d, 900, 412, [R('pregunta: "devolución contrato"', 'mono_r', GRIS)], 10.5)
qs = ["'cláusula devolución 30 días'", "'artículo 3: garantías'", "'política de cambios'", "'historia de la empresa'"]
sq = min(ajustar([R(q, 'mono_r')], 9.5, 136) for q in qs)
y = 438
for q in qs:
    d.rectangle((960, y + 3, 1104, y + 15), fill=FONDO)
    escribir(d, 962, y + 12, [R(q, 'mono_r', GR2)], sq)
    y += 38
im.save(DST); print('ok', sq)
