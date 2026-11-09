import sys; sys.path.insert(0, '.')
from pintar import *
import numpy as np
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); src = im.copy(); d = ImageDraw.Draw(im)
A = np.asarray(src).astype(int)
px = lambda x, y: tuple(int(v) for v in A[y, x])
def oscuro(box, n=15):
    a = A[box[1]:box[3], box[0]:box[2]].reshape(-1, 3)
    return tuple(int(v) for v in a[a.sum(1).argsort()[:n]].mean(0))
# 1. subrayado bajo la N de «Pool × N»
d.rectangle((490, 321, 497, 325), fill=px(486, 323))
# 2. CNN y ViT son alternativas: la flecha entre las cajas pasa a «o bien»
f = px(445, 357)
d.rectangle((448, 350, 463, 364), fill=f)
teal = oscuro((395, 262, 500, 275))
escribir(d, 455.5, 361, [R('o bien', 'fira600i', teal)], 10.5, anchor='m')
# 3. CLIP: «(similitudes ilustrativas)» dentro del panel, sin pisar el borde de la caja
FB = (253, 253, 253)
base = src.crop((845, 609, 1112, 624))
d.rectangle((879, 584, 1110, 625), fill=FB)
d.rectangle((845, 606, 1112, 625), fill=FB)
im.paste(base, (845, 613))
escribir(d, 1103, 595, [R('(similitudes ilustrativas)', 'fira500i', oscuro((995, 584, 1104, 592)))], 10, anchor='r')
escribir(d, 970.5, 607, [R('Conecta espacio visual y textual.', 'fira600', oscuro((883, 593, 1059, 602)))], 12, anchor='m')
# 4. fórmula inferior: «›» deformada → la misma flecha «→» de la línea
im.paste(src.crop((76, 925, 94, 937)), (156, 925))
im.save(DST); print('ok', teal)
