import sys; sys.path.insert(0, '.')
from pintar import *
import numpy as np
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); d = ImageDraw.Draw(im)
A = np.asarray(im).astype(int)
def oscuro(box, n=15):
    a = A[box[1]:box[3], box[0]:box[2]].reshape(-1, 3)
    return tuple(int(v) for v in a[a.sum(1).argsort()[:n]].mean(0))
def claro(box, n=15):
    a = A[box[1]:box[3], box[0]:box[2]].reshape(-1, 3)
    return tuple(int(v) for v in a[(-a.sum(1)).argsort()[:n]].mean(0))

# 1. tabla CNN ↔ Transformer
F = 'firac500'
celdas = [((1506, 262, 1650, 279), 'Mismos W,Q,K,V por bloque',
           [('Mismas W', {}), ('Q', {'sub': True}), (', W', {}), ('K', {'sub': True}), (', W', {}), ('V', {'sub': True}), (' por bloque', {})]),
          ((1506, 282, 1650, 299), 'MHA → Add&Norm → FFN', [('LN → MHA → LN → FFN', {})]),
          ((1378, 322, 1502, 340), 'Si (spatial)', [('Sí (espacial)', {})]),
          ((1506, 322, 1650, 340), 'Si (por bloque)', [('Sí (por bloque)', {})])]
S = None
for box, viejo, nuevo in celdas:
    c = oscuro(box)
    medir = (box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1)
    s = reemplazar(im, box, medir, [R(viejo, F, c)], [R(t, F, c, **o) for t, o in nuevo], align='m', maxw=box[2] - box[0] - 6, size=S)
    if S is None: S = s[0]
    print(viejo, s)

# 2. «combina» → «combinan» (sujeto plural)
c = oscuro((1296, 716, 1522, 729))
reemplazar(im, (1293, 714, 1530, 730), (1295, 715, 1523, 729), [R('GPT-4V, Gemini, LLaMA-3.2V combina:', 'fira500', c)],
           [R('GPT-4V, Gemini, LLaMA-3.2V combinan:', 'fira500', c)], align='l')

# 3. lema: CNN y Transformer comparten receta, no son lo mismo
reg = A[651:681, 1327:1552].reshape(-1, 3); verdes = reg[(reg[:, 1] - reg[:, 0] > 40) & (reg.sum(1) < 400)]
c = tuple(int(v) for v in verdes[verdes.sum(1).argsort()[:30]].mean(0)); f = fondo_de(im, (1300, 649, 1640, 682))
d.rectangle((1300, 650, 1640, 681), fill=f)
for i, l in enumerate(('Misma receta, distinto dominio:', 'imágenes (CNN) y secuencias (Transformer)')):
    escribir(d, 1466, 662 + i * 14.5, [R(l, 'fira600', c)], ajustar([R('imágenes (CNN) y secuencias (Transformer)', 'fira600')], 15, 330), anchor='m')

# 4. fórmula de parámetros: dos capas densas (100 → 16 → 1), como el Programa 9B
fondo = tuple(int(v) for v in np.median(A[880:900, 30:420].reshape(-1, 3)[A[880:900, 30:420].reshape(-1, 3).sum(1) < 250], axis=0))
blanco = claro((32, 882, 415, 898))
d.rectangle((28, 879, 425, 901), fill=fondo)
runs = [R('params = F·(f', 'mono', blanco), R('2', 'mono', blanco, sup=True), R('+1) + 16·(n_flat+1) + (16+1)', 'mono', blanco)]
sz = ajustar(runs, 15.2, 390)
escribir(d, 32, 895, runs, sz); escribir(d, 32.45, 895, runs, sz)
im.save(DST); print('ok', fondo, blanco)
