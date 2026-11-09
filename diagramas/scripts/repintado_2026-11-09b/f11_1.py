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

# 1. «Sin escalar: varianza(QKᵀ) ≈ d_k»
c = oscuro((472, 649, 725, 666)); f = fondo_de(im, (470, 648, 727, 668))
d.rectangle((470, 648, 727, 667), fill=f)
runs = [R('Sin escalar: varianza(QK', 'firac400', c), R('T', 'firac400', c, sup=True), R(') ≈ ', 'firac400', c),
        R('d', 'mathbi', c), R('k', 'mathbi', c, sub=True), R(' → softmax saturada', 'firac400', c)]
escribir(d, 472, 662.5, runs, ajustar(runs, 11.5, 252))

# 2. caja multi-head reescrita
f = fondo_de(im, (1244, 661, 1562, 734)); c = (12, 12, 12)
d.rectangle((1244, 661, 1562, 734), fill=f)
M = 'lmmono'
def linea(x, y, toks, s):
    for t in toks:
        if isinstance(t, str):
            x += escribir(d, x, y, [R(t, M, c)], s)
        else:
            kind = t[0]
            if kind == 'sub':
                x += escribir(d, x, y, [R(t[1], M, c, sub=True)], s)
            elif kind == 'sup':
                x += escribir(d, x, y, [R(t[1], M, c, sup=True)], s)
            elif kind == 'subsup':
                w1 = escribir(d, x, y, [R(t[1], M, c, sub=True, dy=1.5)], s)
                w2 = escribir(d, x, y, [R(t[2], M, c, sup=True, dy=-1)], s)
                x += max(w1, w2)
    return x
L1 = ['MultiHead(Q,K,V) = Concat(head', ('sub', '1'), ',…,head', ('sub', 'h'), ')·W', ('sup', 'O')]
L2 = ['head', ('sub', 'i'), ' = Attention(Q·W', ('subsup', 'i', 'Q'), ', K·W', ('subsup', 'i', 'K'), ', V·W', ('subsup', 'i', 'V'), ')']
L3 = ['       = softmax(Q', ('sub', 'i'), 'K', ('subsup', 'i', 'T'), '/√d', ('sub', 'k'), ' + M)·V', ('sub', 'i')]
def ancho_l(toks, s):
    import PIL.ImageDraw as D
    tmp = Image.new('RGB', (900, 60)); dd = D.Draw(tmp)
    global d
    d0 = d; d = dd; w = linea(0, 40, toks, s); d = d0; return w
s = 13
while max(ancho_l(L, s) for L in (L1, L2, L3)) > 306: s -= 0.25
for i, L in enumerate((L1, L2, L3)):
    linea(1250, 682 + i * 21, L, s)
print('size', s)
im.save(DST)
