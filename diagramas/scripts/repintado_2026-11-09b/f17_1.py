import sys, math; sys.path.insert(0, '.')
from pintar import *
import numpy as np
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); src = im.copy(); d = ImageDraw.Draw(im)
A = np.asarray(src).astype(int)
sat = A.max(2) - A.min(2)
# escala log del eje: 10^4 en y = 489.5, 28.3 px por década (ticks medidos 10^4..10^13)
Y = lambda v: 489.5 - (math.log10(v) - 4) * 28.3
barras = [((95, 130), 60_000, '60 000'), ((140, 185), 124e6, '124 M'), ((197, 235), 1.5e9, '1500 M'),
          ((247, 290), 7e9, '7000 M'), ((300, 345), 175e9, '175 000 M'), ((357, 400), 671e9, '671 000 M')]
recortes = []
for (xa, xb), v, t in barras:
    xc = (xa + xb) // 2
    cols = [x for x in range(xa - 8, xb + 8) if sat[480, x] > 60]
    x0, x1 = min(cols), max(cols)
    ys = [y for y in range(300, 490) if sat[y, xc] > 60 and all(sat[yy, xc] > 60 for yy in range(y, 486))]
    top = min(ys); bot = max(y for y in range(470, 492) if sat[y, xc] > 60)
    recortes.append((x0, x1, top, bot, src.crop((x0, top, x1 + 1, bot + 1)), v, t))
    print(t, x0, x1, top, bot, '->', round(Y(v), 1))
FO = (255, 254, 254)
d.rectangle((83, 266, 410, 487), fill=FO)
im.paste(src.crop((399, 425, 411, 488)), (399, 425))   # borde de la flecha «qué cambia con la escala»
NEG = (25, 25, 25); BL = (255, 255, 255)
for x0, x1, top, bot, crop, v, t in recortes:
    nt = int(round(Y(v)))
    im.paste(crop.resize((crop.width, bot + 1 - nt), Image.BICUBIC), (x0, nt))
    xc = (x0 + x1) / 2
    if v < 1e11:
        escribir(d, xc, nt - 5, [R(t, 'fira700', NEG)], 12.5, anchor='m')
    else:                                   # barras altas: cifra dentro, en vertical
        tt = Image.new('RGBA', (90, 20), (0, 0, 0, 0)); dt = ImageDraw.Draw(tt)
        escribir(dt, 45, 15, [R(t, 'fira700', BL)], 13, anchor='m')
        tt = tt.rotate(90, expand=True)
        im.paste(tt, (int(xc - tt.width / 2), nt + 6), tt)
# flecha ×10 millones: vertical, del tope del Nano-LLM al de DeepSeek-V3
ORO = (178, 128, 20)
xN = (recortes[0][0] + recortes[0][1]) / 2
yN, yD = Y(60_000), Y(671e9)
E = 4; X0, Y0 = 84, 262
capa = Image.new('RGBA', ((410 - X0) * E, (488 - Y0) * E), (0, 0, 0, 0)); dc = ImageDraw.Draw(capa)
P = lambda x, y: ((x - X0) * E, (y - Y0) * E)
# guías punteadas a la altura de los dos extremos
for y, xa, xb in ((yD, xN - 8, recortes[5][0] - 3), ):
    x = xa
    while x < xb:
        dc.line((P(x, y), P(min(x + 4, xb), y)), fill=ORO + (220,), width=int(1.3 * E)); x += 7
dc.line((P(xN, yN - 22), P(xN, yD + 9)), fill=ORO + (255,), width=int(2.6 * E))
for yy, s in ((yD, 1), ):
    dc.polygon([P(xN, yy), P(xN - 6, yy + 11), P(xN + 6, yy + 11)], fill=ORO + (255,))
dc.polygon([P(xN, yN - 18), P(xN - 6, yN - 29), P(xN + 6, yN - 29)], fill=ORO + (255,))
capa = capa.resize(((410 - X0), (488 - Y0)), Image.LANCZOS); im.paste(capa, (X0, Y0), capa)
escribir(d, 134, 291, [R('×10 millones', 'fira700', ORO)], 17)
escribir(d, 134, 308, [R('(7 órdenes de magnitud)', 'fira600', ORO)], 11.5)
im.save(DST); print('ok')

# rótulo «Resultado» del gráfico de la derecha: estaba deformado y montado sobre la punta de la flecha
im = Image.open(DST).convert('RGB'); a = np.array(im)
FOND = np.array([255, 255, 255])
for y in range(394, 464):
    for x in range(455, 469):
        diag = 428 + (x - 436) * 40 / 34
        a[y, x] = FOND
im = Image.fromarray(a); d = ImageDraw.Draw(im)
# borde superior de la flecha, redibujado en el tramo borrado
capa = Image.new('L', (40 * 4, 40 * 4), 0)
ImageDraw.Draw(capa).line(((450 - 440) * 4, (444.5 - 435) * 4, (470.5 - 440) * 4, (468.6 - 435) * 4), fill=255, width=10)
capa = capa.resize((40, 40), Image.LANCZOS)
reg = np.array(im.crop((440, 435, 480, 475))); m = np.array(capa) > 0
keep = np.array(src.crop((440, 435, 480, 475)))
im.paste(Image.new('RGB', (40, 40), (5, 10, 35)), (440, 435), capa)
t = Image.new('RGB', (80, 20), (255, 255, 255)); dt = ImageDraw.Draw(t)
escribir(dt, 4, 14, [R('Resultado', 'dejavu', (0, 0, 0))], 11)
bb = bbox_tinta(t, (0, 0, 80, 20), (255, 255, 255)); t = t.crop((bb[0] - 1, bb[1] - 1, bb[2] + 2, bb[3] + 2)).rotate(90, expand=True)
im.paste(t, (467 - t.width, int(414 - t.height / 2)))
im.save(DST); print('resultado', t.size)
