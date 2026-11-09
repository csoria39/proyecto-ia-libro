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

# 1. columna BPE: la frase es 'la IA', como en las otras dos columnas
col = oscuro((204, 294, 256, 305))
reemplazar(im, (196, 291, 266, 308), (204, 294, 256, 305), [R("Frase: 'IA'", 'fira500', col)], [R("Frase: 'la IA'", 'fira500', col)], align='m', maxw=110)

# 2. inteligencia → [intel][igen][cia] (3 tokens, GPT-2)
FO = fondo_de(im, (160, 400, 305, 432))
verde = oscuro((180, 411, 212, 423))
d.rectangle((165, 401, 302, 431), fill=FO)
piezas = ['intel', 'igen', 'cia']; GAP = 5; PAD = 5
s = 13.5
while sum(ancho([R(p, 'fira600')], s) + 2 * PAD for p in piezas) + GAP * 2 > 128: s -= 0.25
x = 233 - (sum(ancho([R(p, 'fira600')], s) + 2 * PAD for p in piezas) + GAP * 2) / 2
E = 4
for p in piezas:
    w = ancho([R(p, 'fira600')], s) + 2 * PAD
    X0, Y0 = int(x) - 2, 403
    capa = Image.new('RGBA', ((int(w) + 5) * E, 28 * E), (0, 0, 0, 0)); dc = ImageDraw.Draw(capa)
    dc.rounded_rectangle(((x - X0) * E, (405 - Y0) * E, (x + w - X0) * E, (427.5 - Y0) * E), radius=2.5 * E,
                         fill=(255, 255, 255, 255), outline=(58, 125, 70, 255), width=int(1.5 * E))
    capa = capa.resize((int(w) + 5, 28), Image.LANCZOS); im.paste(capa, (X0, Y0), capa)
    escribir(d, x + w / 2, 421.5, [R(p, 'fira600', verde)], s, anchor='m')
    x += w + GAP
# «2 tokens» → «3 tokens» (copiado de la línea de arriba, misma letra)
tres = im.crop((209, 352, 258, 366))
d.rectangle((209, 436, 258, 450), fill=fondo_de(im, (209, 436, 258, 450)))
im.paste(tres, (209, 436))

# 3. tabla de pares: columna vacía «−» de la cabecera
cab = A[346:368, 600:700].reshape(-1, 3); fcab = tuple(int(v) for v in np.median(cab[cab.sum(1) < 400], axis=0))
print('cab', fcab)
im.save(DST)

im = Image.open(DST).convert('RGB'); d = ImageDraw.Draw(im)
A = np.asarray(im).astype(int)
# tabla de pares: quitar la columna vacía «−» (y su separador) de la cabecera
fc = tuple(int(v) for v in np.median(A[349:366, 600:640].reshape(-1, 3)[A[349:366, 600:640].reshape(-1, 3).sum(1) < 300], axis=0))
d.rectangle((643, 348, 697, 366), fill=fc)
# OOV: errata «supercalifragiilistico»
col = oscuro((876, 428, 983, 445))
reemplazar(im, (874, 426, 985, 446), (876, 427, 983, 446), [R("'supercalifragiilistico'", 'fira500', col)], [R("'supercalifragilistico'", 'fira500', col)], align='l')
# implicación: la fertilidad habla de coste y de contexto, no de «razonar mejor»
f = fondo_de(im, (900, 748, 1238, 810)); c = oscuro((905, 750, 1235, 808))
viejo = 'idiomas — consecuencia del corpus de entrenamiento.'
bb = bbox_tinta(im, (900, 792, 1238, 810), f)
s = 20
while bbox_render([R(viejo, 'fira600')], s)[2] - bbox_render([R(viejo, 'fira600')], s)[0] > bb[2] - bb[0]: s -= 0.25
d.rectangle((903, 748, 1236, 810), fill=f)
lin = ['Implicación: el mismo texto ocupa más tokens', 'fuera del inglés: más coste y menos contexto', 'útil — consecuencia del corpus de entrenamiento.']
s2 = min(s, min(ajustar([R(l, 'fira600')], s, 328) for l in lin))
for i, l in enumerate(lin):
    escribir(d, 906, 764.5 + i * 20.5, [R(l, 'fira600', c)], s2)
print('impl', s, s2, bb)
im.save(DST)
