import sys; sys.path.insert(0, '.')
from pintar import *
import numpy as np
from PIL import Image, ImageDraw, ImageChops
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); d = ImageDraw.Draw(im)
src = im.copy()
def pon(p, x, y):
    reg = im.crop((x, y, x + p.width, y + p.height)); im.paste(ImageChops.darker(reg, p), (x, y))
# 1. h4 con el mismo tipo que h1-h3: «h» de h3 (desplazado 92 px, misma caja) y «4» de y4
d.rectangle((728, 250, 766, 274), fill=fondo_de(im, (728, 250, 766, 274)))
pon(src.crop((645, 253, 657, 270)), 645 + 92, 253)
cuatro = np.asarray(src.crop((748, 189, 758, 202)).convert('L')).astype(float)
alfa = np.clip((cuatro.max() - cuatro) / (cuatro.max() - cuatro.min()), 0, 1) ** 0.6
h = np.asarray(src.crop((645, 253, 657, 270))).reshape(-1, 3).astype(int)
teal = tuple(int(v) for v in h[h.sum(1).argsort()[:8]].mean(0))
im.paste(Image.new('RGB', (10, 13), teal), (747, 260), Image.fromarray((alfa * 255).astype(np.uint8)))
im.save(DST); print('ok')

# 2. «Procesamiento independiente»: el MLP da salidas y₁, y₂, y₃ (no h): «y» del recuadro y_t de arriba
im = Image.open(DST).convert('RGB'); d = ImageDraw.Draw(im); src = im.copy()
ye = src.crop((276, 438, 287, 453))
for dy in (0, 49, 99):
    d.rectangle((277, 600 + dy, 286, 612 + dy), fill=fondo_de(im, (270, 598 + dy, 300, 616 + dy)))
    pon(ye, 276, 602 + dy)
im.save(DST); print('y ok')
