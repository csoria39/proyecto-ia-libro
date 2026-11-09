# Diagrama 4.2: δ_m^(1) en el panel 6 (con los glifos de δ_1^(1)) y b¹ como columna ([0.0 0.0]ᵀ).
import sys; sys.path.insert(0, '.')
from pintar import *
import numpy as np
from PIL import Image, ImageDraw, ImageChops
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB')
delta = im.crop((161, 1241, 170, 1254)); sup = im.crop((170, 1237, 186, 1248))
d = ImageDraw.Draw(im)
d.rectangle((162, 1361, 193, 1384), fill=(254, 254, 254))
def pon(p, x, y):
    reg = im.crop((x, y, x + p.width, y + p.height)); im.paste(ImageChops.darker(reg, p), (x, y))
pon(delta, 164, 1365); pon(sup, 173, 1361)
for x in (174.5, 174.8):
    escribir(d, x, 1382, [R('m', 'mathbi', (15, 68, 11))], 12)
a = np.asarray(im).astype(int); reg = a[1126:1135, 443:460].reshape(-1, 3)
col = tuple(int(v) for v in reg[reg.sum(1).argsort()[:10]].mean(0))
escribir(d, 501.5, 1129.5, [R('T', 'math', col)], 10)
im.save(DST)
