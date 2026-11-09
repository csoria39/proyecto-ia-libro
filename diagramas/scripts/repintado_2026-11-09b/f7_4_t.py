import sys; sys.path.insert(0, '.')
from pintar import *
import numpy as np
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); d = ImageDraw.Draw(im)
a = np.asarray(im.crop((528, 929, 730, 948))).reshape(-1, 3).astype(int)
fill = tuple(int(v) for v in np.median(a[a.sum(1) < 300], axis=0))
viejo = '5. INTUICIÓN: BIAS–VARIANZA'
s = 20
while bbox_render([R(viejo, 'firac600')], s)[2] - bbox_render([R(viejo, 'firac600')], s)[0] > 713 - 531: s -= 0.25
d.rectangle((529, 929, 728, 947), fill=fill)
escribir(d, 531 - bbox_render([R(viejo, 'firac600')], s)[0], 943, [R('5. INTUICIÓN: SESGO–VARIANZA', 'firac600', (255, 255, 255))], s)
im.save(DST); print(fill, s)
