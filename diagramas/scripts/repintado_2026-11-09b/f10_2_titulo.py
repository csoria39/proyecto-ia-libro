# Diagrama 10.2: borra los dos trazos sueltos bajo el título (bajo la «n» de «Long» y la «r» de «Memory»).
import sys
import numpy as np
from PIL import Image
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); a = np.array(im)
for x0, x1 in ((896, 909), (1230, 1241)):
    fondo = np.median(np.concatenate([a[48:50, x0:x1].reshape(-1, 3), a[46:50, x0 - 4:x0].reshape(-1, 3)]), axis=0)
    a[46:49, x0:x1] = fondo.astype(np.uint8)
Image.fromarray(a).save(DST)
