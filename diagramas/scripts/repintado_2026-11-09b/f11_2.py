# Diagrama 11.2: borra el trazo oscuro suelto que pisa el borde superior de la tabla del panel 3, sobre «scores»,
# copiando encima un tramo limpio del mismo borde.
import sys
import numpy as np
from PIL import Image
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC); modo = im.mode
a = np.array(im.convert('RGB'))
a[265:270, 868:897] = a[265:270, 838:867]
out = Image.fromarray(a)
if modo == 'P':
    out = out.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
out.save(DST)
