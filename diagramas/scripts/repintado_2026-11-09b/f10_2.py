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
# 1. eje Y de la gráfica de la RNN: «Magnitud del gradiente», en dos líneas y girado
fo = fondo_de(im, (1174, 215, 1206, 310)); c = oscuro((1177, 244, 1188, 290))
d.rectangle((1175, 218, 1205, 310), fill=fo)
t = Image.new('RGB', (110, 34), fo); dt = ImageDraw.Draw(t)
escribir(dt, 55, 13, [R('Magnitud del', 'fira600', c)], 11, anchor='m')
escribir(dt, 55, 27, [R('gradiente', 'fira600', c)], 11, anchor='m')
t = t.rotate(90, expand=True)
im.paste(t, (1190 - t.width // 2, int(264.5 - t.height / 2)))
# 2. intervalos abiertos con la notación del libro
S = None
for medir, viejo, nuevo in (((398, 481, 545, 497), 'controla la cantidad (0-1)', 'controla la cantidad (0, 1)'),
                            ((398, 501, 559, 517), 'controla el contenido (-1,1)', 'controla el contenido (−1, 1)')):
    c = oscuro(medir)
    S = reemplazar(im, (medir[0] - 1, medir[1] - 1, medir[2] + 14, medir[3] + 1), medir, [R(viejo, 'fira500', c)], [R(nuevo, 'fira500', c)], align='l', size=S)[0]
im.save(DST); print('ok')
