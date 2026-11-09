import sys; sys.path.insert(0, '.')
from pintar import *
from PIL import Image
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB')
import numpy as np
def oscuro(box, n=15):
    a = np.asarray(im.crop(box)).reshape(-1, 3).astype(int)
    return tuple(int(v) for v in a[a.sum(1).argsort()[:n]].mean(0))
c = (40, 40, 40)
reemplazar(im, (662, 997, 803, 1024), (662, 997, 803, 1024),
           [R('Bias', 'mathi', c), R('2', 'math', c, sup=True), R(' + ', 'math', c), R('Variance', 'mathi', c), R(' +', 'math', c)],
           [R('Sesgo', 'mathi', c), R('2', 'math', c, sup=True), R(' + ', 'math', c), R('Varianza', 'mathi', c), R(' +', 'math', c)],
           align='l', maxw=140, color=oscuro((662, 997, 803, 1024)))
for borrar, medir, viejo, nuevo, fam in (
        ((550, 1071, 626, 1087), (552, 1072, 624, 1086), 'Bias (sesgo):', 'Sesgo (bias):', 'fira600'),
        ((690, 1071, 800, 1087), (692, 1072, 798, 1086), 'Variance (varianza):', 'Varianza (variance):', 'fira600'),
        ((553, 1128, 624, 1144), (555, 1129, 622, 1143), 'Alta bias →', 'Alto sesgo →', 'fira600'),
        ((556, 1226, 621, 1242), (558, 1227, 619, 1241), '(alto bias)', '(alto sesgo)', 'fira600')):
    col = oscuro(medir)
    reemplazar(im, borrar, medir, [R(viejo, fam, col)], [R(nuevo, fam, col)], align='m', maxw=(borrar[2] - borrar[0]) + 30)
im.save(DST); print('ok')
