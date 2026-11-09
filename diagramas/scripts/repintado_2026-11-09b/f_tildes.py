# Tildes de los títulos (3.1, 7.2, 7.4, 9.1): borra las marcas sueltas sobre «CI» y dibuja la tilde sobre la Ó.
# Uso: python3 f_tildes.py <n> <entrada.png> <salida.png>
import sys
from PIL import Image
from tilde import arregla
CAJAS = {'3.1': [((900, 15, 1000, 60), 3)], '7.2': [((730, 0, 830, 45), 2), ((1130, 0, 1230, 45), 3)],
         '7.4': [((860, 5, 960, 50), 3)], '9.1': [((1210, 0, 1310, 45), 3)]}
n, SRC, DST = sys.argv[1:4]
im = Image.open(SRC).convert('RGB')
for caja, k in CAJAS[n]:
    arregla(im, caja, k)
im.save(DST)
