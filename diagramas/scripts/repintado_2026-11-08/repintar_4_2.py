# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pintar import *
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SRC = os.path.join(BASE, 'originales', 'Diagrama_4.2_antes_repintado_2026-11-08.png')
im = Image.open(SRC).convert('RGB')

# (zona de borrado/medida, texto viejo, texto nuevo, alineación)
cambios = [
    ((646, 1043, 700, 1060), '−0.3799', '0.4013', 'm'),
    ((672, 1068, 715, 1086), '0.4833', '0.3645', 'l'),
    ((622, 1095, 667, 1113), '0.6185', '0.5901', 'l'),
    ((642, 1127, 687, 1145), '0.0165', '0.0220', 'l'),
    ((897, 973, 950, 991), '−0.0439', '−0.0508', 'l'),
    ((802, 1033, 857, 1050), '−0.0067', '−0.0088', 'm'),
    ((808, 1049, 857, 1066), '0.0043', '0.0037', 'm'),
]
for caja, viejo, nuevo, al in cambios:
    c = (12, 12, 12)
    reemplazar(im, caja, caja, [R(viejo, 'mathd', c)], [R(nuevo, 'mathd', c)], align=al)

im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'Diagrama_4.2.png'))
print('ok')
