# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'repintado_2026-11-08'))
from pintar import *
from PIL import Image, ImageDraw
import numpy as np, cv2
im = Image.open(sys.argv[1]).convert('RGB')
a = np.asarray(im).astype(int)
m = np.zeros(a.shape[:2], np.uint8)
y0, y1, x0, x1 = 371, 402, 676, 800
reg = a[y0:y1, x0:x1]
verde = (reg[..., 1] > reg[..., 0] + 15) & (reg[..., 1] > reg[..., 2] + 5)
m[y0:y1, x0:x1][verde] = 255
m = cv2.dilate(m, np.ones((3, 3), np.uint8))
out = cv2.inpaint(np.asarray(im)[:, :, ::-1].copy(), m, 3, cv2.INPAINT_TELEA)[:, :, ::-1]
im = Image.fromarray(out.copy()); d = ImageDraw.Draw(im)
VERDE = (40, 125, 50)
f = fondo_de(im, (680, 376, 752, 401))
d.rectangle((684, 375, 750, 401), fill=f)
escribir(d, 716, 386, [R('Mejor', 'fira600', VERDE)], 11.5, anchor='m')
escribir(d, 716, 399, [R('generalización', 'fira600', VERDE)], 11.5, anchor='m')
im.save(sys.argv[2]); print('ok')
