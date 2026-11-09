# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os, math
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'repintado_2026-11-08'))
from pintar import *
from PIL import Image, ImageDraw
import numpy as np, cv2

SRC = sys.argv[1]; OUT = sys.argv[2]; ZIP = sys.argv[3]
im = Image.open(SRC).convert('RGB')
a = np.asarray(im).copy()

# --- panel 1: flechas de una sola punta hacia abajo ---
P = [(146, 270), (191, 319), (223, 368), (251, 418), (285, 472), (339, 520), (359, 583)]
mask = np.zeros(a.shape[:2], np.uint8)
for (x0, y0), (x1, y1) in zip(P, P[1:]):
    L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
    cv2.line(mask, (int(x0 + ux * 8), int(y0 + uy * 8)), (int(x1 - ux * 8), int(y1 - uy * 8)), 255, 17)
# protege los círculos
for x, y in P:
    cv2.circle(mask, (x, y), 9, 0, -1)
# rótulo deformado «Iteraciones (actualización de pesos)»
cv2.rectangle(mask, (298, 396), (398, 462), 255, -1)
a = cv2.inpaint(a[:, :, ::-1].copy(), mask, 6, cv2.INPAINT_TELEA)[:, :, ::-1]
im = Image.fromarray(a.copy())

E = 4
x0b, y0b, x1b, y1b = 120, 250, 420, 610
capa = im.crop((x0b, y0b, x1b, y1b)).resize(((x1b - x0b) * E, (y1b - y0b) * E), Image.LANCZOS)
dc = ImageDraw.Draw(capa)
def Q(x, y): return ((x - x0b) * E, (y - y0b) * E)
for (xa, ya), (xb, yb) in zip(P, P[1:]):
    L = math.hypot(xb - xa, yb - ya); ux, uy = (xb - xa) / L, (yb - ya) / L
    s = (xa + ux * 10, ya + uy * 10); e = (xb - ux * 11, yb - uy * 11)
    for col, w in (((40, 40, 40), 4.2), ((255, 255, 255), 2.4)):
        dc.line([Q(*s), Q(e[0] - ux * 5, e[1] - uy * 5)], fill=col, width=int(w * E))
    nx, ny = -uy, ux; h = 8; wd = 4.5
    tri = [Q(*e), Q(e[0] - ux * h + nx * wd, e[1] - uy * h + ny * wd), Q(e[0] - ux * h - nx * wd, e[1] - uy * h - ny * wd)]
    dc.polygon(tri, fill=(255, 255, 255), outline=(40, 40, 40), width=int(1.1 * E))
im.paste(capa.resize((x1b - x0b, y1b - y0b), Image.LANCZOS), (x0b, y0b))
d = ImageDraw.Draw(im)
NAR = (240, 110, 20)
def txt_halo(x, y, runs, s):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                escribir(d, x + dx, y + dy, [(r[0], r[1], (255, 255, 255), r[3]) for r in runs], s)
    escribir(d, x, y, runs, s)
txt_halo(302, 422, [R('Iteraciones', 'fira700', NAR)], 16)
txt_halo(302, 439, [R('(actualización', 'fira500', NAR)], 13)
txt_halo(302, 454, [R('de pesos)', 'fira500', NAR)], 13)

# --- panel 4: gradiente y descenso colineales (se toma de la versión verificada del zip) ---
z = Image.open(ZIP).convert('RGB')
im.paste(z.crop((323, 862, 598, 1101)), (323, 862))
im.save(OUT)
print('ok')
