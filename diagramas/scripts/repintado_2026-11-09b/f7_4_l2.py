import sys, math; sys.path.insert(0, '.')
from pintar import *
import numpy as np
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); d = ImageDraw.Draw(im)
FO = (254, 254, 254)
d.rectangle((41, 628, 203, 644), fill=FO)
d.rectangle((34, 645, 203, 771), fill=FO)
O = (32.5, 773.0); B = (122.0, 650.0)
phi = math.radians(-40); A, Bm = 1.0, 0.5
def L(x, y):
    dx, dy = x - B[0], y - B[1]
    u = dx * math.cos(phi) + dy * math.sin(phi); v = -dx * math.sin(phi) + dy * math.cos(phi)
    return (u / A) ** 2 + (v / Bm) ** 2
RAD = 78
ths = np.linspace(0, math.pi / 2, 2000)
pts = [(O[0] + RAD * math.cos(t), O[1] - RAD * math.sin(t)) for t in ths]
P = min(pts, key=lambda p: L(*p)); LP = L(*P)
X0, Y0, X1, Y1 = 34, 628, 204, 772; E = 4
capa = Image.new('RGBA', ((X1 - X0) * E, (Y1 - Y0) * E), (0, 0, 0, 0)); dc = ImageDraw.Draw(capa)
Q = lambda x, y: ((x - X0) * E, (y - Y0) * E)
MOR = (120, 80, 180)
# región permitida por L2 (‖w‖ ≤ r) y circunferencias de la penalización, centradas en el origen
dc.pieslice((Q(O[0] - RAD, O[1] - RAD), Q(O[0] + RAD, O[1] + RAD)), 270, 360, fill=(120, 80, 180, 38))
for r, w, al in ((26, 1.0, 110), (52, 1.0, 110), (RAD, 1.8, 255)):
    dc.arc((Q(O[0] - r, O[1] - r), Q(O[0] + r, O[1] + r)), 270, 360, fill=MOR + (al,), width=int(w * E))
# curvas de nivel de la pérdida, centradas en el óptimo sin regularizar
GR = (90, 90, 100)
for k in (0.12, 0.4, 1.0):
    lev = LP * k
    a, b = A * math.sqrt(lev), Bm * math.sqrt(lev)
    poly = []
    for t in np.linspace(0, 2 * math.pi, 240):
        u, v = a * math.cos(t), b * math.sin(t)
        x = B[0] + u * math.cos(phi) - v * math.sin(phi); y = B[1] + u * math.sin(phi) + v * math.cos(phi)
        poly.append(Q(x, y))
    dc.line(poly + [poly[0]], fill=GR + (200 if k == 1.0 else 120,), width=int((1.4 if k == 1.0 else 1.0) * E))
# trayecto: del óptimo sin regularizar al punto con L2
n = 9
for i in range(n):
    if i % 2: continue
    t0, t1 = i / n, (i + 1) / n
    dc.line((Q(B[0] + (P[0] - B[0]) * t0, B[1] + (P[1] - B[1]) * t0), Q(B[0] + (P[0] - B[0]) * t1, B[1] + (P[1] - B[1]) * t1)), fill=MOR + (255,), width=int(1.6 * E))
for c, rr, col in ((B, 5, (10, 10, 10)), (P, 5, (230, 20, 20))):
    dc.ellipse((Q(c[0] - rr, c[1] - rr), Q(c[0] + rr, c[1] + rr)), fill=col + (255,))
capa = capa.resize((X1 - X0, Y1 - Y0), Image.LANCZOS)
im.paste(capa, (X0, Y0), capa)
escribir(d, 130.5, 651, [R('Sin regularización', 'fira600', (48, 48, 60))], 8.4)
escribir(d, 112, 728, [R('Con L2', 'fira600', (110, 70, 170))], 11)
escribir(d, 112, 741, [R('(más pequeño)', 'fira600', (110, 70, 170))], 8.7)
print('P', P, 'LP', LP)
im.save(DST)
