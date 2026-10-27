# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Diagrama 14 nuevo (ChatGPT, 27-10-2026, a partir de Prompt_Diagrama_14_v18.txt), revisado:
1) Recuadro 4: subíndices ilegibles («x·W,Q», «· W,o») → Q = x·W_Q, K = x·W_K, V = x·W_V y «· W_O».
2) Gráfica de GELU: la curva estaba POR ENCIMA de ReLU para x > 0 y ReLU dibujada curva. GELU(x) = x·Φ(x)
   es siempre ≤ ReLU(x), baja un poco por debajo de 0 (mínimo ≈ −0,17 en x ≈ −0,75) y ReLU es recta.
   Se redibuja con valores reales, x de −3 a 2 (GELU(2) = 1,95, dentro del eje de −1 a 2)."""
import sys, math
sys.path.insert(0, '/home/claude/diag/caps')
from PIL import Image, ImageDraw, ImageFont
from fix_caps import cambiar, CB, MB

im = Image.open('d14_nuevo_orig.png').convert('RGB')
TEAL = (4, 144, 143)
def linea(x, yc, s, talla, color, fondo, caja):
    ImageDraw.Draw(im).rectangle(caja, fill=fondo)
    f = ImageFont.truetype(CB, talla * 4)
    dd = ImageDraw.Draw(im)
    bb = dd.textbbox((0, 0), s, font=f)
    capa = Image.new('L', (bb[2] + 16, bb[3] + 16), 0)
    ImageDraw.Draw(capa).text((8, 8), s, font=f, fill=255)
    capa = capa.resize((capa.width // 4, capa.height // 4), Image.LANCZOS)
    im.paste(Image.new('RGB', capa.size, color), (x - 2, round(yc - (bb[1] + bb[3]) / 8 - 2)), capa)
    return capa.width


print(linea(483, 409, 'Q = x·W_Q,   K = x·W_K,   V = x·W_V', 15, (255, 255, 255), TEAL, (480, 399, 750, 419)))
print(linea(483, 431, 'softmax(QKᵀ/√d_k + máscara)·V  ·  W_O', 15, (255, 255, 255), TEAL, (480, 420, 750, 442)))

# ---- gráfica de GELU
Z = 4
X0, X1, XMIN, XMAX = 885, 1042, -3.0, 2.0
Y0, PXU = 552, 22                       # y de 0 y píxeles por unidad
W, H = 400, 110
ox, oy = 845, 495                        # esquina de la zona redibujada
capa = Image.new('RGB', (W * Z, H * Z), (254, 254, 254))
d = ImageDraw.Draw(capa)


def P(x, y):
    return ((X0 + (x - XMIN) / (XMAX - XMIN) * (X1 - X0) - ox) * Z, (Y0 - y * PXU - oy) * Z)


GRID, EJE, NAVY = (220, 228, 238), (40, 55, 90), (20, 40, 90)
for yv in (1, 2):
    d.line([P(XMIN, yv), P(XMAX, yv)], fill=GRID, width=Z)
d.line([P(0, -1), P(0, 2)], fill=GRID, width=Z)
d.line([P(XMIN, 0), P(XMAX, 0)], fill=GRID, width=Z)
d.line([P(XMIN, -1), P(XMIN, 2)], fill=EJE, width=2 * Z)
d.line([P(XMIN, -1), P(XMAX, -1)], fill=EJE, width=2 * Z)
f = ImageFont.truetype(CB, 13 * Z)
for yv in (-1, 0, 1, 2):
    s = str(yv).replace('-', '−')
    x, y = P(XMIN, yv)
    w = d.textlength(s, font=f)
    d.text((x - w - 8 * Z, y - 8 * Z), s, font=f, fill=NAVY)
for xv in (-3, 0, 2):
    s = str(xv).replace('-', '−')
    x, y = P(xv, -1)
    w = d.textlength(s, font=f)
    d.text((x - w / 2, y + 2 * Z), s, font=f, fill=NAVY)
# ReLU discontinua
GRIS = (140, 150, 165)
def discontinua(pts, largo=6 * Z, hueco=5 * Z):
    acc, dibujar = 0, True
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        L = math.hypot(xb - xa, yb - ya); t = 0
        while t < L:
            paso = min((largo if dibujar else hueco) - acc, L - t)
            if dibujar:
                d.line([(xa + (xb - xa) * t / L, ya + (yb - ya) * t / L),
                        (xa + (xb - xa) * (t + paso) / L, ya + (yb - ya) * (t + paso) / L)], fill=GRIS, width=2 * Z)
            t += paso; acc += paso
            if acc >= (largo if dibujar else hueco):
                acc, dibujar = 0, not dibujar
discontinua([P(-3, 0), P(0, 0), P(2, 2)])
# GELU
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
pts = [P(XMIN + i * (XMAX - XMIN) / 400, gelu(XMIN + i * (XMAX - XMIN) / 400)) for i in range(401)]
d.line(pts, fill=(240, 110, 30), width=3 * Z, joint='curve')
capa = capa.resize((W, H), Image.LANCZOS)
im.paste(capa.crop((0, 0, 200, 95)), (ox, oy))
im.save('d14_corregido.png')
