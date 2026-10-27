"""Corrige el Diagrama 16 (imagen del autor, 1672 x 941):
1) barras a la altura correcta en la escala logarítmica y flecha de 7 órdenes;
2) erratas de texto: línea ilegible en la fase 1, «memaño», «Capitulo» (x2),
   «cásico», «respueste», «1024²»."""
import math
from PIL import Image, ImageDraw, ImageFont

FB = '/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf'
FR = '/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf'
BG = (254, 254, 254)

im = Image.open('d16_orig.png').convert('RGB')
d = ImageDraw.Draw(im)

def font(path, size):
    return ImageFont.truetype(path, size)

def texto_centrado(cx, y, s, f, color, fondo=None):
    w = d.textlength(s, font=f)
    x = cx - w / 2
    if fondo:
        bb = d.textbbox((x, y), s, font=f)
        d.rectangle((bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1), fill=fondo)
    d.text((x, y), s, font=f, fill=color)

# ---------- 1) Barras en escala log ----------
# Marcas del eje (medidas en la imagen): 10^4 en y = 561 ... 10^13 en y = 268  ->  32,56 px por década
def y_de(v):
    return 561 - (math.log10(v) - 4) * 32.56

EJE_Y = 561            # línea horizontal del eje
d.rectangle((57, 292, 392, EJE_Y - 1), fill=BG)   # borra barras, flecha y texto

# línea roja discontinua en 10^12
for x in range(60, 390, 12):
    d.line((x, 300, x + 6, 300), fill=(253, 0, 0), width=2)

barras = [  # (x0, x1, valor, color barra, color etiqueta, etiqueta)
    (67, 105, 6e4,    (4, 135, 105),  (0, 121, 98),   '60.000'),
    (119, 157, 124e6, (6, 102, 200),  (6, 90, 190),   '124 M'),
    (172, 209, 1.5e9, (21, 86, 178),  (21, 80, 170),  '1.500 M'),
    (229, 264, 7e9,   (112, 29, 153), (100, 20, 140), '7.000 M'),
    (282, 319, 175e9, (112, 28, 153), (100, 20, 140), '175.000 M'),
    (339, 375, 671e9, (189, 2, 1),    (174, 0, 0),    '671.000 M'),
]
fl = font(FB, 15)
for x0, x1, v, cb, ce, et in barras:
    yt = round(y_de(v))
    d.rectangle((x0, yt, x1, EJE_Y - 1), fill=cb)
    ye = yt - 19 if v < 5e11 else 279        # la de DeepSeek-V3, por encima de la línea de 10^12
    texto_centrado((x0 + x1) / 2, ye, et, fl, ce, fondo=BG)

# flecha doble de 7 órdenes: de la barra del Nano-LLM a la de DeepSeek-V3
NAR = (232, 120, 0)
xa = 86
ya0, ya1 = round(y_de(671e9)), round(y_de(6e4)) - 22   # sobre la etiqueta «60.000»
d.line((xa, ya0 + 8, xa, ya1 - 8), fill=NAR, width=4)
d.polygon([(xa, ya0), (xa - 7, ya0 + 11), (xa + 7, ya0 + 11)], fill=NAR)
d.polygon([(xa, ya1), (xa - 7, ya1 - 11), (xa + 7, ya1 - 11)], fill=NAR)
ft = font(FB, 19)
d.text((98, 318), '×10 millones', font=ft, fill=NAR)
d.text((110, 340), '(7 órdenes)', font=ft, fill=NAR)

# ---------- 2) Erratas de texto ----------
def reescribir(caja, xy, s, f, color, fondo=BG):
    d.rectangle(caja, fill=fondo)
    d.text(xy, s, font=f, fill=color)

im.save('d16_paso1.png')

from collections import Counter

def fondo_de(caja):
    x0, y0, x1, y1 = caja
    borde = [im.getpixel((x, y0 - 1)) for x in range(x0, x1)] + \
            [im.getpixel((x, y1 + 1)) for x in range(x0, x1)]
    return Counter(borde).most_common(1)[0][0]

def tinta_de(caja):
    x0, y0, x1, y1 = caja
    px = [im.getpixel((x, y)) for x in range(x0, x1) for y in range(y0, y1)]
    px.sort(key=sum)
    n = max(1, len(px) // 40)
    return tuple(sum(p[i] for p in px[:n]) // n for i in range(3))

def parche(caja, s, fpath, ancho_obj, tinta=None, fondo=None, dx=0, talla=None, centrar=False):
    """Borra la caja y escribe s a la izquierda, con el tamaño cuyo ancho se acerca a ancho_obj."""
    x0, y0, x1, y1 = caja
    tinta = tinta or tinta_de(caja)
    fondo = fondo or fondo_de(caja)
    mejor = talla or min(range(9, 26), key=lambda t: abs(d.textlength(s, font=font(fpath, t)) - ancho_obj))
    f = font(fpath, mejor)
    d.rectangle(caja, fill=fondo)
    bb = d.textbbox((0, 0), s, font=f)
    alto = bb[3] - bb[1]
    if centrar:
        dx = (x1 - x0 - d.textlength(s, font=f)) / 2
    d.text((x0 + dx, (y0 + y1) / 2 - alto / 2 - bb[1]), s, font=f, fill=tinta)
    return mejor, tinta, fondo

def parche2(caja, s, fpath, talla, ancho_max, tinta=None, fondo=None, centrar=False, dx=0):
    """Como parche(), pero escribe con la altura de letra del original (talla) y, si el texto
    ocupa más de ancho_max, lo estrecha en horizontal hasta ese ancho (letra condensada)."""
    x0, y0, x1, y1 = caja
    tinta = tinta or tinta_de(caja)
    fondo = fondo or fondo_de(caja)
    f = font(fpath, talla * 4)                       # se dibuja a 4x y se reduce: mejor antialias
    w = int(d.textlength(s, font=f)) + 8
    capa = Image.new('L', (w, talla * 6), 0)
    ImageDraw.Draw(capa).text((4, talla), s, font=f, fill=255)
    capa = capa.crop(capa.getbbox())
    ancho = min(capa.width / 4, ancho_max)
    alto = capa.height / 4
    capa = capa.resize((round(ancho), round(alto)), Image.LANCZOS)
    d.rectangle(caja, fill=fondo)
    xx = x0 + ((x1 - x0 - ancho) / 2 if centrar else dx)
    yy = (y0 + y1) / 2 - alto / 2 + 1
    im.paste(Image.new('RGB', capa.size, tinta), (round(xx), round(yy)), capa)
    return round(ancho), round(alto)

FBC = FB
FRC = FR
print(parche2((853, 349, 1209, 367), 'GPT-3: 300.000 M tokens · DeepSeek-V3: 14,8 billones de tokens',
              FB, 14, 340, tinta=(20, 75, 90), fondo=(217, 244, 236), centrar=True))
print(parche2((470, 626, 640, 640), 'No todo depende del tamaño:', FB, 15, 160, dx=2))
print(parche2((470, 642, 795, 657), 'en el Capítulo 15, un modelo de solo 58.000 parámetros', FR, 15, 320, dx=2))
print(parche2((446, 818, 800, 833), 'La misma idea que pruebas en el Capítulo 15 con un modelo de 2 bloques.', FR, 14, 340, dx=2))
print(parche2((864, 664, 960, 681), 'RLHF (clásico)', FB, 17, 92, dx=2))
print(parche2((966, 787, 1080, 801), 'la respuesta preferida', FR, 15, 102, dx=3))
print(parche2((1539, 657, 1641, 674), '(1 GB = 1024³ bytes)', FR, 14, 96, dx=2))
im.save('d16_corregido.png')
