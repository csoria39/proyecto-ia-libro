# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Diagrama 13 nuevo (ChatGPT, 27-10-2026, a partir de Prompt_Diagrama_13_v18.txt), revisado:
1) Las dos gráficas de barras de «banco» se redibujan a escala (las barras de 0,094 y 0,097 tenían
   longitudes distintas entre sí y muescas; «el» del segundo gráfico salía casi invisible y «préstamo»,
   en letra más pequeña). Valores de la salida impresa en 11.6.
2) Recuadro Multi-Head: «W_Q las mezcla» y «· W_Q» → W_O (la matriz de salida); W_Q, W_K, W_V legibles."""
import sys
sys.path.insert(0, '/home/claude/diag/caps')
from PIL import Image, ImageDraw, ImageFont
from fix_caps import cambiar, CR, MB

im = Image.open('d13_nuevo_orig.png').convert('RGB')
d = ImageDraw.Draw(im)
BG = (254, 254, 254)
AZUL = (2, 102, 205)
TINTA = (25, 30, 40)
EJE = (200, 205, 212)
Z = 4
fl = ImageFont.truetype(CR, 13 * Z)
fv = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 11 * Z)


def texto(x, yc, s, f, color):
    b = d.textbbox((0, 0), s, font=f)
    capa = Image.new('L', (b[2] + 8 * Z, b[3] + 8 * Z), 0)
    ImageDraw.Draw(capa).text((4 * Z, 4 * Z), s, font=f, fill=255)
    capa = capa.crop(capa.getbbox()) if False else capa
    capa = capa.resize((capa.width // Z, capa.height // Z), Image.LANCZOS)
    alto = (b[3] - b[1]) / Z
    y = yc - alto / 2 - b[1] / Z - 4
    im.paste(Image.new('RGB', capa.size, color), (round(x - 4), round(y)), capa)


def grafica(caja, x_barra, filas, ys, px_por_unidad=390):
    d.rectangle(caja, fill=BG)
    d.line((x_barra - 1, ys[0] - 9, x_barra - 1, ys[-1] + 9), fill=EJE, width=1)
    for (pal, v), y in zip(filas, ys):
        texto(caja[0] + 4, y, pal, fl, TINTA)
        largo = round(v * px_por_unidad)
        d.rectangle((x_barra, y - 6, x_barra + largo, y + 6), fill=AZUL)
        texto(x_barra + largo + 7, y, f'{v:.3f}'.replace('.', ','), fv, TINTA)


grafica((1236, 259, 1468, 386), 1300,
        [('el', .094), ('banco', .299), ('está', .094), ('cerca', .119), ('del', .094), ('río', .299)],
        [270, 291, 312, 333, 354, 375])
grafica((1236, 440, 1468, 564), 1300,
        [('el', .097), ('banco', .307), ('me', .097), ('dio', .097), ('un', .097), ('préstamo', .307)],
        [450, 471, 492, 512, 533, 553])

print(cambiar(im, (1243, 655, 1600, 674), 'Cada cabeza tiene sus propias W_Q, W_K, W_V;',
              'Cada cabeza tiene sus propias W_Q, W_K y W_V;', CR, 'izq'))
print(cambiar(im, (1243, 674, 1600, 691), 'sus salidas se concatenan y W_O las mezcla.',
              'sus salidas se concatenan y W_O las mezcla.', CR, 'izq'))
print(cambiar(im, (1243, 694, 1600, 712), 'MultiHead = Concat(head₁, …, head_H) · W_O',
              'MultiHead = Concat(head₁, …, head_H) · W_O', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 'izq'))
im.save('d13_corregido.png')
