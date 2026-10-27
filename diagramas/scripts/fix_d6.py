"""Diagrama 6 (ilustración del autor), edición mínima (Incidencia 5):
1) las tres casillas «−∞» de la tabla de atención causal pasan a «0» (la tabla muestra pesos
   DESPUÉS del softmax, como indica «= 1.0 ✓» en cada fila); el subtítulo lo explica:
   «Solo el pasado: −∞ antes del softmax → 0 después»;
2) «H cabezas aprenden H tipos de relaciones en paralelo:» pasa a
   «H cabezas aprenden en paralelo distintos tipos de relación, por ejemplo:».
El original se conserva en d6_orig.png (orientación de lectura) y en bak_media_opt_image12.png."""
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

FB = '/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf'
FI = '/usr/share/fonts/truetype/crosextra/Carlito-Italic.ttf'
FR = '/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf'
im = Image.open('d6_orig.png').convert('RGB')
d = ImageDraw.Draw(im)


def fondo_de(caja):
    x0, y0, x1, y1 = caja
    borde = [im.getpixel((x, y0 - 1)) for x in range(x0, x1)] + [im.getpixel((x, y1 + 1)) for x in range(x0, x1)]
    return Counter(borde).most_common(1)[0][0]


def texto(caja, s, fpath, talla, ancho_max, tinta, fondo=None, centrar=False, dx=0, dy=0):
    x0, y0, x1, y1 = caja
    fondo = fondo or fondo_de(caja)
    f = ImageFont.truetype(fpath, talla * 4)
    w = int(d.textlength(s, font=f)) + 8
    capa = Image.new('L', (w, talla * 6), 0)
    ImageDraw.Draw(capa).text((4, talla), s, font=f, fill=255)
    capa = capa.crop(capa.getbbox())
    ancho, alto = min(capa.width / 4, ancho_max), capa.height / 4
    capa = capa.resize((round(ancho), round(alto)), Image.LANCZOS)
    d.rectangle(caja, fill=fondo)
    xx = x0 + ((x1 - x0 - ancho) / 2 if centrar else dx)
    im.paste(Image.new('RGB', capa.size, tinta), (round(xx), round((y0 + y1) / 2 - alto / 2 + dy)), capa)


ROJO = (200, 9, 3)
for caja in [(904, 525, 948, 549), (956, 525, 1001, 549), (956, 554, 1001, 579)]:
    texto(caja, '0', FB, 22, 40, (255, 255, 255), fondo=ROJO, centrar=True)
texto((830, 471, 1152, 491), 'Solo el pasado: −∞ antes del softmax → 0 después', FI, 19, 318, (225, 90, 20), dx=3)
texto((1241, 468, 1566, 490), 'H cabezas aprenden en paralelo distintos tipos de relación, por ejemplo:',
      FR, 18, 322, (20, 20, 30), dx=2)
im.save('d6_corregido.png')
