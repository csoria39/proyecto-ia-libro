"""Corrige el Diagrama 15 (imagen del autor, 1672 x 941):
1) quita la forma «(24,)» del paso 1 (el texto de entrenamiento no es un tensor de 24);
2) rehace la fórmula de la pérdida, con subíndices ilegibles: P(yₜ | x₁ … xₜ);
3) leyenda: «Fenaración» -> «Generación»."""
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

FB = '/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf'
FR = '/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

im = Image.open('d15_orig.png').convert('RGB')
d = ImageDraw.Draw(im)

def fondo_de(caja):
    x0, y0, x1, y1 = caja
    borde = [im.getpixel((x, y0 - 1)) for x in range(x0, x1)] + \
            [im.getpixel((x, y1 + 1)) for x in range(x0, x1)]
    return Counter(borde).most_common(1)[0][0]

def parche2(caja, s, fpath, talla, ancho_max, tinta, fondo=None, dx=0):
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
    im.paste(Image.new('RGB', capa.size, tinta), (round(x0 + dx), round((y0 + y1) / 2 - alto / 2 + 1)), capa)

# 1) «(24,)» del paso 1
caja = (300, 215, 348, 236)
d.rectangle(caja, fill=fondo_de(caja))

# 2) fórmula de la pérdida (se conserva «L = −(1/192) · ∑ ln » y se rehace desde «P(»)
MONOB = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
def formula(caja, trozos, talla, tinta, dx=0):
    """Escribe una fórmula con subíndices: trozos = [(texto, es_subindice), ...]."""
    x0, y0, x1, y1 = caja
    Z = 4
    fn, fs = ImageFont.truetype(MONOB, talla * Z), ImageFont.truetype(MONOB, int(talla * 0.68) * Z)
    capa = Image.new('L', ((x1 - x0) * Z, (y1 - y0) * Z), 0)
    dc = ImageDraw.Draw(capa)
    x = dx * Z
    base = (y1 - y0) * Z * 0.5 - talla * Z * 0.62
    for t, sub in trozos:
        f = fs if sub else fn
        dc.text((x, base + (talla * Z * 0.48 if sub else 0)), t, font=f, fill=255)
        x += dc.textlength(t, font=f)
    capa = capa.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    d.rectangle(caja, fill=fondo_de(caja))
    im.paste(Image.new('RGB', capa.size, tinta), (x0, y0), capa)
formula((601, 510, 738, 531), [('P(y', 0), ('t', 1), (' | x', 0), ('1', 1), (' … x', 0), ('t', 1), (')', 0)],
        13, (22, 40, 115), dx=2)

# 3) leyenda
parche2((1505, 884, 1615, 902), 'Generación / muestreo', FB, 15, 104, tinta=(255, 255, 255))

im.save('d15_corregido.png')
