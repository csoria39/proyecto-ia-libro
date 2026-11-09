# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Numeración de los diagramas por capítulo (27-10-2026): «DIAGRAMA 6 — …» → «DIAGRAMA 11.1 — …».

En cada imagen (en orientación de lectura) se localiza la línea del título, se separa por los espacios en
«DIAGRAMA», el número y el resto, y se recompone: se conservan los píxeles originales de «DIAGRAMA» y del
resto del título, y solo se escribe el número nuevo, con la altura y el estrechamiento de los dígitos
originales. El título se vuelve a centrar donde estaba (el del Diagrama 2.1 va alineado a la izquierda)."""
import json
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

NUEVO = {1: '2.1', 2: '3.1', 3: '4.1', 4: '4.2', 5: '5.1', 6: '7.1', 7: '7.2', 8: '7.3', 9: '9.1', 10: '10.1',
         11: '10.2', 12: '11.1', 13: '11.2', 14: '12.1', 15: '13.1', 16: '17.1', 17: 'G.1', 18: '20.1', 19: '20.2'}
VIEJO = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '7', 7: '8', 8: '9', 9: '10', 10: '11', 11: '12', 12: '6',
         13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19'}
FUENTE = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
T = json.load(open('titulos.json'))


def segmentos(mask, gap):
    cols = np.where(mask.any(axis=0))[0]
    segs, ini, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - prev > gap:
            segs.append((ini, prev + 1)); ini = c
        prev = c
    segs.append((ini, prev + 1))
    return segs


def render(s, alto, escala_x, color, fondo):
    f = ImageFont.truetype(FUENTE, 200)
    d = ImageDraw.Draw(Image.new('L', (1, 1)))
    b = d.textbbox((0, 0), s, font=f)
    capa = Image.new('L', (b[2] - b[0] + 20, b[3] - b[1] + 20), 0)
    ImageDraw.Draw(capa).text((10 - b[0], 10 - b[1]), s, font=f, fill=255)
    capa = capa.crop(capa.getbbox())
    # altura de los dígitos: la del primer carácter que sea dígito
    bd = d.textbbox((0, 0), 'D', font=f)
    k = alto / (bd[3] - bd[1])
    w = max(1, round(capa.width * k * escala_x)); h = max(1, round(capa.height * k))
    capa = capa.resize((w, h), Image.LANCZOS)
    out = Image.new('RGB', (w, h), fondo)
    out.paste(Image.new('RGB', (w, h), color), (0, 0), capa)
    return out, k, bd


def anchura_natural(s, alto):
    f = ImageFont.truetype(FUENTE, 200)
    d = ImageDraw.Draw(Image.new('L', (1, 1)))
    capa = Image.new('L', (2000, 400), 0)
    ImageDraw.Draw(capa).text((10, 10), s, font=f, fill=255)
    bb = capa.getbbox()
    bd = d.textbbox((0, 0), 'D', font=f)
    return (bb[2] - bb[0]) * alto / (bd[3] - bd[1])


def renumerar(n):
    t = T[str(n)]
    im = Image.open(f'orig/image{n}.png').convert('RGB').rotate(t['rot'], expand=True)
    a = np.asarray(im).astype(int)
    (_, (x, y, w, h)) = next(wd for wd in t['words'] if 'DIAGRAMA' in wd[0].upper())   # «DIAGRAMA»
    y0, y1 = y - 4, y + h + max(3, h // 4)
    banda = a[y0:y1]
    fondo = Counter(map(tuple, np.concatenate([banda[0], banda[-1]]))).most_common(1)[0][0]
    tinta = np.abs(banda - np.array(fondo)).sum(axis=2) > 120
    # extensión horizontal del título: desde «DIAGRAMA» hasta el primer hueco grande
    segs = segmentos(tinta[:, max(0, x - 5):], gap=max(4, int(h * 0.22)))
    segs = [(s0 + max(0, x - 5), s1 + max(0, x - 5)) for s0, s1 in segs]
    titulo = [segs[0]]
    for s in segs[1:]:
        if s[0] - titulo[-1][1] > 2.2 * h:
            break
        titulo.append(s)
    s_diag, s_num, resto = titulo[0], titulo[1], titulo[2:]
    tx0, tx1 = s_diag[0], titulo[-1][1]
    # dígitos originales: altura y color
    zona = tinta[:, s_num[0]:s_num[1]]
    filas = np.where(zona.any(axis=1))[0]
    alto = filas[-1] - filas[0] + 1
    pix = banda[:, s_num[0]:s_num[1]][zona]
    dd = np.abs(pix - np.array(fondo)).sum(axis=1)
    color = tuple(int(v) for v in pix[dd >= np.percentile(dd, 80)].mean(axis=0))
    # estrechamiento de la letra: el de «DIAGRAMA» (mismo tipo y misma altura de mayúsculas)
    zd = tinta[:, s_diag[0]:s_diag[1]]; fd = np.where(zd.any(axis=1))[0]
    escala_x = (s_diag[1] - s_diag[0]) / anchura_natural('DIAGRAMA', fd[-1] - fd[0] + 1)
    num, k, bd = render(NUEVO[n], alto, escala_x, color, fondo)
    # recomposición
    gap1 = s_num[0] - s_diag[1]
    gap2 = (resto[0][0] - s_num[1]) if resto else 0
    p_diag = im.crop((s_diag[0], y0, s_diag[1], y1))
    p_resto = im.crop((resto[0][0], y0, tx1, y1)) if resto else None
    ancho = p_diag.width + gap1 + num.width + gap2 + (p_resto.width if p_resto else 0)
    if n == 1:
        nx0 = tx0
    else:
        nx0 = round((tx0 + tx1) / 2 - ancho / 2)
    d = ImageDraw.Draw(im)
    d.rectangle((min(tx0, nx0) - 2, y0, max(tx1, nx0 + ancho) + 2, y1 - 1), fill=fondo)
    xx = nx0
    im.paste(p_diag, (xx, y0)); xx += p_diag.width + gap1
    im.paste(num, (xx, y0 + filas[0])); xx += num.width + gap2
    if p_resto:
        im.paste(p_resto, (xx, y0))
    im.save(f'lect_{n}.png')
    im.rotate(-t['rot'], expand=True).save(f'image{n}.png')
    return NUEVO[n], alto, round(escala_x, 2), ancho - (tx1 - tx0)


if __name__ == '__main__':
    for n in range(1, 20):
        print(n, renumerar(n))
