# -*- coding: utf-8 -*-
"""Motor de repintado: borra zonas con el color de fondo y escribe texto con Fira Sans (con respaldo por carácter),
fórmulas en Latin Modern y monoespaciada. Unidades: píxeles de la imagen."""
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import functools, os

# Fira Sans / Fira Sans Condensed / Fira Mono (SIL OFL 1.1) en diagramas/scripts/fuentes/; FIRA_DIR la cambia.
FD = os.environ.get('FIRA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fuentes')) + '/'
DJ = '/usr/share/fonts/truetype/dejavu/'
LM = '/usr/share/texmf/fonts/opentype/public/lm/'
SUBSETS = ['latin', 'latin-ext', 'greek', 'greek-ext', 'symbols2']


@functools.lru_cache(None)
def cmap(path):
    try:
        return set(TTFont(path).getBestCmap().keys())
    except Exception:
        return set()


def fira_paths(peso=600, familia='fira-sans', italic=False):
    est = 'italic' if italic else 'normal'
    out = []
    for s in SUBSETS:
        p = f'{FD}{familia}-{s}-{peso}-{est}.woff'
        if os.path.exists(p):
            out.append(p)
    return out


RESPALDO = {
    'sans': [DJ + 'DejaVuSans-Bold.ttf', DJ + 'DejaVuSans.ttf'],
    'mono': [DJ + 'DejaVuSansMono-Bold.ttf', DJ + 'DejaVuSansMono.ttf'],
}


def familia(nombre):
    """Lista de rutas de fuente, por orden de preferencia."""
    if nombre.startswith('fira'):            # 'fira600', 'fira700i', 'firac600'
        fam = 'fira-sans-condensed' if nombre.startswith('firac') else 'fira-sans'
        resto = nombre[5:] if nombre.startswith('firac') else nombre[4:]
        it = resto.endswith('i')
        peso = int(resto.rstrip('i'))
        return fira_paths(peso, fam, it) + RESPALDO['sans']
    if nombre == 'mono':
        return [LM + 'lmmonolt10-bold.otf', LM + 'lmmono10-regular.otf'] + RESPALDO['mono']
    if nombre == 'monob':
        return [DJ + 'DejaVuSansMono-Bold.ttf']
    if nombre == 'mono_r':
        return [DJ + 'DejaVuSansMono.ttf']
    if nombre == 'mathi':                   # cursiva matemática
        return [LM + 'lmroman10-italic.otf', LM + 'lmroman10-regular.otf', LM + 'latinmodern-math.otf', DJ + 'DejaVuSerif-Italic.ttf']
    if nombre == 'mathbi':
        return [LM + 'lmroman10-bolditalic.otf', LM + 'lmroman10-italic.otf', LM + 'latinmodern-math.otf']
    if nombre == 'mathd':
        return [LM + 'lmromandemi10-regular.otf', LM + 'lmroman10-regular.otf', LM + 'latinmodern-math.otf']
    if nombre == 'math':
        return [LM + 'lmroman10-regular.otf', LM + 'latinmodern-math.otf', DJ + 'DejaVuSerif.ttf']
    if nombre == 'mathb':
        return [LM + 'lmroman10-bold.otf', LM + 'latinmodern-math.otf', DJ + 'DejaVuSerif-Bold.ttf']
    if nombre == 'lmmono':
        return [LM + 'lmmono10-regular.otf', DJ + 'DejaVuSansMono.ttf']
    if nombre == 'free':
        return ['/usr/share/fonts/truetype/freefont/FreeSans.ttf', DJ + 'DejaVuSans.ttf']
    if nombre == 'freeb':
        return ['/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', DJ + 'DejaVuSans-Bold.ttf']
    if nombre == 'freebi':
        return ['/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf', DJ + 'DejaVuSans-BoldOblique.ttf']
    if nombre == 'freei':
        return ['/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf', DJ + 'DejaVuSans-Oblique.ttf']
    if nombre == 'dejavu':
        return [DJ + 'DejaVuSans-Bold.ttf']
    if nombre == 'dejavur':
        return [DJ + 'DejaVuSans.ttf']
    raise KeyError(nombre)


@functools.lru_cache(None)
def fuente(path, size):
    return ImageFont.truetype(path, size)


def elegir(fam, ch):
    for p in familia(fam):
        if ord(ch) in cmap(p):
            return p
    return familia(fam)[-1]


def trozos(texto, fam):
    """Divide el texto en trozos que usan la misma fuente física."""
    out = []
    for ch in texto:
        p = elegir(fam, ch)
        if out and out[-1][1] == p:
            out[-1][0] += ch
        else:
            out.append([ch, p])
    return out


def ancho_run(texto, fam, size, track=0):
    return sum(fuente(p, size).getlength(t) for t, p in trozos(texto, fam)) + track * len(texto)


def run_size(r, size):
    """r = (texto, fam, color, {'sub':True,'sup':True,'escala':0.9, 'dy':0})"""
    opts = r[3] if len(r) > 3 else {}
    s = size * opts.get('escala', 1.0)
    if opts.get('sub') or opts.get('sup'):
        s *= 0.7
    return max(6, round(s))


def ancho(runs, size):
    return sum(ancho_run(r[0], r[1], run_size(r, size), (r[3] if len(r) > 3 else {}).get('track', 0)) for r in runs)


SS = 4   # supermuestreo del texto


def escribir(d, x, y, runs, size, anchor='l', base=None):
    """Escribe una línea de runs con supermuestreo. (x, y): anclaje; y es la LÍNEA BASE. anchor: l, m, r."""
    from PIL import Image as _I, ImageDraw as _D
    if isinstance(runs, str):
        raise TypeError('runs')
    im = d._image
    w = ancho(runs, size)
    if anchor == 'm':
        x -= w / 2
    elif anchor == 'r':
        x -= w
    # capa: margen generoso arriba y abajo
    asc, desc = int(size * 1.6), int(size * 0.8)
    X0, Y0 = int(x) - 4, int(y) - asc
    W, H = int(w) + 12, asc + desc
    for r in runs:
        texto, fam, color = r[0], r[1], r[2]
        opts = r[3] if len(r) > 3 else {}
        s = run_size(r, size)
        dy = opts.get('dy', 0)
        if opts.get('sub'):
            dy += round(size * 0.22)
        if opts.get('sup'):
            dy -= round(size * 0.38)
        track = opts.get('track', 0)
        mask = _I.new('L', (W * SS, H * SS), 0)
        dm = _D.Draw(mask)
        cx = (x - X0) * SS
        for t, p in trozos(texto, fam):
            f = fuente(p, s * SS)
            for ch in (t if track else [t]):
                dm.text((cx, (y + dy - Y0) * SS), ch, font=f, fill=255, anchor='ls')
                cx += f.getlength(ch) + track * SS
        x += (cx / SS) - (x - X0)
        mask = mask.resize((W, H), _I.LANCZOS)
        capa = _I.new('RGB', (W, H), tuple(color) if not isinstance(color, int) else (color,) * 3)
        im.paste(capa, (X0, Y0), mask)
    return w


def ajustar(runs, size, maxw, minimo=7):
    """Reduce el tamaño hasta que la línea quepa en maxw."""
    s = size
    while s > minimo and ancho(runs, s) > maxw:
        s -= 0.5
    return s


def R(texto, fam='fira600', color=(0, 0, 0), **opts):
    return (texto, fam, color, opts)


def hexrgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def fondo_de(im, box):
    """Color de fondo: el más frecuente en el perímetro del recuadro (cuantizado), promediado."""
    import collections
    x0, y0, x1, y1 = box
    px = im.load()
    per = [px[x, y0] for x in range(x0, x1)] + [px[x, y1 - 1] for x in range(x0, x1)] + \
          [px[x0, y] for y in range(y0, y1)] + [px[x1 - 1, y] for y in range(y0, y1)]
    q = collections.Counter((p[0] // 6, p[1] // 6, p[2] // 6) for p in per)
    k = q.most_common(1)[0][0]
    sel = [p for p in per if (p[0] // 6, p[1] // 6, p[2] // 6) == k]
    return tuple(round(sum(c[i] for c in sel) / len(sel)) for i in range(3))


def tinta_de(im, box, fondo=None):
    """Color de tinta dominante (el más alejado del fondo, entre los frecuentes)."""
    import collections
    px = list(im.crop(box).getdata())
    fondo = fondo or fondo_de(im, box)
    lejos = [p for p in px if sum(abs(a - b) for a, b in zip(p, fondo)) > 250]
    if not lejos:
        return (0, 0, 0)
    c = collections.Counter((p[0] // 6 * 6, p[1] // 6 * 6, p[2] // 6 * 6) for p in lejos)
    return c.most_common(1)[0][0]


def altura_mayusculas(im, box, fondo=None):
    """Alto en píxeles de la tinta de una línea de texto (aprox. cuerpo de las letras)."""
    import collections
    reg = im.crop(box)
    px = reg.load()
    fondo = fondo or collections.Counter(reg.getdata()).most_common(1)[0][0]
    W, H = reg.size
    filas = [y for y in range(H) if sum(1 for x in range(W) if sum(abs(a - b) for a, b in zip(px[x, y], fondo)) > 200) > 2]
    return (box[1] + filas[0], box[1] + filas[-1]) if filas else None


def bbox_tinta(im, box, fondo, umbral=160):
    reg = im.crop(box); px = reg.load(); W, H = reg.size
    xs, ys = [], []
    for y in range(H):
        for x in range(W):
            if sum(abs(a - b) for a, b in zip(px[x, y], fondo)) > umbral:
                xs.append(x); ys.append(y)
    if not xs:
        return None
    return (box[0] + min(xs), box[1] + min(ys), box[0] + max(xs) + 1, box[1] + max(ys) + 1)


def bbox_render(runs, size):
    """bbox de la tinta de runs escritos con la línea base en y=0, x=0."""
    from PIL import Image as _I, ImageDraw as _D
    w = int(ancho(runs, size)) + 40
    tmp = _I.new('RGB', (w, int(size * 3)), (0, 0, 0))
    d = _D.Draw(tmp)
    runs2 = [(r[0], r[1], (255, 255, 255), r[3] if len(r) > 3 else {}) for r in runs]
    escribir(d, 10, int(size * 2), runs2, size)
    b = tmp.convert('L').point(lambda v: 255 if v > 90 else 0).getbbox()
    return (b[0] - 10, b[1] - int(size * 2), b[2] - 10, b[3] - int(size * 2))


def reemplazar(im, borrar, medir, viejo, nuevo, align='l', maxw=None, fondo=None, color=None, dx=0, dy=0, size=None):
    """Borra `borrar` y escribe `nuevo` con el tamaño y la línea base del texto `viejo` (medido en `medir`)."""
    fondo = fondo or fondo_de(im, borrar)
    bb = bbox_tinta(im, medir, fondo)
    if color is not None:
        nuevo = [(r[0], r[1], color, r[3] if len(r) > 3 else {}) for r in nuevo]
        viejo = [(r[0], r[1], color, r[3] if len(r) > 3 else {}) for r in viejo]
    if size is None:
        s = 30.0
        while s > 6 and (bbox_render(viejo, s)[2] - bbox_render(viejo, s)[0]) > (bb[2] - bb[0]):
            s -= 0.25
    else:
        s = size
    rb = bbox_render(viejo, s)
    base = bb[1] - rb[1] + dy
    x0 = bb[0] - rb[0]
    d = ImageDraw.Draw(im)
    d.rectangle(borrar, fill=fondo)
    s2 = s
    if maxw:
        while ancho(nuevo, s2) > maxw and s2 > 6:
            s2 -= 0.25
    if align == 'l':
        escribir(d, x0 + dx, base, nuevo, s2)
    elif align == 'm':
        cx = (bb[0] + bb[2]) / 2 + dx
        escribir(d, cx, base, nuevo, s2, anchor='m')
    elif align == 'r':
        escribir(d, bb[2] + dx, base, nuevo, s2, anchor='r')
    return s, s2, base
