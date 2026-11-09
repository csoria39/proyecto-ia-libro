import sys, numpy as np, cv2
sys.path.insert(0, '.')
from pintar import *
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB')
a = np.asarray(im).astype(int)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
gm = (g - r > 40) & (g - b > 40); gm[:, :290] = 0; gm[:, 480:] = 0; gm[:470] = 0; gm[625:] = 0
om = (r - b > 80) & (r > 180); om[:, :535] = 0; om[:, 720:] = 0; om[:470] = 0; om[625:] = 0
mask = cv2.dilate(((gm | om) * 255).astype(np.uint8), np.ones((5, 5), np.uint8))
bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
im = Image.fromarray(cv2.cvtColor(cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB))

E = 4
def trazo(pts, col, ancho, grueso_box):
    x0, y0, x1, y1 = grueso_box
    capa = Image.new('L', ((x1 - x0) * E, (y1 - y0) * E), 0)
    dc = ImageDraw.Draw(capa)
    P = [((x - x0) * E, (y - y0) * E) for x, y in pts]
    dc.line(P, fill=255, width=int(ancho * E), joint='curve')
    for p in (P[0], P[-1]):
        rr = ancho * E / 2; dc.ellipse((p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr), fill=255)
    capa = capa.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    im.paste(Image.new('RGB', capa.size, col), (x0, y0), capa)

# eje horizontal de tanh (línea gris en f = 0) restaurado
trazo([(545, 555), (699, 555)], (160, 161, 170), 1.2, (520, 545, 710, 565))
# eje z de ReLU bajo el tramo plano
trazo([(299, 607.5), (468, 607.5)], (110, 110, 110), 1.2, (280, 598, 480, 618))
# ReLU: bisagra en z = 0 (x = 382), pendiente 1 en unidades del gráfico, hasta z = f = 4
X = lambda z: 382 + 20.75 * z; Y = lambda f: 607.5 - 25.9 * f
trazo([(X(-4), Y(0)), (X(0), Y(0)), (X(4), Y(4))], (20, 95, 28), 2.6, (290, 495, 480, 618))
# tanh verdadera
X = lambda z: 622.5 + 19.5 * z; Y = lambda f: 555 - 51.5 * f
pts = [(X(z), Y(np.tanh(z))) for z in np.linspace(-4, 4, 161)]
trazo(pts, (234, 106, 24), 2.6, (535, 495, 710, 615))

# rangos de softmax: intervalo abierto
c = tinta_de(im, (821, 663, 890, 676))
reemplazar(im, (818, 660, 893, 679), (821, 663, 890, 676), [R('Rango: [0, 1]', 'fira600', c)], [R('Rango: (0, 1)', 'fira600', c)], align='m')
c = (25, 25, 25)
reemplazar(im, (790, 855, 945, 876), (817, 858, 919, 872), [R('[0, 1] (suma = 1)', 'mathd', c)], [R('(0, 1) (suma = 1)', 'mathd', c)], align='m', size=13.5, maxw=200)
# hoja del árbol: salida (no multiclase, no binaria) = regresión
c = tinta_de(im, (395, 1380, 454, 1390))
reemplazar(im, (392, 1377, 458, 1393), (395, 1380, 454, 1392), [R('Usa TANH', 'fira600', c)], [R('Salida lineal', 'fira600', c)], align='m', maxw=140)
c = tinta_de(im, (365, 1399, 487, 1410))
reemplazar(im, (362, 1396, 490, 1413), (365, 1399, 487, 1412), [R('(o ReLU según el caso)', 'fira600', c)], [R('(regresión: sin activación)', 'fira600', c)], align='m', maxw=150)
im.save(DST)
print('ok')
