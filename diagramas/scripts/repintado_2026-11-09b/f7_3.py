import sys; sys.path.insert(0, '.')
from pintar import *
import numpy as np
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); d = ImageDraw.Draw(im)
a = np.asarray(im).astype(int)

# 1. comilla suelta delante de «Apagar»
f = fondo_de(im, (530, 38, 548, 62))
d.rectangle((534, 38, 541, 52), fill=f)

# 2. rótulos h de los pasos 1-3, todos con el mismo estilo (sans cursiva, blanco)
BL = (255, 255, 255)
for x, y, k in ((417.5, 267.5, 1), (417.5, 372.5, 3), (593.5, 320, 2), (593.5, 424, 4), (770, 267.5, 1), (770, 319, 2)):
    xi, yi = int(round(x)), int(round(y))
    col = tuple(int(v) for v in np.median(a[yi - 3:yi + 4, xi - 12:xi - 9].reshape(-1, 3), axis=0))
    E = 4; R0 = 11.2
    capa = Image.new('L', (40 * E, 40 * E), 0); ImageDraw.Draw(capa).ellipse(((20 - R0) * E, (20 - R0) * E, (20 + R0) * E, (20 + R0) * E), fill=255)
    capa = capa.resize((40, 40), Image.LANCZOS)
    im.paste(Image.new('RGB', (40, 40), col), (int(x) - 20, int(y) - 20), capa)
    escribir(d, x - 0.5, y + 5, [R('h', 'fira600i', BL), R(str(k), 'fira600', BL, sub=True)], 15, anchor='m')

# 3. corrección de escala, líneas 1-2
LF = sys.argv[3] if len(sys.argv) > 3 else 'free'
LS = float(sys.argv[4]) if len(sys.argv) > 4 else 14
f = fondo_de(im, (880, 414, 1186, 450))
d.rectangle((880, 414, 1186, 450), fill=f)
escribir(d, 883, 429, [R('Durante el entrenamiento, solo una fracción (1−p)', LF, (0, 0, 0))], LS)
escribir(d, 883, 448, [R('de las neuronas estaba activa en promedio.', LF, (0, 0, 0))], LS)
d.rectangle((880, 452, 1186, 468), fill=f)
escribir(d, 883, 465, [R('En inferencia TODAS están activas.', LF, (0, 0, 0))], LS)

# 4. fila «Escalado» con el tamaño de las demás filas
for box in ((867, 708, 974, 738), (977, 708, 1089, 738), (1092, 708, 1200, 738)):
    d.rectangle(box, fill=fondo_de(im, box))
for x, g in ((975, 173), (976, 205), (1090, 188), (1091, 171)):
    d.line((x, 707, x, 739), fill=(g, g, g))
S = 11.6
escribir(d, 873, 728, [R('Escalado', 'fira500', (0, 0, 0))], S)
escribir(d, 1032.5, 728, [R('activas × 1/(1−p)', 'fira500', (0, 0, 0))], S, anchor='m')
escribir(d, 1146, 728, [R('ninguno', 'fira500', (0, 0, 0))], S, anchor='m')

# 5. bloque de código PyTorch reescrito
f = fondo_de(im, (1236, 614, 1604, 785))
d.rectangle((1236, 614, 1604, 785), fill=f)
codigo = ['import torch', 'import torch.nn as nn', None,
          'class MiRed(nn.Module):',
          '    def __init__(self):',
          '        super().__init__()',
          '        self.fc1 = nn.Linear(4, 8)',
          '        self.drop = nn.Dropout(p=0.5)',
          '        self.fc2 = nn.Linear(8, 1)', None,
          '    def forward(self, x):',
          '        x = torch.relu(self.fc1(x))',
          '        x = self.drop(x)   # solo en train',
          '        return self.fc2(x)']
paso = (782 - 626) / 11.6
y = 626
for l in codigo:
    if l is None:
        y += paso * 0.3; continue
    escribir(d, 1243, y, [R(l, 'mono_r', (35, 35, 35))], 10.4)
    y += paso
im.save(DST); print('ok', y)

# 6. «Pérdida» (eje y de los dos gráficos), con tilde
im = Image.open(DST).convert('RGB')
for x0 in (1234, 1430):
    caja = (x0, 238, x0 + 14, 285)
    fo = fondo_de(im, caja); bb = bbox_tinta(im, caja, fo)
    t = Image.new('RGB', (80, 30), fo); dt = ImageDraw.Draw(t)
    escribir(dt, 5, 20, [R('Pérdida', 'fira600', (0, 0, 0))], 11.3)
    tb = bbox_tinta(t, (0, 0, 80, 30), fo)
    t = t.crop((tb[0] - 1, tb[1] - 1, tb[2] + 2, tb[3] + 2)).rotate(90, expand=True)
    ImageDraw.Draw(im).rectangle((bb[0] - 1, bb[1] - 1, bb[2] + 1, bb[3] + 1), fill=fo)
    # alinear la base de las letras (lado derecho tras rotar) y el centro vertical
    cx = (bb[1] + bb[3]) / 2
    im.paste(t, (bb[2] + 2 - t.width, int(round(cx - t.height / 2))))
im.save(DST)
print('pérdida ok')
