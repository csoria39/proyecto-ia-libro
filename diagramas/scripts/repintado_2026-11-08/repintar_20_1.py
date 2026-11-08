# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pintar import *
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SRC = os.path.join(BASE, 'originales', 'Diagrama_20.1_antes_repintado_2026-11-08.png')
im = Image.open(SRC).convert('RGB')
d = ImageDraw.Draw(im)
GRIS = (60, 60, 60)

# 1. privacidad condicionada
c = tinta_de(im, (39, 773, 290, 786))
reemplazar(im, (30, 769, 300, 790), (30, 769, 300, 790), [R('Privacidad total — datos en tu máquina', 'fira600', c)],
           [R('Privacidad: modelo descargado, sin funciones en la nube', 'fira600', c)], align='m', maxw=248)

# 2. gráfico de vectores = tabla Top-3 (+ uno descartado)
ZONA = (806, 398, 1103, 589)
FONDO = (253, 253, 253)
E = 4
W, H = (ZONA[2] - ZONA[0]) * E, (ZONA[3] - ZONA[1]) * E
capa = Image.new('RGB', (W, H), FONDO)
dc = ImageDraw.Draw(capa)
O = (818, 572)
P = lambda x, y: ((x - ZONA[0]) * E, (y - ZONA[1]) * E)


def flecha(ang, r, col, ancho=2.2):
    a = math.radians(ang)
    x1, y1 = O[0] + r * math.cos(a), O[1] - r * math.sin(a)
    dc.line([P(*O), P(x1, y1)], fill=col, width=int(ancho * E))
    L, w = 10, 4.5
    bx, by = x1 - L * math.cos(a), y1 + L * math.sin(a)
    nx, ny = math.sin(a), math.cos(a)
    dc.polygon([P(x1, y1), P(bx + w * nx, by + w * ny), P(bx - w * nx, by - w * ny)], fill=col)
    return x1, y1


def arco(r, a0, a1, col):
    pasos = 40
    for i in range(pasos):
        if i % 2:
            continue
        t0 = a0 + (a1 - a0) * i / pasos; t1 = a0 + (a1 - a0) * (i + 1) / pasos
        pts = [P(O[0] + r * math.cos(math.radians(t)), O[1] - r * math.sin(math.radians(t))) for t in (t0, t1)]
        dc.line(pts, fill=col, width=int(1.4 * E))


AZUL, VERDE, TEAL, NAR, ROJO = (0, 78, 192), (36, 114, 66), (13, 128, 120), (240, 120, 60), (216, 18, 18)
QANG = 78
vecs = [(0.87, VERDE), (0.74, TEAL), (0.61, NAR), (0.12, ROJO)]
dc.line([P(O[0], O[1]), P(O[0], 405)], fill=(20, 20, 20), width=int(1.6 * E))   # eje
flecha(QANG, 160, AZUL, 2.6)
for k, (sim, col) in enumerate(vecs):
    ang = QANG - math.degrees(math.acos(sim))
    flecha(ang, 150 if k < 3 else 135, col)
arco(48, QANG - math.degrees(math.acos(0.87)), QANG, (40, 40, 40))
arco(34, QANG - math.degrees(math.acos(0.12)), QANG, (40, 40, 40))
dc.ellipse([P(O[0] - 5, O[1] - 5), P(O[0] + 5, O[1] + 5)], fill=(20, 20, 20))
im.paste(capa.resize((ZONA[2] - ZONA[0], ZONA[3] - ZONA[1]), Image.LANCZOS), ZONA[:2])

# etiquetas
escribir(d, 846, 412, [R('q_vec', 'fira600', AZUL)], 14)
escribir(d, 900, 412, [R('pregunta: "devolucion contrato"', 'mono_r', (40, 40, 40))], 10.5)
etiquetas = [('v1_vec (chunk 1)', "'clausula devolucion 30 dias'", 'sim = 0.87 ✓ RECUPERADO', VERDE),
             ('v2_vec (chunk 2)', "'articulo 3: garantias'", 'sim = 0.74 ✓', TEAL),
             ('v3_vec (chunk 3)', "'politica de cambios'", 'sim = 0.61 ✓', NAR),
             ('v4_vec (chunk 4)', "'historia de la empresa'", 'sim = 0.12 ✗ descartado', ROJO)]
y = 438
sq = min(ajustar([R(q, 'mono_r')], 9.5, 136) for _, q, _, _ in etiquetas)
ss = min(ajustar([R(s, 'monob')], 9.5, 136) for _, _, s, _ in etiquetas)
for t, q, s, col in etiquetas:
    escribir(d, 962, y, [R(t, 'fira600', col)], 13)
    escribir(d, 962, y + 12, [R(q, 'mono_r', GRIS)], sq)
    escribir(d, 962, y + 24, [R(s, 'monob', col)], ss)
    y += 38

d.rectangle((1103, 448, 1106, 460), fill=FONDO)   # resto de la comilla final de la versión anterior

# 3. código del paso I
f = fondo_de(im, (392, 783, 704, 818))
d.rectangle((392, 783, 704, 818), fill=f)
codigo = ["prompt = f'''Usa solo esta información:", '{chunk_1}  {chunk_2}  {chunk_3}', "Pregunta: {pregunta}'''"]
for i, l in enumerate(codigo):
    escribir(d, 398, 793 + i * 10.5, [R(l, 'mono_r', (40, 40, 40))], 9.5)

# 4. letras A-D de la fase de indexación (la de consulta ya va de E a J)
COLL = (12, 80, 95)
for letra, x, yc in (('A', 440, 225), ('B', 427, 301), ('C', 417, 377), ('D', 407, 455)):
    escribir(d, x - 11, yc + 7, [R(letra, 'fira700', COLL)], 19, anchor='m')

im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'Diagrama_20.1.png'))
print('ok')
