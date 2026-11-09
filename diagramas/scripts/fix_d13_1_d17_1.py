# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Diagramas 13.1 y 17.1 regenerados por el autor (4-11-2026) con los prompts nuevos: tres erratas de la
ilustración, corregidas sobre la imagen (originales en diagramas/originales/*_autor_2026-11-04.png).

13.1  (segunda versión del autor, d13_1_autor_2026-11-04b.png, con la cabecera corregida)
      · Paso 6: «46 probabiidades que suman 1» → «46 probabilidades que suman 1».
      · Fórmula de la pérdida: «P(y₁ | x₁ … x₁)» → «P(yₜ | x₁ … xₜ)» (los subíndices t, como en 13.3).
      · Leyenda: «Trantformer» y «Geneneción», y los nombres de los colores en inglés → leyenda en español,
        como la del Diagrama 17.1.
17.1  · «un modelo de solo 57 000 parámetros» → «58 000», como en el texto de 17.x (el Programa RZ-2 tiene
        57 696; el prompt decía 57 000 por error)."""
from PIL import Image, ImageDraw, ImageFont

D = 'originales/'
INTER_SB = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
INTER_M = '/usr/share/fonts/opentype/inter/Inter-Medium.otf'
MONO_B = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
SANS = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def texto_en(draw, x, y_centro, s, fuente, color):
    b = draw.textbbox((0, 0), s, font=fuente)
    draw.text((x - b[0], y_centro - (b[1] + b[3]) / 2), s, font=fuente, fill=color)


# ---------------------------------------------------------------- 13.1
im = Image.open(D + 'd13_1_autor_2026-11-04b.png').convert('RGB')
dr = ImageDraw.Draw(im)
# paso 6: la línea entera, centrada donde estaba y con el mismo ancho
dr.rectangle((104, 625, 313, 641), fill=(224, 244, 221))
linea = '46 probabilidades que suman 1'
tam = 8
while ImageFont.truetype(INTER_M, tam + 1).getlength(linea) <= 199:
    tam += 1
f6 = ImageFont.truetype(INTER_M, tam)
b = dr.textbbox((0, 0), linea, font=f6)
dr.text((208 - (b[0] + b[2]) / 2, 637 - b[3] + (b[3] - dr.textbbox((0, 0), '4', font=f6)[3])), linea, font=f6, fill=(0, 0, 0))
fondo_formula, tinta = (252, 236, 234), (38, 31, 30)
sub = ImageFont.truetype(MONO_B, 9)
for x0, x1 in ((685, 691), (765, 771)):            # subíndice de y y del último x
    dr.rectangle((x0, 477, x1, 485), fill=fondo_formula)
    b = dr.textbbox((0, 0), 't', font=sub)
    dr.text(((x0 + x1) / 2 - (b[0] + b[2]) / 2, 484 - b[3]), 't', font=sub, fill=tinta)
navy, blanco = (18, 48, 88), (255, 255, 255)
dr.rectangle((1276, 853, 1598, 929), fill=navy)
f = ImageFont.truetype(INTER_SB, 12)
for yc, s in zip((859, 871.5, 884.5, 897.5, 910, 922.5),
                 ('Azul = Entrada / tokenizador / embeddings', 'Verde azulado = Bloques Transformer',
                  'Verde = Salida / texto generado', 'Naranja = Pérdida / entrenamiento',
                  'Morado = Generación / muestreo', 'Dorado = Parámetros / formas')):
    texto_en(dr, 1280, yc, s, f, blanco)
im.save('Diagrama_13.1.png', optimize=True)

# ---------------------------------------------------------------- 17.1
im = Image.open(D + 'd17_1_autor_2026-11-04.png').convert('RGB')
dr = ImageDraw.Draw(im)
fondo, tinta = (242, 251, 241), (20, 22, 20)
dr.rectangle((711, 599, 719, 611), fill=fondo)      # el «7» de «57 000»
f = ImageFont.truetype(SANS, 13)
b = dr.textbbox((0, 0), '8', font=f)
dr.text((715 - (b[0] + b[2]) / 2, 609.5 - b[3]), '8', font=f, fill=tinta, stroke_width=0)
dr.text((715.4 - (b[0] + b[2]) / 2, 609.5 - b[3]), '8', font=f, fill=tinta)   # un poco más de peso, como sus vecinas
im.save('Diagrama_17.1.png', optimize=True)
print('ok')
