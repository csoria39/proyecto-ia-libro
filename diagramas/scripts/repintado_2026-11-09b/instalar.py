# Instala las imágenes corregidas: guarda la versión anterior en originales/ y genera las imágenes del libro
# (giradas 90° salvo 3.1, 4.1, 4.2 y 7.4), con los nombres de word/media/ del .docx.
# Uso: python3 instalar.py <carpeta con las corregidas, Diagrama n → n.png> <carpeta media> <sufijo, p. ej. 2026-11-09b>
import os, shutil, sys
from PIL import Image
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
NUEVAS, MED, SUF = sys.argv[1], sys.argv[2], sys.argv[3]
M = {'2.1': 1, '3.1': 2, '4.1': 3, '4.2': 4, '7.1': 6, '7.2': 7, '7.3': 8, '7.4': 5, '9.1': 9, '10.1': 10, '10.2': 11,
     '11.1': 12, '11.2': 13, '12.1': 14, '13.1': 15, '17.1': 16, '20.1': 18, '20.2': 19, 'G.1': 17}
ROT0 = {'3.1', '4.1', '4.2', '7.4'}
for f in sorted(os.listdir(NUEVAS)):
    n = f[:-4]
    dst = f'{R}/Diagrama_{n}.png'
    prev = Image.open(dst); modo = prev.mode
    shutil.copy2(dst, f'{R}/originales/Diagrama_{n}_antes_{SUF}.png')
    nuevo = Image.open(os.path.join(NUEVAS, f)).convert('RGB')
    def guarda(im, p):
        if modo == 'P':
            im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        im.save(p, optimize=True)
    guarda(nuevo, dst)
    rep = Image.open(dst).convert('RGB')
    med = rep if n in ROT0 else rep.rotate(90, expand=True)
    guarda(med, f'{MED}/image{M[n]}.png')
    print(n, modo, os.path.getsize(dst), os.path.getsize(f'{MED}/image{M[n]}.png'))
