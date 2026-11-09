import os, shutil
from PIL import Image
R = '/home/claude/proyecto-ia-libro/diagramas'
MED = '/home/claude/v18/media_opt'
M = {'2.1': 1, '3.1': 2, '4.1': 3, '4.2': 4, '7.1': 6, '7.2': 7, '7.3': 8, '7.4': 5, '9.1': 9, '10.1': 10, '10.2': 11,
     '11.1': 12, '11.2': 13, '12.1': 14, '13.1': 15, '17.1': 16, '20.1': 18, '20.2': 19, 'G.1': 17}
ROT0 = {'3.1', '4.1', '4.2', '7.4'}
for f in sorted(os.listdir('/tmp/claude-0/rv/out')):
    n = f[:-4]
    dst = f'{R}/Diagrama_{n}.png'
    prev = Image.open(dst); modo = prev.mode
    shutil.copy2(dst, f'{R}/originales/Diagrama_{n}_antes_2026-11-09b.png')
    nuevo = Image.open(f'/tmp/claude-0/rv/out/{f}').convert('RGB')
    def guarda(im, p):
        if modo == 'P':
            im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        im.save(p, optimize=True)
    guarda(nuevo, dst)
    rep = Image.open(dst).convert('RGB')
    med = rep if n in ROT0 else rep.rotate(90, expand=True)
    guarda(med, f'{MED}/image{M[n]}.png')
    print(n, modo, os.path.getsize(dst), os.path.getsize(f'{MED}/image{M[n]}.png'))
