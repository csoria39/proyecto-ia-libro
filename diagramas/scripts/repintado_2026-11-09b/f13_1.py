import sys; sys.path.insert(0, '.')
from pintar import *
from PIL import Image, ImageDraw, ImageChops
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB'); src = im.copy(); d = ImageDraw.Draw(im)
def pon(p, x, y):
    reg = im.crop((x, y, x + p.width, y + p.height)); im.paste(ImageChops.darker(reg, p), (x, y))
# «0.24 MB» → «0.49 MB»: el Programa 8 guarda 60 000 pesos en float64 (nano_llm_pesos.npz = 488 646 B)
d.rectangle((1443, 380, 1461, 392), fill=fondo_de(im, (1442, 379, 1462, 393)))
pon(src.crop((1452, 380, 1461, 393)), 1444, 380)     # «4» donde estaba el «2»
pon(src.crop((1447, 359, 1456, 372)), 1452, 380)     # «9» de «(93 frases)»
im.save(DST); print('ok')
