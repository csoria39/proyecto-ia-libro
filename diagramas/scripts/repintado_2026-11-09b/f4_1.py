import sys, numpy as np, cv2, math
from PIL import Image, ImageDraw
SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert('RGB')
a = np.asarray(im).astype(int)
m = np.zeros(a.shape[:2], bool)
sub = a[1280:1400, 598:930]; r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
m[1280:1400, 598:930] = (b - g > 60) & (r - g > 20)
yy, xx = np.ogrid[:a.shape[0], :a.shape[1]]
m |= (xx - 710) ** 2 + (yy - 1378.5) ** 2 <= 8 ** 2
mask = cv2.dilate((m * 255).astype(np.uint8), np.ones((3, 3), np.uint8))
bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
im = Image.fromarray(cv2.cvtColor(cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB))
E = 4; X0, Y0, X1, Y1 = 590, 1285, 930, 1400
capa = Image.new('RGBA', ((X1 - X0) * E, (Y1 - Y0) * E), (0, 0, 0, 0)); d = ImageDraw.Draw(capa)
P = lambda x, y: ((x - X0) * E, (y - Y0) * E)
Lmin, Lini, x0 = 1386.0, 1319.0, 607.0
k = math.log((Lini - Lmin) / (1378.5 - Lmin)) / (710 - x0)
pts = [P(x, Lmin + (Lini - Lmin) * math.exp(-k * (x - x0))) for x in np.linspace(x0, 916, 300)]
COL = (52, 18, 128)
d.line(pts, fill=COL, width=int(2.6 * E), joint='curve')
for p in (pts[0], pts[-1]):
    rr = 1.3 * E; d.ellipse((p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr), fill=COL)
c = P(710, 1378.5)
d.ellipse((c[0] - 5.6 * E, c[1] - 5.6 * E, c[0] + 5.6 * E, c[1] + 5.6 * E), fill=(14, 30, 125))
d.ellipse((c[0] - 3.6 * E, c[1] - 3.6 * E, c[0] + 3.6 * E, c[1] + 3.6 * E), fill=(158, 198, 224))
capa = capa.resize((X1 - X0, Y1 - Y0), Image.LANCZOS)
im.paste(capa, (X0, Y0), capa)
im.save(DST); print('ok', k)
