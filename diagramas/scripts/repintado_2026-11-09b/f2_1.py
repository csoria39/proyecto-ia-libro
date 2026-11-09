# Diagrama 2.1: barras de softmax 0.628 / 0.231 / 0.140 (BASE = eje en y=847.5, U = 80 px por unidad).
# Uso: python3 f2_1.py entrada.png salida.png
import sys
from PIL import Image, ImageDraw
im=Image.open(sys.argv[1]).convert('RGB'); px=im.load(); d=ImageDraw.Draw(im)
col=px[956,840]; bg=px[970,800]
print(col,bg)
BASE=847.5; U=80
for (x0,x1),v in zip([(949,963),(979,993),(1009,1023)],[0.628,0.231,0.140]):
    d.rectangle((x0,770,x1-1,847),fill=bg)
    top=BASE-v*U
    # subpixel top edge
    d.rectangle((x0,int(top)+1,x1-1,847),fill=col)
    f=top-int(top); y=int(top)
    for x in range(x0,x1):
        b=px[x,y]; px[x,y]=tuple(int(round(b[i]*f+col[i]*(1-f))) for i in range(3))
im.save(sys.argv[2])
