import numpy as np
from PIL import Image, ImageDraw
def analiza(im, box, umbral=330):
    a=np.asarray(im.crop(box).convert('RGB')).astype(int); dark=a.sum(2)<umbral
    filas=dark.sum(1); cap=int(np.argmax(filas>filas.max()*0.35))
    cuerpo=dark[cap+2:]; cols=cuerpo.any(0)
    segs=[];x=0
    while x<len(cols):
        if cols[x]:
            s=x
            while x<len(cols) and cols[x]: x+=1
            segs.append((s+box[0],x-1+box[0]))
        x+=1
    return cap+box[1], segs, filas
def arregla(im, box, iO, umbral=330, alto=None, color=None):
    cap,segs,_=analiza(im,box,umbral)
    x0,x1=segs[iO]
    a=np.asarray(im.convert('RGB')).astype(int)
    # fondo: mediana de la fila superior del box
    fondo=tuple(int(v) for v in np.median(a[box[1]:cap-1, box[0]:box[2]].reshape(-1,3),axis=0))
    if color is None:
        reg=a[cap+4:box[3], x0:x1+1].reshape(-1,3); color=tuple(int(v) for v in reg[reg.sum(1).argsort()[:20]].mean(0))
    d=ImageDraw.Draw(im)
    d.rectangle((box[0],box[1],box[2],cap-1),fill=fondo)
    # tilde aguda centrada sobre la O
    h=alto or max(4,int((x1-x0)*0.32)); w=(x1-x0)*0.38; cx=(x0+x1)/2+1
    E=8; W=int((x1-x0+20)*E); H=int((h+4)*E)
    m=Image.new('L',(W,H),0); dm=ImageDraw.Draw(m)
    ox=x0-10; oy=cap-h-2
    P=lambda x,y:((x-ox)*E,(y-oy)*E)
    yb=cap-1.5; yt=yb-h
    dm.polygon([P(cx-w*0.5,yb),P(cx-w*0.5+w*0.36,yb),P(cx+w*0.5,yt),P(cx+w*0.5-w*0.42,yt)],fill=255)
    m=m.resize((W//E,H//E),Image.LANCZOS)
    col=Image.new('RGB',m.size,color); im.paste(col,(int(ox),int(oy)),m)
    return cap,segs,fondo,color
