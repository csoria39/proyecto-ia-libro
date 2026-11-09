"""Paginación del libro: títulos que no se quedan solos al pie y saltos de página que no crean páginas en blanco.
1. Estilos Título 1-4: «conservar con el siguiente» (keepNext).
2. Barras de título de programa (fondo 1A3A6B) y títulos de código «📄 …»: keepNext; también la primera línea de
   código tras un título «📄», para que el título nunca quede al pie con una sola línea debajo; y los separadores
   mínimos (interlineado exacto de 1 pt) que van entre un título y su contenido.
3. Párrafos vacíos que solo contienen un salto de página: se quitan y el párrafo siguiente pasa a llevar
   «salto de página anterior» (pageBreakBefore). Así, si el texto anterior llena la página, no queda una en blanco.
"""
import re, sys
doc, sty = sys.argv[1], sys.argv[2]
st = open(sty).read()
n_st = 0
for sid in ('Ttulo1', 'Ttulo2', 'Ttulo3', 'Ttulo4'):
    m = re.search(r'(<w:style [^>]*w:styleId="%s"[^>]*>.*?<w:pPr>)' % sid, st, re.S)
    if m and '<w:keepNext/>' not in st[m.end():m.end() + 200]:
        st = st[:m.end()] + '<w:keepNext/>' + st[m.end():]; n_st += 1
open(sty, 'w').write(st)

s = open(doc).read()
PAR = re.compile(r'<w:p(?=[ >])(?:(?!</w:p>).)*</w:p>|<w:tbl>.*?</w:tbl>', re.S)
items = [m for m in PAR.finditer(s)]
def texto(p): return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p)).strip()
ORDEN = ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore']
def poner(p, tag):
    """Añade <w:tag/> al pPr del párrafo respetando el orden del esquema (pStyle, keepNext, keepLines,
    pageBreakBefore, …)."""
    if '<w:%s/>' % tag in p: return p
    m = re.match(r'^<w:p(?: [^>]*)?>', p)
    head, rest = m.group(0), p[m.end():]
    if not rest.startswith('<w:pPr>'):
        return head + '<w:pPr><w:%s/></w:pPr>' % tag + rest
    pos = len('<w:pPr>')
    for prev in ORDEN[:ORDEN.index(tag)]:
        mm = re.match(r'<w:%s(?: [^>]*)?/>' % prev, rest[pos:])
        if mm: pos += mm.end()
    return head + rest[:pos] + '<w:%s/>' % tag + rest[pos:]
def con_keep(p):
    if p.startswith('<w:tbl') or '<w:keepNext/>' in p or '<w:sectPr' in p: return p
    return poner(p, 'keepNext')
out = []; last = 0; n_t = n_c = n_b = 0
pending_break = False
i = 0
segs = []
for m in items:
    segs.append([m.start(), m.end(), m.group(0)])
k = 0
while k < len(segs):
    a, b, p = segs[k]
    t = texto(p)
    if p.startswith('<w:p') and re.fullmatch(r'<w:p(?: [^>]*)?>(?:<w:pPr>(?:(?!</w:pPr>).)*</w:pPr>)?<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:br w:type="page"/></w:r></w:p>', p, re.S) \
            and '<w:sectPr' not in p and k + 1 < len(segs) and segs[k + 1][2].startswith('<w:p'):
        segs[k][2] = ''
        q = segs[k + 1][2]
        q = poner(q, 'pageBreakBefore')
        segs[k + 1][2] = q; n_b += 1
        k += 1; continue
    if p.startswith('<w:tbl') and len(re.findall(r'<w:tr[ >]', p)) <= 2 and ('📄' in t or t.startswith('PROGRAMA')) and len(t) < 200:
        # barra de título de programa hecha con una tabla de una o dos filas: todos sus párrafos y el separador vacío de
        # detrás se conservan con lo siguiente
        segs[k][2] = re.sub(r'<w:p(?=[ >])(?:(?!</w:p>).)*</w:p>', lambda mm: con_keep(mm.group(0)), p, flags=re.S)
        n_t += 1
        j = k + 1
        while j < len(segs) and segs[j][2].startswith('<w:p') and not texto(segs[j][2]):
            segs[j][2] = con_keep(segs[j][2]); j += 1
        k += 1; continue
    es_titulo = p.startswith('<w:p') and ('w:fill="1A3A6B"' in p or t.startswith('📄'))
    if es_titulo:
        segs[k][2] = con_keep(segs[k][2]); n_t += 1
        if t.startswith('📄'):
            j = k + 1
            while j < len(segs) and segs[j][2].startswith('<w:p') and 'w:line="20"' in segs[j][2] and not texto(segs[j][2]):
                segs[j][2] = con_keep(segs[j][2]); j += 1
            # las dos primeras líneas de código con texto (y las líneas en blanco que haya entre ellas)
            vistas = 0
            while j < len(segs) and vistas < 2 and segs[j][2].startswith('<w:p') and not texto(segs[j][2]).startswith('📄'):
                if texto(segs[j][2]): vistas += 1
                if vistas < 2: segs[j][2] = con_keep(segs[j][2])
                j += 1
            n_c += 1
    k += 1
res = []; pos = 0
for a, b, p in segs:
    res.append(s[pos:a]); res.append(p); pos = b
res.append(s[pos:])
open(doc, 'w').write(''.join(res))
print('estilos', n_st, '| títulos', n_t, '| primeras líneas de código', n_c, '| saltos convertidos', n_b)
