"""Índice alfabético: páginas que cambian al repaginar (los títulos de sección que bajaban solos al pie pasan a la
página siguiente)."""
import sys
doc = sys.argv[1]; s = open(doc).read()
C = [(' 301-302<', ' 302<'), (' 65, 74-75, 77, 79, 83, 85, 87<', ' 65, 75, 77, 79, 83, 85, 87<'),
     (' 82, 90-91, 93, 149-150, 182, 344<', ' 83, 90-91, 93, 149-150, 182, 344<'),
     (' 119, 151, 251, 264-265, 275, 282-283<', ' 119, 151, 251, 265, 275, 282-283<'),
     (' 251, 259-260, 262, 271, 273, 353<', ' 251, 260, 262, 271, 273, 353<'),
     (' 215-216, 221, 243, 304<', ' 215-216, 221, 243, 305<'),
     (' 251-252, 270, 353<', ' 251-252, 271, 353<'),
     (' 14, 17, 134, 136, 154, 202, 355, 371<', ' 14, 18, 134, 136, 154, 202, 355, 371<')]
for a, b in C:
    n = s.count(a)
    if n != 1: raise SystemExit(f'{a!r}: {n} apariciones')
    s = s.replace(a, b)
open(doc, 'w').write(s); print('índice: 8 entradas')
