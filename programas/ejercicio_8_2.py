# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
from collections import Counter
 
texto = 'la red neuronal aprende'
 
# Contar todos los bigramas
bigramas = Counter(texto[i:i+2] for i in range(len(texto)-1))
total    = sum(bigramas.values())
 
print(f'Texto: "{texto}"')
print(f'Total de bigramas: {total}\n')
print('Top 10 bigramas:')
print(f"{'Bigrama':>10}  {'Frec.':>8}  {'Prob.':>8}  Barra")
print('-' * 45)
 
for bg, freq in bigramas.most_common(10):
    prob  = freq / total
    barra = '█' * int(prob * 200)
    print(f'{repr(bg):>10}  {freq:>8}  {prob:>8.4f}  {barra}')
 
bg_max, f_max = bigramas.most_common(1)[0]
empates = [bg for bg, f in bigramas.items() if f == f_max]
print(f'\nBigrama mas frecuente: {bg_max!r} ({f_max} veces)'
      + (f'; empatan {len(empates)}: {empates}' if len(empates) > 1 else ''))
print()
 
# Probabilidad condicional: P(siguiente | actual)
print('Probabilidad de que despues de a venga p:')
p_ap = bigramas.get('ap', 0) / sum(v for k,v in bigramas.items() if k[0]=='a')
print(f'  P("p" | "a") = {p_ap:.4f}')
