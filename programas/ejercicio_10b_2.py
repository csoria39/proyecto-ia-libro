# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import math
from collections import Counter
 
def ngramas(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
 
def bleu_multi_ref(candidato, referencias, n_max=4):
    '''BLEU con multiples referencias validas'''
    cand = candidato.lower().split()
    refs = [r.lower().split() for r in referencias]
    
    precisiones = []
    for n in range(1, n_max+1):
        cand_ng = Counter(ngramas(cand, n))
        # Para cada n-grama, tomar el MAXIMO conteo entre todas las referencias
        max_ref_ng = Counter()
        for ref in refs:
            ref_ng = Counter(ngramas(ref, n))
            for ng, cnt in ref_ng.items():
                max_ref_ng[ng] = max(max_ref_ng[ng], cnt)
        
        coincidencias = sum(min(cnt, max_ref_ng[ng]) for ng, cnt in cand_ng.items())
        total = max(len(cand) - n + 1, 0)
        precisiones.append(coincidencias/total if total > 0 else 0.0)
    
    if any(p==0 for p in precisiones):
        return 0.0
    log_avg = sum(math.log(p) for p in precisiones) / n_max
    # Brevity penalty: usar la referencia mas cercana en longitud
    len_ref_cercana = min(refs, key=lambda r: abs(len(r)-len(cand)))
    bp = math.exp(min(0, 1 - len(len_ref_cercana)/max(len(cand),1)))
    return bp * math.exp(log_avg)
 
candidato = 'el modelo aprende rapido de los datos'
referencias = [
    'el modelo aprende rapidamente de los datos',
    'el modelo aprende rapido con los datos',
    'la red aprende velozmente de la informacion',
]
 
score = bleu_multi_ref(candidato, referencias, n_max=2)
print(f'Candidato: "{candidato}"')
print(f'Referencias: {len(referencias)} variantes validas')
print(f'BLEU-2 (multi-referencia): {score:.4f}')
print()
print('Con multiples referencias, el BLEU suele ser MAS ALTO')
print('porque hay mas oportunidades de coincidir con alguna variante valida.')
