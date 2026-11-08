# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Cifras del panel «El algoritmo BPE — paso a paso» del Diagrama 9.1.
Usa las funciones del Programa BPE-1 (contar_pares, fusionar_par, texto_a_vocab_inicial) sobre el corpus del
diagrama, {'bajo': 6, 'bajar': 3, 'caja': 4}, y muestra cada fusión y la segmentación tras cada una.
Se ejecuta desde la carpeta programas/ (importa programa_BPE1_entrenador.py)."""
import sys, os, io, contextlib
sys.path.insert(0, os.getcwd())
with contextlib.redirect_stdout(io.StringIO()):          # BPE-1 ejecuta su demo al importarse
    from programa_BPE1_entrenador import texto_a_vocab_inicial, contar_pares, fusionar_par

corpus = ' '.join(['bajo'] * 6 + ['bajar'] * 3 + ['caja'] * 4)
vocab = texto_a_vocab_inicial(corpus)
print('Vocabulario inicial:', sorted({c for p in vocab for c in p}))
for n in range(1, 11):
    pares = contar_pares(vocab)
    top = sorted(pares.items(), key=lambda kv: -kv[1])[:4]
    mejor = max(pares, key=pares.get)
    vocab = fusionar_par(vocab, mejor)
    print(f'\nFusión {n}: {mejor[0]} + {mejor[1]} -> {"".join(mejor)}  (frecuencia {pares[mejor]})')
    print('   pares más frecuentes antes de fusionar:', ', '.join(f"{a}+{b}:{f}" for (a, b), f in top))
    for pal, f in vocab.items():
        print(f'   {"".join(pal).replace("</w>", ""):6s} x{f}: ' + ' '.join(f'[{t}]' for t in pal))
