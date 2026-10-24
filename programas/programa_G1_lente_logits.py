# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA G-1 — La lente de logits: ver pensar a GPT-2
# Capítulo 16, sección 16.5
#
# La vía residual de GPT-2 tiene siempre 768 números por token.
# Al final, la LayerNorm final y wteᵀ la convierten en una
# predicción. ¿Y si aplicamos esa misma conversión a la salida
# de CADA bloque, no solo del último? Veremos cómo se va formando
# la predicción capa a capa (nostalgebraist, 2020).
# Requisitos: programa_8bis_gpt2_real.py y la carpeta gpt2/.
# =======================================================
import numpy as np

fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])     # pesos, tokenizador y forward()


def lente(frase, objetivo):
    """Predicción tras cada bloque para el token que sigue a `frase`."""
    ids = codificar(frase)
    obj = codificar(objetivo)[0]
    _, (_, _, cache) = forward(np.array([ids]))
    # La entrada del bloque 1 (embeddings) y la salida de cada bloque
    salidas = [params['wte'][ids] + params['wpe'][:len(ids)]] + [c['X'][0] for c in cache]
    print(f'\n«{frase}» → ¿«{objetivo}»?')
    print('  capa          más probable        prob.   P(objetivo)  puesto')
    for k, X in enumerate(salidas):
        xf, _ = layer_norm_fwd(X[-1], params['lnf_g'], params['lnf_b'])   # última posición
        probs = softmax((xf @ params['wte'].T).astype(np.float64))
        mejor = int(np.argmax(probs))
        puesto = int((probs > probs[obj]).sum()) + 1
        capa = 'embeddings' if k == 0 else f'bloque {k:2d}'
        print(f'  {capa:12s}  {decodificar([mejor])!r:18s}  {probs[mejor]:.3f}   '
              f'{probs[obj]:.4f}     {puesto:>6d}')


lente('The capital of France is', ' Paris')
lente('The Eiffel Tower is located in the city of', ' Paris')
lente('One, two, three, four,', ' five')
lente('I enjoy walking with my cute', ' dog')
