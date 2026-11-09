"""Comprueba el backward de un programa RZ con diferencias finitas en TODOS sus tensores.
Uso: python gradcheck.py programa.py 'marca de corte' 'expresión del lote' [args del programa]"""
import sys, io, contextlib
import numpy as np
prog, corte, lote_expr = sys.argv[1:4]
sys.argv = [prog] + sys.argv[4:]
src = open(prog, encoding='utf-8').read()
src = src[:src.index(corte)]
import os
if os.environ.get('SIN_SUELO'):
    src = src.replace('+ 1e-10)', ')')   # la derivada analítica es la de log(p), sin el suelo
G = {'__name__': 'gc'}
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, G)
params, fb = G['params'], G['forward_backward']
rng = np.random.RandomState(0)
for k, v in params.items():                 # pesos no triviales (γ ≠ 1, β ≠ 0, sesgos ≠ 0)
    v += rng.randn(*v.shape) * 0.05
tok, tgt, w = eval(lote_expr, G)
grads, _ = fb(tok, tgt, w)
peor, eps = 0.0, float(__import__("os").environ.get("EPS", "1e-5"))
print('Tensor      error relativo ||num − ana|| / (||num|| + ||ana||), como el Ejercicio 13.6')
for k, v in params.items():
    idx = [tuple(rng.randint(s) for s in v.shape) for _ in range(6)]
    num, ana = [], []
    for ix in idx:
        old = v[ix]
        v[ix] = old + eps; lp = fb(tok, tgt, w)[1]
        v[ix] = old - eps; lm = fb(tok, tgt, w)[1]
        v[ix] = old
        num.append((lp - lm) / (2 * eps)); ana.append(grads[k][ix])
    num, ana = np.array(num), np.array(ana)
    if max(abs(num).max(), abs(ana).max()) < 1e-8:
        print(f'{k:10s} gradiente 0 (num y ana)'); continue
    e = np.linalg.norm(num - ana) / (np.linalg.norm(num) + np.linalg.norm(ana))
    peor = max(peor, e)
    print(f'{k:10s} {e:.1e}')
print(f'Error relativo máximo: {peor:.1e}')
