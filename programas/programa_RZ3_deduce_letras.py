# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA RZ-3 — El Nano-LLM razona paso a paso
# Capítulo 15 — Deducir con hechos que nunca ha visto
#
# El modelo del Programa 8 (GPT-2 en miniatura, NumPy puro),
# como en RZ-2, aprende a DEDUCIR a partir de hechos que cambian
# en cada problema.
#
#   Hechos:   q>m, m>c, c>x, ...   («q es mayor que m»)
#   Pregunta: ¿q>x?
#   Razonamiento escrito por el modelo:  q > m > c > x  ⇒ SÍ
#
# Cambios respecto al Programa RZ-2:
#   1. Tarea de deducción: 12 hechos con letras al azar; la
#      respuesta exige encadenar 1, 2, 3… hechos (transitividad).
#      Como las letras cambian en cada problema, no hay nada que
#      memorizar: solo sirve razonar sobre los hechos dados.
#   2. Cadena de razonamiento: el modelo escribe cada paso antes
#      de responder (modo 'paso_a_paso'). En modo 'directo' debe
#      responder sí/no sin escribir pasos.
#   3. Embedding del carácter anterior (emb_prev): cada posición
#      sabe qué letra la precede, así un hecho «cx» se puede
#      buscar por su primera letra con una sola atención.
#   4. Se entrena SOLO con deducciones de 1 a 4 pasos y se
#      examina con 1 a 10 pasos: de 5 a 10 nunca los ha visto.
#   5. Los embeddings (wte, wpe, emb_prev) empiezan con desviación
#      EMB_STD = 0.5 en lugar del 0.02 del Programa 8: con 0.02 el
#      modelo no sale de la meseta en esta tarea (sección 15.4).
#
# Uso:  python programa_RZ3_deduce_letras.py [modo] [emb_prev]
#   modo      paso_a_paso (por defecto) o directo
#   emb_prev  si (por defecto) o no
# Los tres experimentos del capítulo:
#   python programa_RZ3_deduce_letras.py                    -> aprende
#   python programa_RZ3_deduce_letras.py directo            -> no aprende
#   python programa_RZ3_deduce_letras.py paso_a_paso no     -> no aprende
# Tiempo aproximado: unos 8 minutos en un portátil.
# =======================================================
import sys
import numpy as np

MODO = sys.argv[1] if len(sys.argv) > 1 else 'paso_a_paso'     # o 'directo'
USAR_EMB_PREV = (sys.argv[2] if len(sys.argv) > 2 else 'si') == 'si'
N_HECHOS, PASOS_MAX_ENTRENO = 12, 4

# -- Vocabulario --------------------------------------------
LETRAS = 'abcdefghijklmnopqrstuvwxyz'
VOCAB = ['.'] + list(LETRAS) + [',', '|', '?', ':', 'S', 'N', '\n']
V = len(VOCAB)
C2I = {c: i for i, c in enumerate(VOCAB)}
I2C = dict(enumerate(VOCAB))


def problema(L, rng, relleno=0):
    """Tres cadenas disjuntas de hechos; la primera tiene L pasos.
    Pregunta: ¿inicio de la cadena 1 > t?  t es el final de la
    cadena 1 (respuesta S) o el final de otra cadena (N)."""
    r1 = rng.randint(1, N_HECHOS - L); r2 = N_HECHOS - L - r1
    n = list(rng.permutation(list(LETRAS))[:N_HECHOS + 3])
    c1, c2, c3 = n[:L + 1], n[L + 1:L + 2 + r1], n[L + 2 + r1:]
    hechos = [a + b for c in (c1, c2, c3) for a, b in zip(c[:-1], c[1:])]
    rng.shuffle(hechos)
    si = rng.rand() < 0.5
    t = c1[-1] if si else (c2[-1] if rng.rand() < 0.5 else c3[-1])
    prompt = '.' * relleno + ','.join(hechos) + '|' + c1[0] + '?' + t + ':'
    razonamiento = ''.join(c1) if MODO == 'paso_a_paso' else ''
    return prompt, razonamiento + ('S' if si else 'N')


# -- Hiperparámetros ---------------------------------------
CTX, D, D_FF, N_BLQ, H = 56, 48, 192, 2, 2
DH = D // H
LR, PASOS, BATCH, CLIP, RELLENO_MAX = 3e-3, 2000, 64, 1.0, 8
EMB_STD = 0.5          # desviación inicial de los embeddings (Programa 8: 0.02)
np.random.seed(42)


STD_RES = 0.02 / np.sqrt(2 * N_BLQ)


def randn(shape, std=0.02):
    return np.random.randn(*shape) * std


params = {'wte': randn((V, D), EMB_STD), 'wpe': randn((CTX, D), EMB_STD),
          'emb_prev': randn((V, D), EMB_STD)}
params['lnf_g'] = np.ones(D); params['lnf_b'] = np.zeros(D)
for k in range(N_BLQ):
    params[f'ln1_g{k}'] = np.ones(D); params[f'ln1_b{k}'] = np.zeros(D)
    for n in 'qkv':
        params[f'W{n}{k}'] = randn((D, D)); params[f'b{n}{k}'] = np.zeros(D)
    params[f'Wo{k}'] = randn((D, D), STD_RES); params[f'bo{k}'] = np.zeros(D)
    params[f'ln2_g{k}'] = np.ones(D); params[f'ln2_b{k}'] = np.zeros(D)
    params[f'W1{k}'] = randn((D, D_FF)); params[f'b1{k}'] = np.zeros(D_FF)
    params[f'W2{k}'] = randn((D_FF, D), STD_RES); params[f'b2{k}'] = np.zeros(D)
if not USAR_EMB_PREV:
    params['emb_prev'][:] = 0.0          # existe pero no se usa ni se entrena
n_par = sum(p.size for k, p in params.items() if USAR_EMB_PREV or k != 'emb_prev')
print(f'Modo: {MODO} | emb_prev: {"sí" if USAR_EMB_PREV else "no"} | parámetros: {n_par:,}')
MASCARA = np.triu(np.full((CTX, CTX), -1e9), k=1)


# -- Utilidades (las del Programa 8) -------------------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv)


def layer_norm_bwd(dy, g, cache):
    xhat, inv = cache
    ejes = tuple(range(dy.ndim - 1))
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, (dy * xhat).sum(ejes), dy.sum(ejes)


C_GELU = np.sqrt(2 / np.pi)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(C_GELU * (x + 0.044715 * x * x * x)))


def gelu_bwd(x):
    t = np.tanh(C_GELU * (x + 0.044715 * x * x * x))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C_GELU * (1 + 3 * 0.044715 * x * x)


def separar(x):        # (B,T,D) -> (B,H,T,DH)
    B, T, _ = x.shape
    return x.reshape(B, T, H, DH).transpose(0, 2, 1, 3)


def unir(x):           # (B,H,T,DH) -> (B,T,D)
    B, _, T, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, D)


def tmm(a, b):
    """Suma sobre lote y posiciones de aᵀ·b: el gradiente de una capa lineal."""
    return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])


# -- Forward (el del Programa 8) ----------------------------
def forward(tok):
    B, T = tok.shape
    X = params['wte'][tok] + params['wpe'][:T]
    if USAR_EMB_PREV:
        X[:, 1:] += params['emb_prev'][tok[:, :-1]]      # quién va delante de mí
    cache = []
    for k in range(N_BLQ):
        Xin = X
        a, ln1 = layer_norm_fwd(Xin, params[f'ln1_g{k}'], params[f'ln1_b{k}'])
        Q, K, Vm = (separar(a @ params[f'W{n}{k}'] + params[f'b{n}{k}']) for n in 'qkv')
        pesos = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(DH) + MASCARA[:T, :T])
        att = unir(pesos @ Vm)
        X1 = Xin + att @ params[f'Wo{k}'] + params[f'bo{k}']
        m, ln2 = layer_norm_fwd(X1, params[f'ln2_g{k}'], params[f'ln2_b{k}'])
        h_pre = m @ params[f'W1{k}'] + params[f'b1{k}']
        h = gelu(h_pre)
        X = X1 + h @ params[f'W2{k}'] + params[f'b2{k}']
        cache.append(dict(a=a, ln1=ln1, Q=Q, K=K, Vm=Vm, pesos=pesos,
                          att=att, m=m, ln2=ln2, h_pre=h_pre, h=h))
    xf, lnf = layer_norm_fwd(X, params['lnf_g'], params['lnf_b'])
    return xf @ params['wte'].T, (xf, lnf), cache


# -- Forward + Backward (el del Programa 8, con la pérdida ponderada por w) --
def forward_backward(tok, tgt, w):
    B, T = tok.shape
    logits, (xf, lnf), cache = forward(tok)
    probs = softmax(logits)
    bi, ti = np.arange(B)[:, None], np.arange(T)[None, :]
    loss = -(np.log(probs[bi, ti, tgt] + 1e-10) * w).sum() / w.sum()
    g = {k: np.zeros_like(v) for k, v in params.items()}
    dlog = probs.copy()
    dlog[bi, ti, tgt] -= 1.0
    dlog *= (w / w.sum())[..., None]          # solo cuentan las posiciones con w = 1
    g['wte'] += tmm(dlog, xf)                 # wte como matriz de salida
    dX, g['lnf_g'], g['lnf_b'] = layer_norm_bwd(dlog @ params['wte'], params['lnf_g'], lnf)
    for k in reversed(range(N_BLQ)):
        c = cache[k]
        g[f'W2{k}'] = tmm(c['h'], dX)
        g[f'b2{k}'] = dX.sum((0, 1))
        dh_pre = (dX @ params[f'W2{k}'].T) * gelu_bwd(c['h_pre'])
        g[f'W1{k}'] = tmm(c['m'], dh_pre)
        g[f'b1{k}'] = dh_pre.sum((0, 1))
        dx1, g[f'ln2_g{k}'], g[f'ln2_b{k}'] = layer_norm_bwd(dh_pre @ params[f'W1{k}'].T, params[f'ln2_g{k}'], c['ln2'])
        dX1 = dX + dx1
        g[f'Wo{k}'] = tmm(c['att'], dX1)
        g[f'bo{k}'] = dX1.sum((0, 1))
        datt = separar(dX1 @ params[f'Wo{k}'].T)
        dpesos = datt @ c['Vm'].transpose(0, 1, 3, 2)
        dVm = c['pesos'].transpose(0, 1, 3, 2) @ datt
        s = (dpesos * c['pesos']).sum(-1, keepdims=True)
        dscores = c['pesos'] * (dpesos - s) / np.sqrt(DH)
        dQ, dK, dV = unir(dscores @ c['K']), unir(dscores.transpose(0, 1, 3, 2) @ c['Q']), unir(dVm)
        da = np.zeros_like(c['a'])
        for n, dn in zip('qkv', (dQ, dK, dV)):
            g[f'W{n}{k}'] = tmm(c['a'], dn)
            g[f'b{n}{k}'] = dn.sum((0, 1))
            da += dn @ params[f'W{n}{k}'].T
        dxin, g[f'ln1_g{k}'], g[f'ln1_b{k}'] = layer_norm_bwd(da, params[f'ln1_g{k}'], c['ln1'])
        dX = dX1 + dxin
    g['wpe'][:T] = dX.sum(0)
    np.add.at(g['wte'], tok, dX)              # wte también como embedding de entrada
    if USAR_EMB_PREV:
        np.add.at(g['emb_prev'], tok[:, :-1], dX[:, 1:])
    return g, loss


# -- Lotes de entrenamiento (deducciones de 1 a 4 pasos) ---
def lote(rng):
    X = np.zeros((BATCH, CTX - 1), int); Y = np.zeros_like(X); W = np.zeros(X.shape)
    for i in range(BATCH):
        p, r = problema(rng.randint(1, PASOS_MAX_ENTRENO + 1), rng, rng.randint(0, RELLENO_MAX + 1))
        s = [C2I[c] for c in p + r + '\n']
        s += [0] * (CTX - len(s))
        X[i], Y[i] = s[:-1], s[1:]
        W[i, len(p) - 1:len(p) + len(r)] = 1            # solo lo que escribe el modelo
    return X, Y, W


# -- Generación greedy hasta S o N -------------------------
def resolver(prompts, max_chars=12):
    seqs = [[C2I[c] for c in p] for p in prompts]
    salida = ['' for _ in prompts]
    for _ in range(max_chars):
        T = max(len(s) for s in seqs)
        X = np.array([s + [0] * (T - len(s)) for s in seqs])
        logits, _, _ = forward(X)
        for i, s in enumerate(seqs):
            if salida[i][-1:] in ('S', 'N'):
                continue
            nx = int(logits[i, len(s) - 1].argmax())
            s.append(nx); salida[i] += I2C[nx]
    return salida


def examen(L, n=200):
    rng = np.random.RandomState(1000 + L)
    probs = [problema(L, rng) for _ in range(n)]
    sal = resolver([p for p, _ in probs])
    resp = np.mean([s[-1:] == r[-1] for s, (_, r) in zip(sal, probs)])
    todo = np.mean([s == r for s, (_, r) in zip(sal, probs)])
    return resp, todo


# -- Entrenamiento -----------------------------------------
def lr_warmup_coseno(t, lr_max, t_total, t_warmup=150, lr_min=1e-4):
    if t < t_warmup:
        return lr_max * (t + 1) / t_warmup
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (t - t_warmup) / (t_total - t_warmup)))


m_adam = {k: np.zeros_like(v) for k, v in params.items()}
v_adam = {k: np.zeros_like(v) for k, v in params.items()}
rng = np.random.RandomState(7)
print(f'\nEntrenando con deducciones de 1 a {PASOS_MAX_ENTRENO} pasos...')
for paso in range(PASOS):
    X, Y, W = lote(rng)
    g, loss = forward_backward(X, Y, W)
    norma = np.sqrt(sum((x ** 2).sum() for x in g.values()))
    if norma > CLIP:
        g = {k: x * CLIP / norma for k, x in g.items()}
    t, lr = paso + 1, lr_warmup_coseno(paso, LR, PASOS)
    for k in params:
        m_adam[k] = 0.9 * m_adam[k] + 0.1 * g[k]
        v_adam[k] = 0.999 * v_adam[k] + 0.001 * g[k] ** 2
        params[k] -= lr * (m_adam[k] / (1 - 0.9 ** t)) / (np.sqrt(v_adam[k] / (1 - 0.999 ** t)) + 1e-8)
    if paso % 250 == 0 or paso == PASOS - 1:
        print(f'Paso {paso:5d}: pérdida {loss:.4f} | aciertos con 3 pasos {examen(3, 50)[0]:5.0%}'
              f' | con 7 pasos (nunca vistos) {examen(7, 50)[0]:5.0%}')

# -- Examen final ------------------------------------------
print('\n=== EXAMEN: 200 problemas nuevos por cada número de pasos ===')
print('pasos  respuesta  razonamiento completo correcto')
for L in range(1, N_HECHOS - 1):
    r, t = examen(L)
    nota = '' if L <= PASOS_MAX_ENTRENO else '  ← nunca visto al entrenar'
    print(f'{L:5d}  {r:9.1%}  {t:9.1%}{nota}')

# -- Tres ejemplos legibles --------------------------------
print('\n=== EJEMPLOS ===')
rng = np.random.RandomState(2026)
for L in (2, 6, 9):
    p, r = problema(L, rng)
    sal = resolver([p])[0]
    hechos = p.split('|')[0].split(',')
    s, t = p.split('|')[1][0], p.split('|')[1][2]
    print('Hechos:', ', '.join(f'{h[0]}>{h[1]}' for h in hechos))
    print(f'¿{s}>{t}?   Modelo: {" > ".join(sal[:-1]) + "  ⇒  " if sal[:-1] else ""}'
          f'{"SÍ" if sal[-1:] == "S" else "NO"}   (correcto: {"SÍ" if r[-1] == "S" else "NO"})\n')
