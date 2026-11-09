# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA RZ-1 — ¿El Nano-LLM memoriza o generaliza?
# Capítulo 15 — Diagnóstico riguroso del Programa 8
#
# Tres pruebas sobre el mismo corpus de 93 frases:
#   1. Apartamos 10 frases que el modelo NUNCA verá y medimos
#      la pérdida en ellas a lo largo del entrenamiento.
#   2. Comparamos con un contador de n-gramas (sin red
#      neuronal): ¿supera el transformer a contar letras?
#   3. Con el modelo del Programa 8 ya entrenado, medimos qué
#      parte del texto que genera es copia literal del corpus.
#
# Requisitos: programa_8_nano_llm.py en la misma carpeta (se
# lee su TEXTO) y haberlo ejecutado antes (nano_llm_pesos.npz).
# El modelo es el MISMO del Programa 8 (GPT-2 en miniatura):
# sus funciones se copian aquí con un único cambio, que los
# parámetros se pasan como argumento (p), porque la prueba 3
# usa a la vez el modelo nuevo y el del Programa 8.
# =======================================================
import re
from collections import defaultdict
import numpy as np

# -- Corpus: el TEXTO del Programa 8 -------------------------
fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
TEXTO = fuente.split('TEXTO = """')[1].split('"""')[0]
VOCAB = sorted(set(TEXTO))
V = len(VOCAB)
C2I = {c: i for i, c in enumerate(VOCAB)}
I2C = dict(enumerate(VOCAB))
FRASES = TEXTO.strip().split('\n')

# 1 de cada 9 frases va a VALIDACIÓN (Capítulo 7, 7.5)
VAL_IDX = list(range(4, len(FRASES), 9))
VAL = [FRASES[i] for i in VAL_IDX]
TRAIN = [f for i, f in enumerate(FRASES) if i not in VAL_IDX]
texto_train = '\n' + '\n'.join(TRAIN) + '\n'
DATOS = np.array([C2I[c] for c in texto_train])
print(f'Frases: {len(FRASES)} | entrenamiento: {len(TRAIN)} | validación: {len(VAL)}')
print('Ejemplos de frases de validación (el modelo no las verá):')
for f in VAL[:3]:
    print('   ', f)

# -- Hiperparámetros: idénticos al Programa 8 ----------------
CTX, D, H, D_FF, N_BLQ = 24, 48, 4, 192, 2
DH = D // H
LR, PASOS, BATCH, CLIP = 3e-3, 12000, 8, 1.0
STD_RES = 0.02 / np.sqrt(2 * N_BLQ)
np.random.seed(42)


def randn(shape, std=0.02):
    return np.random.randn(*shape) * std


def nuevos_params():
    """Los mismos tensores del Programa 8, con su misma inicialización."""
    p = {'wte': randn((V, D)), 'wpe': randn((CTX, D)),
         'lnf_g': np.ones(D), 'lnf_b': np.zeros(D)}
    for k in range(N_BLQ):
        p[f'ln1_g{k}'] = np.ones(D); p[f'ln1_b{k}'] = np.zeros(D)
        for n in 'qkv':
            p[f'W{n}{k}'] = randn((D, D)); p[f'b{n}{k}'] = np.zeros(D)
        p[f'Wo{k}'] = randn((D, D), STD_RES); p[f'bo{k}'] = np.zeros(D)
        p[f'ln2_g{k}'] = np.ones(D); p[f'ln2_b{k}'] = np.zeros(D)
        p[f'W1{k}'] = randn((D, D_FF)); p[f'b1{k}'] = np.zeros(D_FF)
        p[f'W2{k}'] = randn((D_FF, D), STD_RES); p[f'b2{k}'] = np.zeros(D)
    return p


MASCARA = np.triu(np.full((CTX, CTX), -1e9), k=1)


# -- Las funciones del Programa 8, con los parámetros p como argumento --
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv)


def ln_bwd(dy, g, cache):
    xhat, inv = cache
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, (dy * xhat).sum((0, 1)), dy.sum((0, 1))


C_GELU = np.sqrt(2 / np.pi)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(C_GELU * (x + 0.044715 * x * x * x)))


def gelu_bwd(x):
    t = np.tanh(C_GELU * (x + 0.044715 * x * x * x))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C_GELU * (1 + 3 * 0.044715 * x * x)


def separar(x):   # (B, T, D) -> (B, H, T, DH)
    B, T, _ = x.shape
    return x.reshape(B, T, H, DH).transpose(0, 2, 1, 3)


def unir(x):      # (B, H, T, DH) -> (B, T, D)
    B, _, T, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, D)


def tmm(a, b):
    """Σ sobre lote y posiciones de aᵀ·b (gradiente de una capa lineal)."""
    return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])


# -- Forward por lotes: tok (B, T) -> logits (B, T, V) -------
def forward(p, tok):
    B, T = tok.shape
    X = p['wte'][tok] + p['wpe'][:T]
    cache = []
    for k in range(N_BLQ):
        a, c1 = ln_fwd(X, p[f'ln1_g{k}'], p[f'ln1_b{k}'])
        Q, K, Vm = (separar(a @ p[f'W{n}{k}'] + p[f'b{n}{k}']) for n in 'qkv')
        pesos = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(DH) + MASCARA[:T, :T])
        att = unir(pesos @ Vm)
        X1 = X + att @ p[f'Wo{k}'] + p[f'bo{k}']
        m_, c2 = ln_fwd(X1, p[f'ln2_g{k}'], p[f'ln2_b{k}'])
        hp = m_ @ p[f'W1{k}'] + p[f'b1{k}']
        h = gelu(hp)
        X = X1 + h @ p[f'W2{k}'] + p[f'b2{k}']
        cache.append((a, c1, Q, K, Vm, pesos, att, m_, c2, hp, h))
    xf, cf = ln_fwd(X, p['lnf_g'], p['lnf_b'])
    return xf @ p['wte'].T, (xf, cf, cache)


def perdida(p, tok, tgt, w):
    probs = softmax(forward(p, tok)[0])
    B, T = tok.shape
    lp = np.log(probs[np.arange(B)[:, None], np.arange(T)[None, :], tgt] + 1e-10)
    return -(lp * w).sum() / w.sum()


# -- Backward (las mismas derivadas del Programa 8) -----------
def forward_backward(p, tok, tgt):
    B, T = tok.shape
    logits, (xf, cf, cache) = forward(p, tok)
    probs = softmax(logits)
    bi, ti = np.arange(B)[:, None], np.arange(T)[None, :]
    loss = -np.log(probs[bi, ti, tgt] + 1e-10).mean()
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dl = probs.copy(); dl[bi, ti, tgt] -= 1.0; dl /= B * T
    g['wte'] += tmm(dl, xf)
    dX, g['lnf_g'], g['lnf_b'] = ln_bwd(dl @ p['wte'], p['lnf_g'], cf)
    for k in reversed(range(N_BLQ)):
        a, c1, Q, K, Vm, pesos, att, m_, c2, hp, h = cache[k]
        g[f'W2{k}'] = tmm(h, dX); g[f'b2{k}'] = dX.sum((0, 1))
        dhp = (dX @ p[f'W2{k}'].T) * gelu_bwd(hp)
        g[f'W1{k}'] = tmm(m_, dhp); g[f'b1{k}'] = dhp.sum((0, 1))
        dx1, g[f'ln2_g{k}'], g[f'ln2_b{k}'] = ln_bwd(dhp @ p[f'W1{k}'].T, p[f'ln2_g{k}'], c2)
        dX1 = dX + dx1
        g[f'Wo{k}'] = tmm(att, dX1); g[f'bo{k}'] = dX1.sum((0, 1))
        datt = separar(dX1 @ p[f'Wo{k}'].T)
        dpesos = datt @ Vm.transpose(0, 1, 3, 2)
        dVm = pesos.transpose(0, 1, 3, 2) @ datt
        dsc = pesos * (dpesos - (dpesos * pesos).sum(-1, keepdims=True)) / np.sqrt(DH)
        da = np.zeros_like(a)
        for n, dn in zip('qkv', (unir(dsc @ K), unir(dsc.transpose(0, 1, 3, 2) @ Q), unir(dVm))):
            g[f'W{n}{k}'] = tmm(a, dn); g[f'b{n}{k}'] = dn.sum((0, 1))
            da += dn @ p[f'W{n}{k}'].T
        dxin, g[f'ln1_g{k}'], g[f'ln1_b{k}'] = ln_bwd(da, p[f'ln1_g{k}'], c1)
        dX = dX1 + dxin
    g['wpe'][:T] = dX.sum(0)
    np.add.at(g['wte'], tok, dX)
    return g, loss


def lr_warmup_coseno(t, lr_max, t_total, t_warmup=150, lr_min=1e-4):
    if t < t_warmup:
        return lr_max * (t + 1) / t_warmup
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (t - t_warmup) / (t_total - t_warmup)))


# -- Ventanas de evaluación: cada frase con su propio contexto --
def ventanas(frases):
    xs, ys, ws = [], [], []
    for f in frases:
        ids = [C2I[c] for c in '\n' + f + '\n']
        for s in range(0, len(ids) - 1, CTX):
            n = min(CTX, len(ids) - 1 - s)
            st = max(0, s + n - CTX)              # máximo contexto posible
            x, y = ids[st:st + CTX], ids[st + 1:st + CTX + 1]
            w = [0] * (s - st) + [1] * (len(y) - (s - st))
            pad = CTX - len(y)
            xs.append(x[:len(y)] + [0] * pad); ys.append(y + [0] * pad); ws.append(w + [0] * pad)
    return np.array(xs), np.array(ys), np.array(ws, float)


XT, YT, WT = ventanas(TRAIN)
XV, YV, WV = ventanas(VAL)

# =======================================================
# PRUEBA 1 — Curva de pérdida en frases vistas y no vistas
# =======================================================
print('\n=== PRUEBA 1: pérdida en entrenamiento y en validación ===')
print(f'Referencia: adivinar al azar = ln({V}) = {np.log(V):.3f}')
p = nuevos_params()
m = {k: np.zeros_like(v) for k, v in p.items()}
v = {k: np.zeros_like(x) for k, x in p.items()}
rng = np.random.RandomState(7)
mejor = (9.9, 0)
for paso in range(PASOS):
    ii = rng.randint(0, len(DATOS) - CTX - 1, BATCH)
    x = np.stack([DATOS[i:i + CTX] for i in ii]); y = np.stack([DATOS[i + 1:i + CTX + 1] for i in ii])
    g, _ = forward_backward(p, x, y)
    norma = np.sqrt(sum((gg ** 2).sum() for gg in g.values()))
    if norma > CLIP:
        g = {k: gg * CLIP / norma for k, gg in g.items()}
    t, lr = paso + 1, lr_warmup_coseno(paso, LR, PASOS)
    for k in p:
        m[k] = 0.9 * m[k] + 0.1 * g[k]
        v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
        p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    if paso % 250 == 0 or paso == PASOS - 1:
        lv = perdida(p, XV, YV, WV)
        if lv < mejor[0]:
            mejor = (lv, paso)
        if paso % 1000 == 0 or paso == PASOS - 1:
            lt = perdida(p, XT, YT, WT)
            print(f'Paso {paso:5d}: entrenamiento {lt:.3f} | validación {lv:.3f}')
print(f'Mejor validación: {mejor[0]:.3f} en el paso {mejor[1]} '
      f'(ahí habría parado el Early Stopping del Capítulo 7)')


# =======================================================
# PRUEBA 2 — ¿Supera el transformer a un contador de n-gramas?
# =======================================================
def ngramas(n):
    """Modelo de n-gramas de caracteres con suavizado Witten-Bell."""
    cnt = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
    for i in range(len(texto_train)):
        for k in range(min(n, i + 1)):
            cnt[k][texto_train[i - k:i]][texto_train[i]] += 1

    def prob(ctx, c):
        q = 1.0 / V
        for k in range(min(n, len(ctx) + 1)):
            d = cnt[k].get(ctx[len(ctx) - k:] if k else '')
            if d:
                N, Tt = sum(d.values()), len(d)
                q = (d.get(c, 0) + Tt * q) / (N + Tt)
        return q
    return prob


def entropia_cruzada(prob, frases):
    tot = n = 0
    for f in frases:
        s = '\n' + f + '\n'
        for j in range(1, len(s)):
            tot -= np.log(prob(s[max(0, j - CTX):j], s[j])); n += 1
    return tot / n


print('\n=== PRUEBA 2: contadores de n-gramas (sin red neuronal) ===')
print('contexto   entrenamiento   validación')
for n in range(1, 7):
    pr = ngramas(n)
    print(f'{n - 1:5d} car. {entropia_cruzada(pr, TRAIN):12.3f} {entropia_cruzada(pr, VAL):12.3f}')

# =======================================================
# PRUEBA 3 — ¿Cuánto del texto generado es copia literal?
# =======================================================
print('\n=== PRUEBA 3: texto generado por el Programa 8 (todo el corpus) ===')
p8 = dict(np.load('nano_llm_pesos.npz'))
PALABRAS = set(re.findall(r'[a-záéíóúñ]+', TEXTO))


def top_p(pr, tp=0.9):
    o = np.argsort(pr)[::-1]
    n = int(np.searchsorted(np.cumsum(pr[o]), tp)) + 1
    q = np.zeros_like(pr); q[o[:n]] = pr[o[:n]]
    return q / q.sum()


def generar_lote(semilla, n_chars, k, rng, tau=0.8):
    toks = np.array([[C2I[c] for c in semilla]] * k)
    for _ in range(n_chars):
        lg = forward(p8, toks[:, -CTX:])[0][:, -1]
        sig = [rng.choice(V, p=top_p(softmax(lg[i] / tau))) for i in range(k)]
        toks = np.concatenate([toks, np.array(sig)[:, None]], 1)
    return [''.join(I2C[t] for t in fila) for fila in toks]


def fraccion_copiada(s, L=20):
    """Fracción de caracteres dentro de algún trozo de >= L caracteres
    que aparece tal cual en el corpus."""
    cub = np.zeros(len(s), bool)
    for i in range(len(s) - L + 1):
        if s[i:i + L] in TEXTO:
            cub[i:i + L] = True
    return cub.mean()


rng = np.random.RandomState(2026)
print('semilla              copia(>=20 car.)  líneas idénticas  palabras inventadas')
for semilla in ['la inteligencia', 'un modelo ', 'el gato ', 'mañana la ']:
    textos = generar_lote(semilla, 150, 10, rng)
    copia = np.mean([fraccion_copiada(t) for t in textos])
    lineas = [l for t in textos for l in t[len(semilla):].split('\n')[1:-1]]
    identicas = sum(l in FRASES for l in lineas)
    palabras = [w for t in textos for w in re.findall(r'[a-záéíóúñ]+', t[len(semilla):])[:-1]]
    nuevas = sorted({w for w in palabras if w not in PALABRAS})
    print(f'{semilla!r:20s} {copia:10.0%} {identicas:10d}/{len(lineas):<3d} '
          f'{len(nuevas):8d} {", ".join(nuevas[:3])}')
print('\nEjemplo con una semilla que no está en el corpus:')
print(textos[0])
