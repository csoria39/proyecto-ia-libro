# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA RZ-2 — El Nano-LLM aprende una regla: sumar
# Capítulo 15 — De memorizar a generalizar
#
# El modelo del Programa 8 (GPT-2 en miniatura, NumPy puro)
# entrenado con sumas de dos cifras.
# Se examina con 2 000 sumas que NUNCA ha visto: si acierta,
# no puede haberlas memorizado; ha extraído la regla.
#
# Cambios respecto al Programa 8:
#   1. Datos: sumas «47+85=» en lugar de frases. De las 10 000
#      sumas posibles, 2 000 se apartan para el examen.
#   2. Resultado escrito al revés (unidades primero), como se
#      suma a mano: cada cifra depende solo de lo ya escrito.
#   3. Atención de 2 cabezas (el Programa 8 usa 4).
#   4. Lotes vectorizados (B, T, D) en lugar de un bucle.
#   5. La pérdida solo cuenta las 3 cifras del resultado.
#   6. Todo lo demás es el Programa 8: bloques Pre-LN con γ y β,
#      GELU, LayerNorm final, pesos compartidos, inicialización de
#      GPT-2 y Adam con warmup + coseno y recorte del gradiente.
#
# Uso:  python programa_RZ2_suma.py [semilla] [n_entreno] [cabezas] [excluir]
#   n_entreno  sumas de entrenamiento (por defecto 8000)
#   cabezas    cabezas de atención (por defecto 2)
#   excluir    p. ej. 75: no se entrena con ninguna suma cuyas
#              unidades sean 7 y 5 (ni 5 y 7); el examen son esas
#              200 sumas. Comprueba si el modelo deduce un hecho
#              que nunca vio.
# Tiempo aproximado: unos 3 minutos.
# =======================================================
import sys
import numpy as np

# -- Datos: sumas de dos cifras ----------------------------
VOCAB = list('0123456789+=\n')
V = len(VOCAB)
C2I = {c: i for i, c in enumerate(VOCAB)}
I2C = dict(enumerate(VOCAB))


def formatear(a, b):
    """47+85=132  ->  '47+85=231\\n'  (resultado al revés, 3 cifras)"""
    return f'{a:02d}+{b:02d}=' + str(a + b).zfill(3)[::-1] + '\n'


SEMILLA = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_ENTRENO = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
H = int(sys.argv[3]) if len(sys.argv) > 3 else 2          # cabezas de atención
EXCLUIR = sys.argv[4] if len(sys.argv) > 4 else None

pares = [(a, b) for a in range(100) for b in range(100)]      # 10 000 sumas
np.random.RandomState(0).shuffle(pares)                       # partición fija
TEST, TRAIN = pares[:2000], pares[2000:2000 + N_ENTRENO]       # 2 000 nunca vistas
if EXCLUIR:
    u1, u2 = int(EXCLUIR[0]), int(EXCLUIR[1])
    prohibida = lambda a, b: sorted((a % 10, b % 10)) == sorted((u1, u2))
    TEST = [q for q in pares if prohibida(*q)]
    TRAIN = [q for q in pares if not prohibida(*q)][:N_ENTRENO]


def a_tensor(ps):
    S = np.array([[C2I[c] for c in formatear(a, b)] for a, b in ps])
    return S[:, :-1], S[:, 1:]                                 # entrada, objetivo


X_TR, Y_TR = a_tensor(TRAIN)
X_TE, Y_TE = a_tensor(TEST)
PESO = np.zeros(9); PESO[5:8] = 1      # solo se puntúan las 3 cifras del resultado

# -- Hiperparámetros ---------------------------------------
CTX, D, D_FF, N_BLQ = 9, 48, 192, 2
DH = D // H                            # 24 dimensiones por cabeza (con H = 2)
LR, PASOS, BATCH, CLIP = 3e-3, 6000, 64, 1.0
np.random.seed(SEMILLA)

STD_RES = 0.02 / np.sqrt(2 * N_BLQ)    # Wo y W2, como en el Programa 8


def randn(shape, std=0.02):
    return np.random.randn(*shape) * std


params = {'wte': randn((V, D)), 'wpe': randn((CTX, D))}
params['lnf_g'] = np.ones(D); params['lnf_b'] = np.zeros(D)
for k in range(N_BLQ):
    params[f'ln1_g{k}'] = np.ones(D); params[f'ln1_b{k}'] = np.zeros(D)
    for n in 'qkv':
        params[f'W{n}{k}'] = randn((D, D)); params[f'b{n}{k}'] = np.zeros(D)
    params[f'Wo{k}'] = randn((D, D), STD_RES); params[f'bo{k}'] = np.zeros(D)
    params[f'ln2_g{k}'] = np.ones(D); params[f'ln2_b{k}'] = np.zeros(D)
    params[f'W1{k}'] = randn((D, D_FF)); params[f'b1{k}'] = np.zeros(D_FF)
    params[f'W2{k}'] = randn((D_FF, D), STD_RES); params[f'b2{k}'] = np.zeros(D)
print(f'Parámetros: {sum(p.size for p in params.values()):,}')
print(f'Semilla {SEMILLA} | cabezas {H} | sumas de entrenamiento: {len(TRAIN)} | '
      f'sumas de examen (no vistas): {len(TEST)}' + (f' | excluidas unidades {EXCLUIR[0]} y {EXCLUIR[1]}' if EXCLUIR else ''))
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
    return g, loss


# -- Examen: decodificación greedy de las 3 cifras ---------
def resolver(X):
    seq = X[:, :6].copy()                                  # 'aa+bb='
    for _ in range(3):
        logits, _, _ = forward(seq)
        seq = np.concatenate([seq, logits[:, -1].argmax(-1)[:, None]], 1)
    return seq[:, 6:9]


def exactitud(X, Y):
    return (resolver(X) == Y[:, 5:8]).all(1).mean()


def lr_warmup_coseno(t, lr_max, t_total, t_warmup=150, lr_min=1e-4):
    if t < t_warmup:
        return lr_max * (t + 1) / t_warmup                 # +1: el paso 0 ya aprende
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (t - t_warmup) / (t_total - t_warmup)))


# -- Entrenamiento (Adam + recorte de gradiente) -----------
m_adam = {k: np.zeros_like(v) for k, v in params.items()}
v_adam = {k: np.zeros_like(v) for k, v in params.items()}
print('\nEntrenando...  (exactitud = suma entera correcta)')
for paso in range(PASOS):
    lr = lr_warmup_coseno(paso, LR, PASOS)
    idx = np.random.randint(0, len(X_TR), BATCH)
    g, loss = forward_backward(X_TR[idx], Y_TR[idx], np.tile(PESO, (BATCH, 1)))
    norma = np.sqrt(sum((x ** 2).sum() for x in g.values()))
    if norma > CLIP:
        g = {k: x * CLIP / norma for k, x in g.items()}
    t = paso + 1
    for k in params:
        m_adam[k] = 0.9 * m_adam[k] + 0.1 * g[k]
        v_adam[k] = 0.999 * v_adam[k] + 0.001 * g[k] ** 2
        params[k] -= lr * (m_adam[k] / (1 - 0.9 ** t)) / (np.sqrt(v_adam[k] / (1 - 0.999 ** t)) + 1e-8)
    if paso % 500 == 0 or paso == PASOS - 1:
        print(f'Paso {paso:5d}: pérdida {loss:.4f} | exactitud entrenamiento '
              f'{exactitud(X_TR[:2000], Y_TR[:2000]):6.1%} | sumas NUNCA vistas {exactitud(X_TE, Y_TE):6.1%}')

# -- Resultado ---------------------------------------------
print('\n=== EXAMEN CON SUMAS QUE EL MODELO NUNCA VIO ===')
res = resolver(X_TE[:10])
for (a, b), r in zip(TEST[:10], res):
    pred = int(''.join(I2C[i] for i in r)[::-1])
    print(f'{a:2d} + {b:2d} = {pred:3d}   {"✓" if pred == a + b else "✗ (correcto: " + str(a + b) + ")"}')
print(f'\nAciertos en las {len(TEST)} sumas no vistas: {exactitud(X_TE, Y_TE):.1%}')

# -- ¿Cómo lo hace? A qué cifras mira cada paso -----------
_, _, cache = forward(X_TE[:, :9])
nombres = ['a(dec)', 'a(uni)', '+', 'b(dec)', 'b(uni)', '=', 'r(uni)', 'r(dec)', 'r(cen)']
print(f'\n=== ¿A QUÉ CIFRAS ATIENDE CADA CABEZA? (media sobre {len(TEST)} sumas) ===')
for k in range(N_BLQ):
    for q, que in [(5, 'unidades'), (6, 'decenas '), (7, 'centenas')]:
        txt = []
        for c in range(H):
            fila = cache[k]['pesos'][:, c, q].mean(0)
            top = np.argsort(fila)[::-1][:2]
            txt.append(f'cabeza {c}: ' + ' + '.join(f'{nombres[j]} {fila[j]:.2f}' for j in top))
        print(f'Bloque {k}, al escribir las {que} -> ' + ' | '.join(txt))
