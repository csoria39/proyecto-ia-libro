# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA RZ-4 — El Nano-LLM razona con frases del corpus
# Capítulo 15 — Deducción con frases al estilo del Programa 8
# El modelo del Programa 8 (GPT-2 en miniatura, NumPy puro),
# como en RZ-2 y RZ-3, deduce conclusiones a partir de frases
# al estilo del corpus de IA del Programa 8:
#
#   Hechos:   la regularización dificulta el sobreajuste.
#             el sobreajuste limita la generalización.
#             ... (más frases, algunas sin relación)
#   Pregunta: ¿la regularización ayuda a la generalización?
#   Razonamiento escrito por el modelo (cita cada hecho que usa):
#     la regularización dificulta el sobreajuste   [efecto acumulado: perjudica]
#     el sobreajuste limita la generalización      [efecto acumulado: favorece]
#     la pregunta dice «ayuda a» (favorece): coincide  ⇒  SÍ
#
# Cambios respecto al Programa RZ-3:
#   1. Frases en vez de letras: 40 conceptos y 8 verbos tomados
#      del corpus del Programa 8 (datos, sobreajuste, atención,
#      «dificulta», «limita», «acelera», «estabiliza»…).
#   2. Tokens de palabra en vez de caracteres: cada concepto
#      («la regularización») y cada verbo es un token, como hacen
#      los LLM reales con su tokenizador. A nivel de carácter, 10
#      frases ocuparían ~400 posiciones y el entrenamiento en NumPy
#      sería decenas de veces más lento.
#   3. Razonamiento con signo: unos verbos favorecen y otros
#      perjudican, así que en cada paso el modelo debe acumular el
#      efecto (perjudicar algo que perjudica = favorecer). En cada
#      paso escribe el verbo del hecho que usa, luego el efecto
#      acumulado y luego el concepto siguiente. Escribir el verbo
#      fue necesario: sin él, el modelo encontraba bien la cadena
#      de conceptos pero acertaba el efecto solo por azar, porque
#      tenía que buscar el verbo y combinarlo en un único paso.
#   4. Tres respuestas: SÍ, NO o NO SE PUEDE DEDUCIR (cuando la
#      pregunta une conceptos de cadenas distintas). La pregunta se
#      escribe internamente con forma de hecho («B verbo ?»), así que
#      comprobar si se ha llegado a B es la misma operación que dar
#      un paso más. Fue la clave: con la pregunta en otro formato, el
#      modelo seguía bien la cadena pero, al final, aprendía a
#      comprobar si había llegado a B solo con algunas semillas.
#   5. emb_prev2 además de emb_prev: cada posición sabe también
#      qué token va dos lugares antes. Así, en «A verbo B», la
#      posición de B «sabe» a la vez su sujeto (A) y su verbo, y
#      el modelo puede buscar el hecho por su sujeto con una sola
#      atención.
#   6. Los conceptos, los verbos y el orden de las frases se
#      sortean en cada problema: no hay nada que memorizar.
#   7. Se entrena SOLO con deducciones de 1 a 4 pasos y se
#      examina con 1 a 8: de 5 a 8 nunca los ha visto.
#   8. Reinicio de seguridad: si en el paso 3000 el modelo no acierta
#      el 80 % de las deducciones de 3 pasos (se quedó atascado en una
#      meseta), el programa vuelve a empezar con otra semilla.
# Uso: python programa_RZ4_deduce_frases.py [semilla]
# Tiempo aproximado: 15–20 minutos en un portátil.
# =======================================================
import sys
import numpy as np

SEMILLA = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_HECHOS, PASOS_MAX_ENTRENO = 10, 4

# -- Vocabulario de palabras (todas aparecen en el corpus del Programa 8) --
CONCEPTOS = ['los datos', 'el modelo', 'el entrenamiento', 'el sobreajuste', 'la regularización',
             'la generalización', 'la atención', 'el embedding', 'la tokenización', 'la pérdida',
             'el gradiente', 'la normalización', 'el ruido', 'la memoria', 'la precisión',
             'el error', 'la escala', 'la eficiencia', 'el costo', 'la incertidumbre',
             'la validación', 'el contexto', 'el vocabulario', 'la temperatura', 'la variedad',
             'el sesgo', 'la privacidad', 'la interpretabilidad', 'el paralelismo', 'la recompensa',
             'la exploración', 'la explotación', 'el aprendizaje', 'la inferencia',
             'el preentrenamiento', 'el ajuste fino', 'la simulación', 'la confianza',
             'la arquitectura', 'el optimizador']
FAVORECE = ['ayuda a', 'estabiliza', 'acelera', 'mejora']          # efecto +
PERJUDICA = ['dificulta', 'limita', 'complica', 'ralentiza']       # efecto −
VERBOS = FAVORECE + PERJUDICA
ESPECIALES = ['<pad>', '.', '¿', '?', '[+]', '[−]', 'SÍ', 'NO', 'NO SE PUEDE DEDUCIR']
VOCAB = ESPECIALES + CONCEPTOS + VERBOS
V = len(VOCAB)
W2I = {w: i for i, w in enumerate(VOCAB)}
RESPUESTAS = {W2I['SÍ'], W2I['NO'], W2I['NO SE PUEDE DEDUCIR']}


def signo(verbo):
    return +1 if verbo in FAVORECE else -1


def problema(L, rng, relleno=0):
    """Tres cadenas de hechos «A verbo B.»; la primera tiene L pasos.
    Pregunta: ¿inicio de la cadena 1 <verbo> t?  Si t es el final de la
    cadena 1, la respuesta depende del efecto acumulado; si t es el final
    de otra cadena, no se puede deducir."""
    r1 = rng.randint(1, N_HECHOS - L); r2 = N_HECHOS - L - r1
    c = [CONCEPTOS[i] for i in rng.permutation(len(CONCEPTOS))[:N_HECHOS + 3]]
    c1, c2, c3 = c[:L + 1], c[L + 1:L + 2 + r1], c[L + 2 + r1:]
    verbo = {}
    hechos = []
    for cad in (c1, c2, c3):
        for a, b in zip(cad[:-1], cad[1:]):
            v = VERBOS[rng.randint(len(VERBOS))]
            verbo[a] = v
            hechos.append([a, v, b, '.'])
    rng.shuffle(hechos)
    conectado = rng.rand() < 0.5
    t = c1[-1] if conectado else (c2[-1] if rng.rand() < 0.5 else c3[-1])
    vq = VERBOS[rng.randint(len(VERBOS))]
    # La pregunta «¿A verbo B?» se escribe como «¿ A ? ¿ B verbo ?»:
    #  - «B verbo ?» tiene la misma forma que un hecho («B verbo …»). Al llegar al
    #    final de la cadena, la MISMA búsqueda que el modelo hace en cada paso
    #    («¿qué verbo sigue a este concepto?») encuentra el verbo de la pregunta
    #    solo si ha llegado a B. Si no lo encuentra, la cadena se ha cortado.
    #  - Los «?» y «¿» de en medio evitan que la pregunta parezca un hecho cuyo
    #    sujeto sea A, que confundiría el primer paso.
    prompt = ['<pad>'] * relleno + [w for h in hechos for w in h] + ['¿', c1[0], '?', '¿', t, vq, '?']
    # razonamiento: concepto, y por cada paso: verbo, efecto acumulado, concepto
    razon, s = [c1[0]], +1
    for a, b in zip(c1[:-1], c1[1:]):
        s *= signo(verbo[a])
        razon += [verbo[a], '[+]' if s > 0 else '[−]', b]
    if not conectado:
        # la cadena termina sin llegar a B: no hay más hechos que seguir
        return prompt, razon + ['NO SE PUEDE DEDUCIR']
    # último «paso» con el verbo de la pregunta: si el efecto total coincide con
    # lo que afirma la pregunta, el efecto combinado es + y la respuesta es SÍ
    s2 = s * signo(vq)
    return prompt, razon + [vq, '[+]' if s2 > 0 else '[−]', '?', 'SÍ' if s2 > 0 else 'NO']


# -- Hiperparámetros ---------------------------------------
CTX, D, D_FF, N_BLQ, H = 78, 48, 192, 2, 2
DH = D // H
LR, PASOS, BATCH, CLIP, RELLENO_MAX = 3e-3, 6000, 64, 1.0, 14
# RELLENO_MAX: al entrenar se añade un relleno al azar al principio, para que las
# cadenas cortas ocupen también las posiciones finales que usarán las largas.
# Reinicio: con tan pocos parámetros, algunas inicializaciones se quedan atascadas
# en una meseta. Si en el paso COMPROBACION el modelo no acierta al menos el 80 %
# de las deducciones de 3 pasos, se reinicia con otra semilla (máx. INTENTOS).
COMPROBACION, UMBRAL, INTENTOS = 3000, 0.80, 4
if len(sys.argv) > 2:
    PASOS = int(sys.argv[2])


STD_RES = 0.02 / np.sqrt(2 * N_BLQ)


def randn(shape, std=0.02):
    return np.random.randn(*shape) * std


params = {}


def inicializar(semilla):
    np.random.seed(semilla)
    params.clear()
    params.update({'wte': randn((V, D)), 'wpe': randn((CTX, D)),
                   'emb_prev': randn((V, D)), 'emb_prev2': randn((V, D))})
    params['lnf_g'] = np.ones(D); params['lnf_b'] = np.zeros(D)
    for k in range(N_BLQ):
        params[f'ln1_g{k}'] = np.ones(D); params[f'ln1_b{k}'] = np.zeros(D)
        for n in 'qkv':
            params[f'W{n}{k}'] = randn((D, D)); params[f'b{n}{k}'] = np.zeros(D)
        params[f'Wo{k}'] = randn((D, D), STD_RES); params[f'bo{k}'] = np.zeros(D)
        params[f'ln2_g{k}'] = np.ones(D); params[f'ln2_b{k}'] = np.zeros(D)
        params[f'W1{k}'] = randn((D, D_FF)); params[f'b1{k}'] = np.zeros(D_FF)
        params[f'W2{k}'] = randn((D_FF, D), STD_RES); params[f'b2{k}'] = np.zeros(D)


inicializar(SEMILLA)
print(f'Semilla {SEMILLA} | vocabulario: {V} tokens | parámetros: {sum(p.size for p in params.values()):,}')
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
    X[:, 1:] += params['emb_prev'][tok[:, :-1]]          # token anterior
    X[:, 2:] += params['emb_prev2'][tok[:, :-2]]         # token de dos posiciones antes
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
    np.add.at(g['emb_prev'], tok[:, :-1], dX[:, 1:])
    np.add.at(g['emb_prev2'], tok[:, :-2], dX[:, 2:])
    return g, loss


# -- Lotes de entrenamiento (deducciones de 1 a 4 pasos) ---
def lote(rng):
    X = np.zeros((BATCH, CTX - 1), int); Y = np.zeros_like(X); W = np.zeros(X.shape)
    for i in range(BATCH):
        p, r = problema(rng.randint(1, PASOS_MAX_ENTRENO + 1), rng, rng.randint(0, RELLENO_MAX + 1))
        s = [W2I[w] for w in p + r]
        s += [0] * (CTX - len(s))
        X[i], Y[i] = s[:-1], s[1:]
        W[i, len(p) - 1:len(p) + len(r) - 1] = 1        # solo lo que escribe el modelo
    return X, Y, W


# -- Generación greedy hasta la respuesta -----------------
def resolver(prompts, max_tokens=3 * N_HECHOS + 4):
    seqs = [[W2I[w] for w in p] for p in prompts]
    salida = [[] for _ in prompts]
    for _ in range(max_tokens):
        activos = [i for i in range(len(seqs)) if len(seqs[i]) < CTX
                   and not (salida[i] and W2I[salida[i][-1]] in RESPUESTAS)]
        if not activos:
            break
        T = max(len(seqs[i]) for i in activos)
        X = np.array([seqs[i] + [0] * (T - len(seqs[i])) for i in activos])
        logits, _, _ = forward(X)
        for j, i in enumerate(activos):
            nx = int(logits[j, len(seqs[i]) - 1].argmax())
            seqs[i].append(nx); salida[i].append(VOCAB[nx])
    return salida


def examen(L, n=200):
    rng = np.random.RandomState(1000 + L)
    probs = [problema(L, rng) for _ in range(n)]
    sal = resolver([p for p, _ in probs])
    resp = np.mean([s[-1:] == r[-1:] for s, (_, r) in zip(sal, probs)])
    todo = np.mean([s == r for s, (_, r) in zip(sal, probs)])
    return resp, todo


# -- Entrenamiento -----------------------------------------
def lr_warmup_coseno(t, lr_max, t_total, t_warmup=150, lr_min=1e-4):
    if t < t_warmup:
        return lr_max * (t + 1) / t_warmup
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (t - t_warmup) / (t_total - t_warmup)))


def entrenar(semilla):
    """Entrena desde cero; devuelve False si en la comprobación sigue atascado."""
    inicializar(semilla)
    m_adam = {k: np.zeros_like(v) for k, v in params.items()}
    v_adam = {k: np.zeros_like(v) for k, v in params.items()}
    rng = np.random.RandomState(semilla + 7)
    print(f'\nSemilla {semilla}: entrenando con deducciones de 1 a {PASOS_MAX_ENTRENO} pasos...', flush=True)
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
                  f' | con 7 pasos (nunca vistos) {examen(7, 50)[0]:5.0%}', flush=True)
        if paso == COMPROBACION and examen(3, 100)[0] < UMBRAL:
            print(f'  → atascado en una meseta en el paso {paso}: se reinicia con otra semilla', flush=True)
            return False
    return True


for intento in range(INTENTOS):
    if entrenar(SEMILLA + 100 * intento):
        break
else:
    print(f'\nAVISO: ninguno de los {INTENTOS} intentos salió de la meseta; '
          'el examen siguiente corresponde a un modelo que no ha aprendido.')

# -- Examen final ------------------------------------------
print('\n=== EXAMEN: 200 problemas nuevos por cada número de pasos ===')
print('pasos  respuesta  razonamiento completo correcto')
for L in range(1, N_HECHOS - 1):
    r, t = examen(L)
    nota = '' if L <= PASOS_MAX_ENTRENO else '  ← nunca visto al entrenar'
    print(f'{L:5d}  {r:9.1%}  {t:9.1%}{nota}')


# -- Ejemplos legibles -------------------------------------
def frase(a, v, b):
    txt = f'{a} {v} {b}'
    return txt.replace(' a el ', ' al ')


def legible(razon, objetivo):
    out, actual = '', razon[0]
    for i in range(1, len(razon) - 1, 3):
        v, sg, b = (razon[i:i + 3] + ['?', '?', '?'])[:3]
        if b == '?':                                    # paso con el verbo de la pregunta
            tipo = 'favorece' if v in FAVORECE else 'perjudica'
            out += (f'\n      la pregunta dice «{v}» ({tipo}): '
                    f'{"coincide" if sg == "[+]" else "no coincide"} con el efecto acumulado')
            break
        efecto = 'favorece' if sg == '[+]' else 'perjudica' if sg == '[−]' else sg
        out += f'\n      {frase(actual, v, b):48s} [efecto acumulado: {efecto}]'
        actual = b
    else:
        out += f'\n      la cadena termina en «{actual}», no en «{objetivo}»'
    return out


print('\n=== EJEMPLOS ===')
rng = np.random.RandomState(2026)
for L in (2, 5, 7):
    p, r = problema(L, rng)
    sal = resolver([p])[0]
    toks = [w for w in p if w != '<pad>']
    q = toks.index('¿')
    print('Hechos:')
    for i in range(0, q, 4):
        print('   ', frase(*toks[i:i + 3]) + '.')
    print(f'Pregunta: ¿{frase(toks[q + 1], toks[q + 5], toks[q + 4])}?')
    print('Razonamiento del modelo:', legible(sal, toks[q + 4]))
    print(f'      ⇒  {sal[-1]}   (correcto: {r[-1]})\n')
