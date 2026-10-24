# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA 8 — Nano-LLM: un GPT-2 en miniatura
# Modelo de lenguaje de nivel carácter con la arquitectura
# de GPT-2 (2019): embeddings de token y de posición
# aprendidos, 2 bloques Pre-LN (LayerNorm con γ y β antes de
# cada subcapa), atención causal de 4 cabezas, FFN con GELU,
# LayerNorm final y pesos compartidos entre la entrada y la
# salida. Solo cambian los tamaños (GPT-2 small: 12 bloques,
# 12 cabezas, D = 768, contexto 1024, 50 257 tokens BPE).
# NumPy puro: backpropagation analítico completo, mini-lotes,
# Adam con warmup + coseno y recorte del gradiente. Guarda
# los pesos en nano_llm_pesos.npz para los capítulos siguientes.
# =======================================================
import numpy as np
# -- Texto de entrenamiento (93 líneas, vocabulario de 46 caracteres) --
TEXTO = """
la inteligencia artificial aprende de los datos.
los modelos de lenguaje predicen el siguiente token.
las redes neuronales tienen capas de neuronas.
el aprendizaje profundo usa muchas capas ocultas.
la atención permite ver todo el contexto a la vez.
los transformers son la base de los llm modernos.
entrenar un modelo requiere muchos datos y cálculo.
el futuro pertenece a quienes comprenden la ia.
un modelo pequeño puede aprender patrones simples.
la tokenización convierte texto en números enteros.
cada carácter del vocabulario tiene un índice único.
el embedding transforma índices en vectores densos.
la codificación posicional indica el orden del texto.
la función softmax convierte logits en probabilidades.
el optimizador adam ajusta los pesos, paso a paso.
una red profunda combina atención y capas ocultas.
generar texto significa predecir, uno a uno, cada token.
la temperatura controla cuánta variedad hay al generar.
un buen modelo equilibra memoria y generalización.
¿comprender la ia? requiere práctica, paciencia y datos.
también existe el software: gpt-2 fue un hito clave.
un modelo (bien entrenado) generaliza; uno malo, no.
¡el aprendizaje nunca termina! sigue explorando.
un vector es una lista ordenada de números reales.
una matriz agrupa vectores en filas y columnas.
el producto punto mide cuánto se parecen dos vectores.
la retropropagación calcula gradientes hacia atrás.
el descenso de gradiente ajusta los pesos poco a poco.
una época recorre todo el conjunto de entrenamiento.
el sobreajuste ocurre cuando el modelo memoriza demasiado.
la regularización ayuda a que el modelo generalice mejor.
un lote agrupa varios ejemplos para entrenar juntos.
la normalización estabiliza las activaciones internas.
una neurona combina entradas, pesos y un sesgo.
la función de pérdida mide el error del modelo.
el aprendizaje supervisado usa ejemplos etiquetados.
una capa oculta transforma la información paso a paso.
el descenso estocástico usa lotes pequeños y aleatorios.
la validación comprueba si el modelo generaliza bien.
un hiperparámetro se ajusta antes de entrenar el modelo.
la arquitectura define cómo se conectan las capas.
el vocabulario agrupa todos los símbolos posibles.
una secuencia es un orden concreto de tokens.
el contexto limita cuántos tokens ve el modelo a la vez.
la generalización permite responder a datos nuevos.
comprender estos conceptos abre la puerta a la ia moderna.
un dataset agrupa todos los ejemplos disponibles.
el conjunto de test evalúa el modelo al final, una sola vez.
la precisión mide cuántas predicciones son correctas.
el error mide cuánto se aleja la predicción del valor real.
una etiqueta indica la respuesta correcta de un ejemplo.
la inferencia usa el modelo ya entrenado, sin ajustar pesos.
el preentrenamiento aprende patrones generales del lenguaje.
el ajuste fino especializa un modelo en una tarea concreta.
un token puede ser una letra, una parte de palabra o una palabra.
la ventana de contexto limita cuánto texto ve el modelo.
un residual conecta la entrada directamente con la salida.
la normalización evita que las activaciones crezcan sin control.
el ruido en los datos dificulta encontrar el patrón real.
una red ancha tiene muchas neuronas en cada capa.
una red profunda tiene muchas capas apiladas.
el paralelismo permite entrenar varias partes a la vez.
la memoria limita cuántos datos caben en una gráfica.
un acelerador gráfico multiplica matrices muy rápido.
la escala mide cuántos parámetros tiene un modelo.
un modelo grande necesita muchos más datos para aprender bien.
la eficiencia importa tanto como la precisión final.
el costo de entrenar crece con el tamaño del modelo.
la interpretabilidad explica por qué el modelo decide algo.
un sesgo en los datos puede producir un modelo injusto.
la ética en la ia estudia el impacto social de los modelos.
la privacidad protege los datos personales de los usuarios.
un modelo robusto resiste pequeños cambios en la entrada.
la incertidumbre mide cuánta confianza tiene el modelo.
un ejemplo difícil confunde incluso a un buen modelo.
la curva de aprendizaje muestra el progreso del entrenamiento.
un mínimo local complica encontrar la mejor solución.
el paisaje de pérdida describe cómo cambia el error.
una meseta ralentiza el progreso del entrenamiento.
el ruido ayuda a veces a escapar de mínimos malos.
la exploración prueba caminos nuevos durante el entrenamiento.
la explotación aprovecha lo que ya funciona bien.
un modelo generativo produce contenido nuevo y original.
un modelo discriminativo distingue entre categorías distintas.
la clasificación asigna una categoría a cada ejemplo.
la regresión predice un número continuo, no una categoría.
el aprendizaje por refuerzo premia las buenas decisiones.
un agente aprende actuando dentro de un entorno.
la recompensa guía el aprendizaje del agente.
la política define cómo actúa el agente en cada estado.
un entorno simulado permite entrenar sin riesgo real.
la simulación acelera enormemente el aprendizaje del agente.
comprender bien lo básico facilita entender lo avanzado.
"""
# -- Tokenizador ----------------------------------------
VOCAB = sorted(set(TEXTO))
V = len(VOCAB)
C2I = {c: i for i, c in enumerate(VOCAB)}
I2C = dict(enumerate(VOCAB))
DATOS = np.array([C2I[c] for c in TEXTO], dtype=int)
# -- Hiperparámetros (entre paréntesis, los de GPT-2 small) --
CTX    = 24     # longitud del contexto (1024)
D      = 48     # dimensión del modelo (768)
H      = 4      # cabezas de atención (12)
DH     = D // H # dimensión de cada cabeza: 12 (64)
D_FF   = 192    # dimensión interna FFN, 4xD (3072)
N_BLQ  = 2      # bloques transformer (12)
LR     = 3e-3   # tasa de aprendizaje máxima
PASOS  = 12000  # pasos de entrenamiento (cada paso ve BATCH ventanas al azar;
                #  no son épocas: 12000 pasos ≈ 454 pasadas por el corpus)
BATCH  = 8      # secuencias por lote (promedia el gradiente, suaviza la curva)
CLIP   = 1.0    # recorte de la norma del gradiente (Capítulo 5, 5.8)
np.random.seed(42)
# -- Parámetros del modelo (inicialización de GPT-2) --------
def randn(shape, std=0.02):
    return np.random.randn(*shape) * std
# Las proyecciones que se suman a la vía residual (Wo y W2) se
# inician con una desviación menor, 0.02/sqrt(2·N_BLQ), como en GPT-2.
STD_RES = 0.02 / np.sqrt(2 * N_BLQ)
params = {
    'wte': randn((V, D)),     # embedding de tokens (y matriz de salida)
    'wpe': randn((CTX, D)),   # embedding de posiciones
    'lnf_g': np.ones(D), 'lnf_b': np.zeros(D),   # LayerNorm final
}
for k in range(N_BLQ):
    params[f'ln1_g{k}'] = np.ones(D);  params[f'ln1_b{k}'] = np.zeros(D)
    for n in 'qkv':
        params[f'W{n}{k}'] = randn((D, D)); params[f'b{n}{k}'] = np.zeros(D)
    params[f'Wo{k}'] = randn((D, D), STD_RES); params[f'bo{k}'] = np.zeros(D)
    params[f'ln2_g{k}'] = np.ones(D);  params[f'ln2_b{k}'] = np.zeros(D)
    params[f'W1{k}'] = randn((D, D_FF)); params[f'b1{k}'] = np.zeros(D_FF)
    params[f'W2{k}'] = randn((D_FF, D), STD_RES); params[f'b2{k}'] = np.zeros(D)
total = sum(p.size for p in params.values())
print(f'Vocabulario: {V} caracteres')
print(f'Total de parámetros: {total:,}')
print(f'Texto de entrenamiento: {len(DATOS)} tokens, '
      f'{TEXTO.strip().count(chr(10)) + 1} líneas')
# Máscara causal: se calcula una vez con el tamaño máximo (CTX, CTX)
# y se recorta a [:T, :T] cuando la secuencia es más corta.
MASCARA_CAUSAL = np.triu(np.full((CTX, CTX), -1e9), k=1)

# -- Utilidades ---------------------------------------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)
def layer_norm_fwd(x, g, b, eps=1e-5):
    """LayerNorm de GPT-2: normaliza cada fila y aplica escala g (γ) y desplazamiento b (β)."""
    mu = x.mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv)
def layer_norm_bwd(dy, g, cache):
    """Derivada exacta de LayerNorm: devuelve dx, dg y db."""
    xhat, inv = cache
    ejes = tuple(range(dy.ndim - 1))      # todas las filas (lote y posiciones)
    dg = (dy * xhat).sum(ejes)
    db = dy.sum(ejes)
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db
C_GELU = np.sqrt(2 / np.pi)
def gelu(x):
    """GELU con la aproximación tanh que usa GPT-2."""
    return 0.5 * x * (1 + np.tanh(C_GELU * (x + 0.044715 * x * x * x)))
def gelu_bwd(x):
    """Derivada de gelu(x)."""
    t = np.tanh(C_GELU * (x + 0.044715 * x * x * x))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C_GELU * (1 + 3 * 0.044715 * x * x)
def separar(x):   # (B, T, D) -> (B, H, T, DH): una matriz por cabeza
    B, T, _ = x.shape
    return x.reshape(B, T, H, DH).transpose(0, 2, 1, 3)
def unir(x):      # (B, H, T, DH) -> (B, T, D)
    B, _, T, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, D)
def tmm(a, b):
    """Suma sobre lote y posiciones de aᵀ·b: el gradiente de una capa lineal."""
    return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])

# -- Forward ---------------------------------------------------
def forward(tokens):
    """tokens (B, T) -> logits (B, T, V), y lo que el backward necesita."""
    B, T = tokens.shape
    X = params['wte'][tokens] + params['wpe'][:T]
    cache = []
    for k in range(N_BLQ):
        Xin = X
        # Subcapa 1: LayerNorm -> atención causal multi-cabeza -> suma residual
        a, ln1 = layer_norm_fwd(Xin, params[f'ln1_g{k}'], params[f'ln1_b{k}'])
        Q, K, Vm = (separar(a @ params[f'W{n}{k}'] + params[f'b{n}{k}']) for n in 'qkv')
        pesos = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(DH) + MASCARA_CAUSAL[:T, :T])
        att = unir(pesos @ Vm)
        X1 = Xin + att @ params[f'Wo{k}'] + params[f'bo{k}']
        # Subcapa 2: LayerNorm -> FFN con GELU -> suma residual
        m, ln2 = layer_norm_fwd(X1, params[f'ln2_g{k}'], params[f'ln2_b{k}'])
        h_pre = m @ params[f'W1{k}'] + params[f'b1{k}']
        h = gelu(h_pre)
        X = X1 + h @ params[f'W2{k}'] + params[f'b2{k}']
        cache.append(dict(a=a, ln1=ln1, Q=Q, K=K, Vm=Vm, pesos=pesos,
                          att=att, m=m, ln2=ln2, h_pre=h_pre, h=h))
    xf, lnf = layer_norm_fwd(X, params['lnf_g'], params['lnf_b'])
    logits = xf @ params['wte'].T          # pesos compartidos: la salida usa wte
    return logits, (xf, lnf, cache)

# -- Forward + Backward (un mini-lote entero) ---------------------
def forward_backward(tokens, targets):
    """
    Pasa el mini-lote por el modelo y calcula, con backpropagation
    exacto (regla de la cadena), el gradiente de la pérdida media
    respecto a TODOS los parámetros.
    """
    B, T = tokens.shape
    logits, (xf, lnf, cache) = forward(tokens)
    probs = softmax(logits)
    bi, ti = np.arange(B)[:, None], np.arange(T)[None, :]
    loss = -np.log(probs[bi, ti, targets] + 1e-10).mean()
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dlogits = probs.copy()
    dlogits[bi, ti, targets] -= 1.0          # softmax + entropía cruzada
    dlogits /= B * T
    grads['wte'] += tmm(dlogits, xf)         # wte como matriz de salida
    dxf = dlogits @ params['wte']
    dX, grads['lnf_g'], grads['lnf_b'] = layer_norm_bwd(dxf, params['lnf_g'], lnf)
    for k in reversed(range(N_BLQ)):
        c = cache[k]
        # Subcapa 2 hacia atrás. Residual: dX llega entero a X1 y a la FFN
        grads[f'W2{k}'] = tmm(c['h'], dX)
        grads[f'b2{k}'] = dX.sum((0, 1))
        dh_pre = (dX @ params[f'W2{k}'].T) * gelu_bwd(c['h_pre'])
        grads[f'W1{k}'] = tmm(c['m'], dh_pre)
        grads[f'b1{k}'] = dh_pre.sum((0, 1))
        dm = dh_pre @ params[f'W1{k}'].T
        dx1, grads[f'ln2_g{k}'], grads[f'ln2_b{k}'] = layer_norm_bwd(dm, params[f'ln2_g{k}'], c['ln2'])
        dX1 = dX + dx1
        # Subcapa 1 hacia atrás. Residual: dX1 llega entero a Xin y a la atención
        grads[f'Wo{k}'] = tmm(c['att'], dX1)
        grads[f'bo{k}'] = dX1.sum((0, 1))
        datt = separar(dX1 @ params[f'Wo{k}'].T)
        dpesos = datt @ c['Vm'].transpose(0, 1, 3, 2)
        dVm = c['pesos'].transpose(0, 1, 3, 2) @ datt
        s = (dpesos * c['pesos']).sum(-1, keepdims=True)
        dscores = c['pesos'] * (dpesos - s) / np.sqrt(DH)   # derivada de softmax
        dQ = unir(dscores @ c['K'])
        dK = unir(dscores.transpose(0, 1, 3, 2) @ c['Q'])
        dV = unir(dVm)
        da = np.zeros_like(c['a'])
        for n, dn in zip('qkv', (dQ, dK, dV)):
            grads[f'W{n}{k}'] = tmm(c['a'], dn)
            grads[f'b{n}{k}'] = dn.sum((0, 1))
            da += dn @ params[f'W{n}{k}'].T
        dxin, grads[f'ln1_g{k}'], grads[f'ln1_b{k}'] = layer_norm_bwd(da, params[f'ln1_g{k}'], c['ln1'])
        dX = dX1 + dxin
    grads['wpe'][:T] = dX.sum(0)
    np.add.at(grads['wte'], tokens, dX)      # wte también como embedding de entrada
    return grads, loss

# -- Learning rate: warmup lineal + decaimiento coseno --------
def lr_warmup_coseno(t, lr_max, t_total, t_warmup=150, lr_min=1e-4):
    """
    Sube linealmente de 0 a lr_max durante t_warmup pasos (evita
    pasos erráticos al principio, cuando los gradientes son más
    inestables), luego decae en coseno hasta lr_min (afina los
    pesos al final). Es el esquema estándar al entrenar LLMs.
    """
    if t < t_warmup:
        return lr_max * (t + 1) / t_warmup     # +1: el paso 0 ya aprende
    t_adj = t - t_warmup
    t_decay = t_total - t_warmup
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t_adj / t_decay))

# -- Entrenamiento con Adam (mini-lotes de BATCH secuencias) --
m_adam = {k: np.zeros_like(v) for k, v in params.items()}
v_adam = {k: np.zeros_like(v) for k, v in params.items()}
beta1, beta2, eps_adam, t_adam = 0.9, 0.999, 1e-8, 0
LR_MAX = LR
print('\nEntrenando Nano-LLM (backpropagation analítico, mini-lotes)...')
print(f'CTX={CTX} | D={D} | D_FF={D_FF} | Bloques={N_BLQ} | Batch={BATCH}\n')
for paso in range(PASOS):
    lr_actual = lr_warmup_coseno(paso, LR_MAX, PASOS)
    inicios = np.random.randint(0, len(DATOS) - CTX - 1, size=BATCH)
    xb = np.stack([DATOS[i:i + CTX] for i in inicios])          # (BATCH, CTX)
    yb = np.stack([DATOS[i + 1:i + CTX + 1] for i in inicios])  # desplazado 1
    grads_acc, loss = forward_backward(xb, yb)
    # Recorte del gradiente: si la norma global supera CLIP, se reescala
    norma = np.sqrt(sum((g ** 2).sum() for g in grads_acc.values()))
    if norma > CLIP:
        for k in grads_acc:
            grads_acc[k] *= CLIP / norma
    t_adam += 1
    for k in params:
        g = grads_acc[k]
        m_adam[k] = beta1 * m_adam[k] + (1 - beta1) * g
        v_adam[k] = beta2 * v_adam[k] + (1 - beta2) * g ** 2
        m_h = m_adam[k] / (1 - beta1 ** t_adam)
        v_h = v_adam[k] / (1 - beta2 ** t_adam)
        params[k] -= lr_actual * m_h / (np.sqrt(v_h) + eps_adam)
    if paso % 1000 == 0 or paso == PASOS - 1:
        print(f'Paso {paso:5d}: pérdida = {loss:.4f}   lr = {lr_actual:.5f}   '
              f'(bits/char = {loss/np.log(2):.2f})')

# -- Estrategias de muestreo (se estudian a fondo en el Capítulo 14) --
# generar() usa temperatura + top_p_filtrar; greedy y top_k_filtrar
# quedan definidas para experimentar con ellas.
def greedy(probs):
    """Siempre el token más probable. Puede quedar 'atrapado' en bucles."""
    return np.argmax(probs)
def top_k_filtrar(probs, k):
    """Conserva solo los k tokens más probables."""
    indices_ord = np.argsort(probs)[::-1]
    nueva = np.zeros_like(probs)
    nueva[indices_ord[:k]] = probs[indices_ord[:k]]
    return nueva / nueva.sum()
def top_p_filtrar(probs, p):
    """Conserva el mínimo conjunto de tokens que suman probabilidad p (nucleus sampling)."""
    indices_ord = np.argsort(probs)[::-1]
    probs_ord   = probs[indices_ord]
    suma_acum   = np.cumsum(probs_ord)
    n_tokens = int(np.searchsorted(suma_acum, p)) + 1
    nueva = np.zeros_like(probs)
    nueva[indices_ord[:n_tokens]] = probs[indices_ord[:n_tokens]]
    return nueva / nueva.sum()

# -- Generación de texto -----------------------------------
def generar(semilla, n_chars=120, temperatura=0.8, top_p=0.9):
    """
    Genera texto carácter a carácter.
    temperatura > 1: más aleatorio | temperatura < 1: más determinista
    top_p: nucleus sampling — conserva solo el conjunto mínimo de
    caracteres cuya probabilidad acumulada alcanza top_p, evitando
    elegir caracteres de probabilidad residual que producen "sopa
    de letras". Con top_p=1.0 equivale al muestreo puro.
    """
    tokens = [C2I.get(c, 0) for c in semilla[-CTX:]]
    resultado = semilla
    for _ in range(n_chars):
        ctx = tokens[-CTX:]
        logits = forward(np.array([ctx]))[0][0, -1]
        probs = softmax(logits / temperatura)
        if top_p < 1.0:
            probs = top_p_filtrar(probs, top_p)
        siguiente = int(np.random.choice(V, p=probs))
        tokens.append(siguiente)
        resultado += I2C[siguiente]
    return resultado

print('\n=== TEXTO GENERADO POR EL NANO-LLM ===')
print(generar('la inteligencia', n_chars=120))
print()
print(generar('los modelos de', n_chars=120))

# -- Guardar los pesos entrenados ------------------------------
# Los Capítulos 11, 14, 15 y 18 reutilizan este modelo ya entrenado:
#   params = dict(np.load('nano_llm_pesos.npz'))
np.savez('nano_llm_pesos.npz', **params)
print("\nPesos guardados en nano_llm_pesos.npz")
