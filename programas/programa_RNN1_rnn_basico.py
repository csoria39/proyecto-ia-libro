# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(42)
 
# ================================================================
# RNN BASICO PARA PREDICCION DE SECUENCIAS DE CARACTERES
# Mismo problema que el Nano-LLM del Cap 13, pero con RNN.
# ================================================================
 
# Texto de ejemplo
TEXTO = 'la inteligencia artificial aprende del texto de entrenamiento'
chars = sorted(set(TEXTO))
V     = len(chars)
c2i   = {c: i for i, c in enumerate(chars)}
i2c   = dict(enumerate(chars))
datos = [c2i[c] for c in TEXTO]
 
# Hiperparametros
D = 8    # dimension del embedding
H = 16   # dimension del estado oculto
 
# Parametros del RNN (escalas elegidas para que el estado oculto se vea)
W_xh = np.random.randn(H, D) * 0.25   # entrada -> estado
W_hh = np.random.randn(H, H) * 0.25   # estado -> estado
b_h  = np.zeros(H)                     # sesgo del estado
W_hy = np.random.randn(V, H) * 0.01   # estado -> salida
b_y  = np.zeros(V)                     # sesgo de la salida
E    = np.random.randn(V, D)          # tabla de embeddings
 
def softmax(x): e=np.exp(x-x.max()); return e/e.sum()
 
def rnn_forward(tokens):
    """
    Pasa una secuencia de tokens por el RNN paso a paso.
    Devuelve los logits y el historial de estados ocultos.
    """
    T       = len(tokens)
    h       = np.zeros(H)          # estado inicial: ceros
    estados = [h.copy()]            # guardar h0
    logits  = []
    
    for t in range(T):
        x_t = E[tokens[t]]          # embedding del token actual
        # Ecuacion del RNN: combinar estado anterior + entrada actual
        h   = np.tanh(W_hh @ h + W_xh @ x_t + b_h)
        y_t = W_hy @ h + b_y        # prediccion
        estados.append(h.copy())
        logits.append(y_t)
    
    return np.array(logits), estados
 
# ================================================================
# VISUALIZAR EL ESTADO OCULTO EVOLUCIONANDO
# ================================================================
secuencia  = datos[:12]            # primeros 12 caracteres
chars_seq  = TEXTO[:12]
logits_seq, estados = rnn_forward(secuencia)
 
print('=== RNN: ESTADO OCULTO A TRAVES DEL TIEMPO ===')
print(f'Secuencia: "{chars_seq}"')
print(f'Vocab: {V} chars | Embedding: {D}d | Estado oculto: {H}d')
print()
print('Cada columna es el estado oculto h_t (primeras 6 dims).')
print('Observa como cambia con cada nuevo caracter.')
print()
print(f"{'Paso':>5}  {'Char':>5}  {'h_t[:6] (estado oculto — la memoria)':>40}")
print('-' * 58)
for t, (c, h) in enumerate(zip('·'+chars_seq, estados)):   # paso 0: h0
    char_show = repr(c) if c != ' ' else "' '"
    h_str = '[' + ' '.join(f'{v:>5.2f}' for v in h[:6]) + ']'
    print(f'{t:>5}  {char_show:>5}  {h_str}')
 
# ================================================================
# ENTRENAMIENTO BASICO (10 pasos para demostrar el mecanismo)
# ================================================================
def calcular_perdida(tokens):
    logits, _ = rnn_forward(tokens[:-1])
    perdida   = 0.0
    for t in range(len(logits)):
        p = softmax(logits[t])
        perdida -= np.log(p[tokens[t+1]] + 1e-10)
    return perdida / len(logits)
 
print(f'\nPerdida inicial (antes de entrenar): {calcular_perdida(datos):.4f}')
print(f'Perdida aleatoria esperada: ln({V}) = {np.log(V):.4f}')
 
print()
print('=== DIFERENCIA CLAVE CON EL TRANSFORMER ===')
print(f'RNN: procesa {len(datos)} tokens de UNO EN UNO (secuencial).')
print(f'     Para procesar el token 50, necesita los pasos 1..49.')
print(f'     NO se puede paralelizar en GPUs.')
print(f'Transformer: procesa TODOS los tokens en PARALELO.')
print(f'     El entrenamiento es ordenes de magnitud mas rapido.')
