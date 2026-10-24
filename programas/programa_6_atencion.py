# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =======================================================
# MECANISMO DE SELF-ATTENTION PASO A PASO
# =======================================================
 
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # estabilidad
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
 
def self_attention(X, Wq, Wk, Wv, mascara_causal=False):
    """
    Calcula el self-attention de una secuencia.
    X:    (T, d) — T tokens, cada uno de dimensión d
    Wq,Wk,Wv: (d, d) — matrices de proyección aprendibles
    mascara_causal: True en el decoder (no ver el futuro)
    """
    T, d = X.shape
    d_k  = Wk.shape[1]
    
    # 1. Proyecciones
    Q = X @ Wq   # (T, d) — Queries
    K = X @ Wk   # (T, d) — Keys
    V = X @ Wv   # (T, d) — Values
    
    # 2. Puntuaciones de similitud (producto escalar escalado)
    scores = (Q @ K.T) / np.sqrt(d_k)   # (T, T)
    
    # 3. Máscara causal: el token en posición t no puede
    #    ver tokens en posiciones > t (el futuro)
    if mascara_causal:
        mascara = np.triu(np.full((T, T), -1e9), k=1)
        scores  = scores + mascara
    
    # 4. Softmax -> pesos de atención (cada fila suma 1)
    pesos = softmax(scores, axis=-1)     # (T, T)
    
    # 5. Suma ponderada de Values
    salida = pesos @ V                   # (T, d)
    
    return salida, pesos
 
 
# -- Demostración con una frase corta -------------------
np.random.seed(3)
 
# Simulamos 5 tokens con embeddings de dimensión 4
T, d = 5, 4
frase_tokens = ['la', 'red', 'aprende', 'del', 'texto']
 
# Embeddings (en la práctica vienen de la tabla de embeddings)
X = np.random.randn(T, d) * 0.5
 
# Matrices de proyección (en la práctica se aprenden)
Wq = np.random.randn(d, d) * 0.3
Wk = np.random.randn(d, d) * 0.3
Wv = np.random.randn(d, d) * 0.3
 
# Calculamos la atención con y sin máscara causal
sal_full, pesos_full   = self_attention(X, Wq, Wk, Wv, False)
sal_causal, pesos_caus = self_attention(X, Wq, Wk, Wv, True)
 
np.set_printoptions(precision=3, suppress=True, linewidth=80)
 
print('=== SELF-ATTENTION ===')
print(f'Secuencia: {frase_tokens}')
print(f'T={T} tokens, dimensión d={d}\n')
 
print('Pesos de atención BIDIRECCIONAL (fila=query, col=key):')
print('(cada fila indica cuánto atiende ese token a los demás)')
print(pesos_full.round(3))
 
print('\nPesos de atención CAUSAL (solo mira el pasado):')
print(pesos_caus.round(3))
 
print('\nInterpretación del token "aprende" (fila 2):')
for i, (tok, peso) in enumerate(zip(frase_tokens, pesos_caus[2])):
    barra = '█' * int(peso * 40)
    print(f'  -> {tok:10s} {peso:.3f}  {barra}')
