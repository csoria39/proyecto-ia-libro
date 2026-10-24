# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np

# =======================================================
# UN BLOQUE TRANSFORMER COMPLETO (diseño de GPT-2: Pre-LN)
# (versión simplificada sin Multi-Head para claridad)
# =======================================================

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):
    """Normaliza cada fila y aplica escala g (γ) y desplazamiento b (β)"""
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def gelu(x):
    """La activación de GPT-2: una ReLU 'suave'"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class BloqueTransformer:
    """
    Un bloque del Transformer decoder, como en GPT-2 (Pre-LN):
    1. LayerNorm -> self-attention causal -> suma residual
    2. LayerNorm -> Feed-Forward (GELU)   -> suma residual
    """

    def __init__(self, d, d_ff=None):
        d_ff = d_ff or d * 4
        # Dos LayerNorm con escala (γ) y desplazamiento (β)
        self.g1, self.b_ln1 = np.ones(d), np.zeros(d)
        self.g2, self.b_ln2 = np.ones(d), np.zeros(d)
        # Matrices de atención, cada una con su sesgo
        self.Wq = np.random.randn(d, d) * 0.02;  self.bq = np.zeros(d)
        self.Wk = np.random.randn(d, d) * 0.02;  self.bk = np.zeros(d)
        self.Wv = np.random.randn(d, d) * 0.02;  self.bv = np.zeros(d)
        self.Wo = np.random.randn(d, d) * 0.02;  self.bo = np.zeros(d)
        # Matrices FFN
        self.W1 = np.random.randn(d, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d) * 0.02
        self.b2 = np.zeros(d)

    def atencion_causal(self, X):
        T, d = X.shape
        Q = X @ self.Wq + self.bq
        K = X @ self.Wk + self.bk
        V = X @ self.Wv + self.bv
        scores = Q @ K.T / np.sqrt(d)
        # Máscara causal: -inf en el triángulo superior
        mascara = np.triu(np.full((T, T), -1e9), k=1)
        pesos   = softmax(scores + mascara)
        return (pesos @ V) @ self.Wo + self.bo

    def ffn(self, X):
        h = gelu(X @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward(self, X):
        # Sub-bloque 1: norm -> self-attention -> residual
        X = X + self.atencion_causal(layer_norm(X, self.g1, self.b_ln1))
        # Sub-bloque 2: norm -> FFN -> residual
        X = X + self.ffn(layer_norm(X, self.g2, self.b_ln2))
        return X


# -- Demostración ---------------------------------------
np.random.seed(5)
T, d = 6, 16    # 6 tokens, dimensión 16

bloque = BloqueTransformer(d=16, d_ff=64)

# Entrada: 6 tokens aleatorios
X_entrada = np.random.randn(T, d)
X_salida  = bloque.forward(X_entrada)

print('=== BLOQUE TRANSFORMER ===')
print(f'Entrada: {T} tokens x {d} dimensiones')
print(f'Salida:  {X_salida.shape[0]} tokens x {X_salida.shape[1]} dimensiones')
print(f'El bloque mantiene las dimensiones (T={T}, d={d})')

# Contar parámetros: atención (4 matrices + 4 sesgos), FFN, 2 LayerNorm
n_params = (4*d*d + 4*d) + (2 * d*(d*4) + d*4 + d) + 2 * 2*d
print(f'Parámetros del bloque: {n_params}')

# Pre-LN: la salida es la entrada MÁS lo que aportan las subcapas
cambio = X_salida - X_entrada
print(f'\nLa vía residual conserva la entrada:')
print(f'  Std de la entrada:        {X_entrada.std():.4f}')
print(f'  Std de lo que se suma:    {cambio.std():.4f}')
print(f'  Std de la salida:         {X_salida.std():.4f}')
