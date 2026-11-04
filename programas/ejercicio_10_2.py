# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(42)
 
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
 
class RNNConPuerta:
    '''RNN con una puerta de olvido simplificada (estilo GRU minimalista)'''
    def __init__(self, D, H):
        self.H = H
        self.W_xh = np.random.randn(H, D) * 0.1
        self.W_hh = np.random.randn(H, H) * 0.1
        self.b_h  = np.zeros(H)
        # Parametros de la puerta de olvido
        self.W_xf = np.random.randn(H, D) * 0.1
        self.W_hf = np.random.randn(H, H) * 0.1
        self.b_f  = np.zeros(H)
    
    def forward(self, tokens, embeddings):
        h = np.zeros(self.H)
        puertas_historial = []
        for tok in tokens:
            x = embeddings[tok]
            # Puerta de olvido: decide cuanto conservar del pasado
            f = sigmoid(self.W_xf @ x + self.W_hf @ h + self.b_f)
            # Candidato de nuevo estado (como el RNN normal)
            h_candidato = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            # Combinar: f decide cuanto pasado conservar
            h = f * h + (1 - f) * h_candidato
            puertas_historial.append(f.mean())
        return h, puertas_historial
 
# Comparacion con texto simple
V, D, H = 20, 8, 16
emb = np.random.randn(V, D) * 0.1
tokens = np.random.randint(0, V, 30)
 
red = RNNConPuerta(D, H)
h_final, puertas = red.forward(tokens, emb)
 
print('=== RNN CON PUERTA DE OLVIDO ===\n')
print(f'Valor medio de la puerta a lo largo de la secuencia:')
for i, p in enumerate(puertas[:10]):
    barra = '#' * int(p * 30)
    print(f'  paso {i:2d}: {p:.3f}  {barra}')
print()
print('f cercano a 1: el modelo conserva mucho el estado anterior')
print('f cercano a 0: el modelo prioriza la nueva informacion')
print('Esta puerta, ENTRENADA, aprenderia cuando "olvidar" el contexto.')
