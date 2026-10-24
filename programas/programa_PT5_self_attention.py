# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class SelfAttentionCausal(nn.Module):
    '''
    Self-attention causal con PyTorch.
    Wq, Wk, Wv son nn.Linear -> gradientes automaticos.
    '''
    def __init__(self, d):
        super().__init__()
        self.d  = d
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
    
    def forward(self, X):
        # X: (T, d)
        T = X.shape[0]
        Q = self.Wq(X)   # (T, d)
        K = self.Wk(X)   # (T, d)
        V = self.Wv(X)   # (T, d)
        
        # Puntuaciones escaladas
        scores = (Q @ K.T) / (self.d ** 0.5)   # (T, T)
        
        # Mascara causal: el token t no puede ver t+1, t+2...
        mask   = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        scores = scores + mask
        
        # Pesos de atencion y salida
        pesos  = F.softmax(scores, dim=-1)  # (T, T)
        return self.Wo(pesos @ V), pesos    # (T, d)
 
 
torch.manual_seed(3)
T, d = 5, 8
X    = torch.randn(T, d)
 
attn = SelfAttentionCausal(d)
with torch.no_grad():
    salida, pesos = attn(X)
 
print(f'=== SELF-ATTENTION PYTORCH ===')
print(f'Entrada: {X.shape}  ->  Salida: {salida.shape}')
print(f'Pesos de atencion ({T}x{T}):')
print(pesos.numpy().round(3))
print(f'Filas suman 1: {torch.allclose(pesos.sum(dim=-1), torch.ones(T))}')
n_p = sum(p.numel() for p in attn.parameters())
print(f'Parametros de la capa: {n_p}')
 
# Comparativa de codigo con NumPy:
# NumPy:   salida, pesos = self_attention(X, Wq, Wk, Wv)
#          -> requeria pasar matrices como argumentos
# PyTorch: salida, pesos = attn(X)
#          -> las matrices son parte del objeto, con gradientes
