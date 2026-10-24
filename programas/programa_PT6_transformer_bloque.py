# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import torch
import torch.nn as nn
 
class BloqueTransformer(nn.Module):
    '''
    Bloque Transformer usando modulos nativos de PyTorch.
    Mismo diseño que el Programa 7 NumPy (Pre-LN, como GPT-2):
      LayerNorm + Self-Attention causal + residual
      LayerNorm + Feed-Forward (GELU) + residual
    '''
    def __init__(self, d, d_ff, n_heads=2):
        super().__init__()
        # Multi-Head Attention nativo
        self.atencion = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=False   # X tiene forma (T, B, d)
        )
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(approximate='tanh'),
            nn.Linear(d_ff, d)
        )
        # Dos capas de normalizacion
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
    
    def forward(self, X):
        # X: (T, d) -- sin dimension de batch
        T  = X.shape[0]
        
        # Mascara causal: impide ver el futuro
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        
        # Sub-bloque 1: norm + Self-Attention + residual
        a = self.norm1(X).unsqueeze(1)   # (T, 1, d) para MultiheadAttention
        attn_out, _ = self.atencion(a, a, a,
                                    attn_mask=mask,
                                    is_causal=True)
        X = X + attn_out.squeeze(1)
        
        # Sub-bloque 2: norm + FFN + residual
        X = X + self.ffn(self.norm2(X))
        return X
 
 
torch.manual_seed(5)
d, d_ff, T = 16, 64, 6
bloque = BloqueTransformer(d, d_ff, n_heads=2)
X = torch.randn(T, d)
 
with torch.no_grad():
    Y = bloque(X)
 
n_p = sum(p.numel() for p in bloque.parameters())
print(f'Bloque Transformer PyTorch: d={d}, d_ff={d_ff}, n_heads=2')
print(f'Entrada: {X.shape}  ->  Salida: {Y.shape}')
print(f'Parametros: {n_p:,}  (los mismos 3.280 del Programa 7)')
print(f'Std entrada: {X.std().item():.4f}   Std salida: {Y.std().item():.4f}')
 
# Alternativa aun mas compacta: nn.TransformerEncoderLayer
capa_rapida = nn.TransformerEncoderLayer(
    d_model=d, nhead=2, dim_feedforward=d_ff,
    dropout=0.0, batch_first=True, norm_first=True
)
n_p2 = sum(p.numel() for p in capa_rapida.parameters())
print(f'\nTransformerEncoderLayer: {n_p2:,} params (equivalente)')
