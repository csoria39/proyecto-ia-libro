# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA PT-7 — Nano-LLM completo con PyTorch
# Apéndice D — El MISMO GPT-2 en miniatura del Programa 8,
# ahora con autograd: PyTorch calcula el backward por nosotros.
#
# Comprobado con PyTorch 2.14 (CPU): imprime
# «Total de parámetros: 60,000», igual que el Programa 8.
#
# Las piezas son las de GPT-2 y las del Programa 8:
#   - embeddings de token y de posición aprendidos
#   - bloques Pre-LN: nn.LayerNorm (con γ y β) ANTES de cada subcapa
#   - atención causal de 4 cabezas, con sesgo en Q, K, V y Wo
#   - FFN 48 -> 192 (GELU, aproximación tanh) -> 48
#   - LayerNorm final y pesos compartidos entre entrada y salida
# Cambia el backward (autograd) y nada más.
# =======================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Corpus: el mismo TEXTO del Programa 8 ------------------
fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
TEXTO = fuente.split('TEXTO = """')[1].split('"""')[0]
VOCAB = sorted(set(TEXTO))
V = len(VOCAB)
C2I = {c: i for i, c in enumerate(VOCAB)}
I2C = dict(enumerate(VOCAB))
DATOS = torch.tensor([C2I[c] for c in TEXTO], dtype=torch.long)

# -- Hiperparámetros: idénticos al Programa 8 ----------------
CTX, D, H, D_FF, N_BLQ = 24, 48, 4, 192, 2
LR, PASOS, BATCH, CLIP = 3e-3, 12000, 8, 1.0
STD_RES = 0.02 / math.sqrt(2 * N_BLQ)
torch.manual_seed(42)


class Bloque(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)               # γ y β incluidos
        self.Wq, self.Wk, self.Wv = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.Wo = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.W1 = nn.Linear(D, D_FF)
        self.W2 = nn.Linear(D_FF, D)

    def atencion(self, a):                       # a: (B, T, D)
        B, T, _ = a.shape
        separar = lambda x: x.view(B, T, H, D // H).transpose(1, 2)   # (B, H, T, DH)
        Q, K, Vm = separar(self.Wq(a)), separar(self.Wk(a)), separar(self.Wv(a))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(D // H)
        futuro = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        pesos = F.softmax(scores.masked_fill(futuro, float('-inf')), dim=-1)
        return self.Wo((pesos @ Vm).transpose(1, 2).reshape(B, T, D))

    def forward(self, X):
        X = X + self.atencion(self.ln1(X))                          # Pre-LN
        X = X + self.W2(F.gelu(self.W1(self.ln2(X)), approximate='tanh'))
        return X


class NanoLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.wpe = nn.Embedding(CTX, D)
        self.bloques = nn.ModuleList([Bloque() for _ in range(N_BLQ)])
        self.lnf = nn.LayerNorm(D)
        for nombre, p in self.named_parameters():   # inicialización de GPT-2
            if nombre.endswith('bias'):
                nn.init.zeros_(p)
            elif p.dim() == 2:
                std = STD_RES if ('Wo' in nombre or 'W2' in nombre) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx):                      # idx: (B, T)
        T = idx.shape[1]
        X = self.wte(idx) + self.wpe(torch.arange(T))
        for b in self.bloques:
            X = b(X)
        return self.lnf(X) @ self.wte.weight.T   # pesos compartidos


modelo = NanoLLM()
print(f'Vocabulario: {V} caracteres')
print(f'Total de parámetros: {sum(p.numel() for p in modelo.parameters()):,}')

# -- Entrenamiento: Adam + warmup/coseno + recorte ------------
opt = torch.optim.Adam(modelo.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)


def lr_factor(t, t_warmup=150, lr_min=1e-4):
    if t < t_warmup:
        return (t + 1) / t_warmup
    c = 0.5 * (1 + math.cos(math.pi * (t - t_warmup) / (PASOS - t_warmup)))
    return (lr_min + (LR - lr_min) * c) / LR


plan = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

for paso in range(PASOS):
    i = torch.randint(0, len(DATOS) - CTX - 1, (BATCH,))
    xb = torch.stack([DATOS[j:j + CTX] for j in i])
    yb = torch.stack([DATOS[j + 1:j + CTX + 1] for j in i])
    loss = F.cross_entropy(modelo(xb).reshape(-1, V), yb.reshape(-1))
    opt.zero_grad()
    loss.backward()                              # todo el backward en una línea
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), CLIP)
    opt.step()
    plan.step()
    if paso % 1000 == 0 or paso == PASOS - 1:
        print(f'Paso {paso:5d}: pérdida = {loss.item():.4f}   '
              f'(bits/char = {loss.item() / math.log(2):.2f})')


# -- Generación: temperatura + top-p (como el Programa 8) ------
@torch.no_grad()
def generar(semilla, n_chars=120, temperatura=0.8, top_p=0.9):
    tokens = [C2I.get(c, 0) for c in semilla[-CTX:]]
    for _ in range(n_chars):
        logits = modelo(torch.tensor([tokens[-CTX:]]))[0, -1]
        probs = F.softmax(logits / temperatura, dim=-1)
        orden = torch.argsort(probs, descending=True)
        acumulada = torch.cumsum(probs[orden], 0)
        n = int((acumulada < top_p).sum()) + 1
        nucleo = torch.zeros_like(probs)
        nucleo[orden[:n]] = probs[orden[:n]]
        tokens.append(int(torch.multinomial(nucleo / nucleo.sum(), 1)))
    return semilla + ''.join(I2C[t] for t in tokens[len(semilla[-CTX:]):])


print('\n=== TEXTO GENERADO POR EL NANO-LLM (PyTorch) ===')
print(generar('la inteligencia'))
print()
print(generar('los modelos de'))
