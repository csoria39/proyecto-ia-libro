# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import torch
import torch.nn as nn
import torch.optim as optim
 
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]],         dtype=torch.float32)
 
# ============================================================
# CLASE MLP CON nn.Module
# Hereda de nn.Module para registrar los parametros
# automaticamente y ser compatible con todos los optimizadores
# ============================================================
class MLP(nn.Module):
    def __init__(self, n_ent, n_ocul, n_sal):
        super().__init__()
        # nn.Sequential encadena las capas en orden
        self.red = nn.Sequential(
            nn.Linear(n_ent, n_ocul),  # Z1 = X @ W1 + b1
            nn.ReLU(),                  # A1 = ReLU(Z1)
            nn.Linear(n_ocul, n_sal),  # Z2 = A1 @ W2 + b2
            nn.Sigmoid()               # y^ = sigmoid(Z2)
        )
    
    def forward(self, x):
        return self.red(x)
 
 
# Crear modelo, perdida y optimizador
torch.manual_seed(42)
modelo      = MLP(2, 8, 1)
criterio    = nn.BCELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)
 
n_params = sum(p.numel() for p in modelo.parameters())
print(f'Arquitectura: 2 -> 8 (ReLU) -> 1 (Sigmoid)')
print(f'Parametros:   {n_params}')
print(f'Estructura:')
for nombre, param in modelo.named_parameters():
    print(f'  {nombre:>20}: {list(param.shape)}')
print()
 
# ============================================================
# ENTRENAMIENTO: el bucle mas sencillo posible
# ============================================================
for epoca in range(5000):
    pred = modelo(X)
    loss = criterio(pred, y)
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()
    if epoca % 1000 == 0:
        print(f'Epoca {epoca:5d}: perdida = {loss.item():.6f}')
 
# Evaluacion final
print('\nXOR resuelto con MLP PyTorch:')
with torch.no_grad():
    pred = modelo(X)
correcto = 0
for xi, yi, pi in zip(X, y, pred):
    ok = 'v' if round(pi.item()) == int(yi.item()) else 'x'
    correcto += round(pi.item()) == int(yi.item())
    print(f'  XOR{xi.int().tolist()} = {int(yi.item())}  pred={pi.item():.4f}  {ok}')
print(f'Precision: {correcto}/4')
