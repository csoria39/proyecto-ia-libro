# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)   # pesos iniciales reproducibles
 
# Datos: puerta logica AND
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[0],[0],[1]],         dtype=torch.float32)
 
# ============================================================
# EL MODELO: una sola neurona = nn.Linear(2, 1)
# PyTorch crea automaticamente los pesos W (2x1) y sesgo b (1)
# ============================================================
neurona = nn.Linear(2, 1)
 
# BCEWithLogitsLoss = sigmoid + BCE en una sola operacion
# (mas estable numericamente que aplicarlos por separado)
criterio    = nn.BCEWithLogitsLoss()
optimizador = optim.SGD(neurona.parameters(), lr=0.5)
 
print('=== NEURONA AND CON PYTORCH ===')
print(f'Parametros iniciales:')
print(f'  Pesos: {neurona.weight.data.numpy().round(3)}')
print(f'  Sesgo: {neurona.bias.data.item():.4f}')
print()
 
# ============================================================
# BUCLE DE ENTRENAMIENTO
# Comparacion con NumPy:
#   NumPy:   ~15 lineas de gradientes manuales
#   PyTorch: 4 lineas (zero_grad, forward, backward, step)
# ============================================================
for epoca in range(10_000):
    logits = neurona(X)          # forward
    loss   = criterio(logits, y) # perdida
    optimizador.zero_grad()      # limpiar gradientes
    loss.backward()              # backprop automatico
    optimizador.step()           # actualizar pesos
 
    if epoca % 2000 == 0:
        print(f'Epoca {epoca:5d}: perdida = {loss.item():.6f}')
 
# Resultados finales
print(f'\nPesos finales: {neurona.weight.data.numpy().round(3)}')
print(f'Sesgo final:   {neurona.bias.data.item():.4f}')
print('\nPrueba final:')
with torch.no_grad():
    probs = torch.sigmoid(neurona(X))
for xi, yi, pi in zip(X, y, probs):
    ok = 'v' if round(pi.item()) == int(yi.item()) else 'x'
    print(f'  AND{xi.int().tolist()} = {int(yi.item())}  pred={pi.item():.4f}  {ok}')
