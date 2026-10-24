# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import torch
import torch.nn.functional as F
 
# ============================================================
# PARTE 1: Tensores y comparativa con NumPy
# ============================================================
import numpy as np
 
# NumPy
a_np = np.array([1.0, 2.0, 3.0])
# PyTorch — sintaxis casi identica
a_pt = torch.tensor([1.0, 2.0, 3.0])
 
print('NumPy: ', a_np)
print('PyTorch:', a_pt)
print('Suma NumPy: ', a_np.sum())
print('Suma PyTorch:', a_pt.sum())
print()
 
# ============================================================
# PARTE 2: Autograd — gradientes automáticos
# ============================================================
# Queremos calcular la derivada de L = (w*x - y)^2
# respecto a w, evaluada en w=2, x=3, y=10
 
x = torch.tensor(3.0)
y = torch.tensor(10.0)
w = torch.tensor(2.0, requires_grad=True)  # <-- activa autograd
 
# Forward: calcular la perdida
pred = w * x
L    = (pred - y) ** 2
 
print(f'w={w.item()}, x={x.item()}, y={y.item()}')
print(f'pred = w*x = {pred.item()}')
print(f'L = (pred-y)^2 = {L.item()}')
 
# Backward: PyTorch calcula dL/dw automáticamente
L.backward()
 
print(f'dL/dw (autograd): {w.grad.item()}')
 
# Verificación manual: dL/dw = 2*(w*x - y)*x = 2*(6-10)*3 = -24
dL_dw_manual = 2 * (w.item()*x.item() - y.item()) * x.item()
print(f'dL/dw (manual):   {dL_dw_manual}')
print(f'Coinciden: {abs(w.grad.item() - dL_dw_manual) < 1e-5}')
 
# ============================================================
# PARTE 3: Funciones de activación — torch vs implementación propia
# ============================================================
z = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
 
print(f"{'z':>6}  {'sigmoid':>10}  {'ReLU':>8}  {'tanh':>8}")
print('-' * 38)
for zi, s, r, t in zip(z,
                        torch.sigmoid(z),
                        F.relu(z),
                        torch.tanh(z)):
    print(f'{zi.item():>6.1f}  {s.item():>10.4f}  {r.item():>8.2f}  {t.item():>8.4f}')
 
# Softmax: devuelve distribucion de probabilidad
logits = torch.tensor([2.0, 1.0, 0.5, -0.5])
probs  = F.softmax(logits, dim=0)
print(f'Softmax: {probs.numpy().round(4)}  suma={probs.sum().item():.6f}')
