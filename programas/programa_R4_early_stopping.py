# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(42)
 
# =============================================================
# CLASE EARLY STOPPING
# Monitoriza el error de validacion y detiene el entrenamiento
# cuando no mejora durante 'paciencia' epocas consecutivas.
# =============================================================
class EarlyStopping:
    def __init__(self, paciencia=50, delta=1e-4):
        """
        paciencia: epocas sin mejora antes de parar
        delta:     mejora minima para considerar que ha habido progreso
        """
        self.paciencia     = paciencia
        self.delta         = delta
        self.mejor_val     = float('inf')
        self.contador      = 0
        self.mejor_epoca   = 0
        self.mejores_pesos = None
    
    def registrar(self, val_loss, pesos, epoca):
        """
        Llamar en cada epoca con el error de validacion.
        Devuelve True si hay que parar, False si continuar.
        """
        if val_loss < self.mejor_val - self.delta:
            # Ha mejorado: guardar estos pesos
            self.mejor_val   = val_loss
            self.contador    = 0
            self.mejor_epoca = epoca
            # Copia profunda de los pesos
            self.mejores_pesos = {k: v.copy() for k, v in pesos.items()}
            return False  # seguir entrenando
        else:
            # No ha mejorado: incrementar contador
            self.contador += 1
            if self.contador >= self.paciencia:
                return True  # parar
            return False     # todavia tenemos paciencia
 
 
# MLP simple para la demo
def relu(z):   return np.maximum(0, z)
def d_relu(z): return (z > 0).astype(float)
 
class MLP:
    def __init__(self, n_oc, s=42):
        np.random.seed(s)
        self.W1=np.random.randn(1,n_oc)*np.sqrt(2); self.b1=np.zeros(n_oc)
        self.W2=np.random.randn(n_oc,1)*0.1;        self.b2=np.zeros(1)
        self.m={'W1':np.zeros_like(self.W1),'b1':np.zeros_like(self.b1),'W2':np.zeros_like(self.W2),'b2':np.zeros_like(self.b2)}
        self.v={k:np.zeros_like(x) for k,x in self.m.items()}; self.t=0
    def pesos(self): return {'W1':self.W1,'b1':self.b1,'W2':self.W2,'b2':self.b2}
    def cargar(self, d): self.W1=d['W1'].copy();self.b1=d['b1'].copy();self.W2=d['W2'].copy();self.b2=d['b2'].copy()
    def forward(self, X): self.X=X; self.Z1=X@self.W1+self.b1; self.A1=relu(self.Z1); return self.A1@self.W2+self.b2
    def backward(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        dW2=self.A1.T@g; db2=g.sum(0); dA1=g@self.W2.T          # g ya trae el 1/N del MSE
        dZ1=dA1*d_relu(self.Z1); dW1=self.X.T@dZ1; db1=dZ1.sum(0)
        self.t+=1
        for n_,g_ in [('W1',dW1),('b1',db1),('W2',dW2),('b2',db2)]:
            self.m[n_]=b1*self.m[n_]+(1-b1)*g_; self.v[n_]=b2*self.v[n_]+(1-b2)*g_**2
            mh=self.m[n_]/(1-b1**self.t); vh=self.v[n_]/(1-b2**self.t)
            getattr(self,n_)[...] -= lr*mh/(np.sqrt(vh)+eps)
    def train_ep(self, X, y, lr):
        p=self.forward(X); e=p-y.reshape(-1,1); self.backward(2*e/len(X),lr); return float(np.mean(e**2))
    def mse(self, X, y): return float(np.mean((self.forward(X)-y.reshape(-1,1))**2))
 
 
# Dataset
def gen(n, s): np.random.seed(s); X=np.sort(np.random.uniform(0,1,n)).reshape(-1,1); return X, np.sin(2*np.pi*X.ravel())+np.random.normal(0,0.25,n)
X_tr, y_tr   = gen(20, 42)
X_val, y_val = gen(200, 99)
 
# =============================================================
# ENTRENAMIENTO CON EARLY STOPPING
# =============================================================
red = MLP(256)
es  = EarlyStopping(paciencia=1000, delta=1e-5)   # aumentado de 100: evita parar en una meseta temprana
 
print('=== ENTRENAMIENTO CON EARLY STOPPING ===')
print(f'Modelo: 256 neuronas | Paciencia: 1000 epocas\n')
print(f"{'Epoca':>7}  {'Train':>10}  {'Val':>10}  Evento")
print('-'*52)
 
epocas_totales = 30_000   # aumentado de 10_000: con Adam hace falta mas margen
for epoca in range(epocas_totales):
    lt = red.train_ep(X_tr, y_tr, lr=0.01)   # ajustado para Adam (antes 0.005)
    lv = red.mse(X_val, y_val)
    
    parar = es.registrar(lv, red.pesos(), epoca)
    
    # Mostrar hitos importantes
    if epoca < 5 or epoca % 500 == 0 or parar:
        evento = ''
        if es.contador == 0: evento = '<-- nuevo mejor'
        if parar:            evento = '<-- PARADA ANTICIPADA'
        print(f'{epoca:>7}  {lt:>10.5f}  {lv:>10.5f}  {evento}')
    
    if parar:
        break
 
# Restaurar los mejores pesos
red.cargar(es.mejores_pesos)
 
print(f'\nResumen:')
print(f'  Epocas entrenadas:  {epoca + 1} de {epocas_totales} posibles')
print(f'  Mejor epoca:        {es.mejor_epoca}')
print(f'  Mejor val (guardado): {es.mejor_val:.5f}')
print(f'  Val con pesos finales s/ ES: podria ser mucho peor')
print(f'  Val restaurando mejores pesos: {red.mse(X_val, y_val):.5f}')
