# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(42)
 
# ================================================================
# ESCENARIO: sistema de scoring crediticio
# Dos grupos con las MISMAS caracteristicas objetivas,
# pero el Grupo 0 tiene historicamente menos acceso al credito.
# Ese sesgo historico queda 'congelado' en los datos, DE DOS FORMAS:
# (1) menor probabilidad de aprobacion, y (2) un historial crediticio
# registrado peor (aunque la capacidad real de pago sea identica).
# ================================================================
 
def generar_dataset(n=1000):
    datos = []
    for _ in range(n):
        grupo     = np.random.randint(0, 2)
        ingresos  = np.clip(np.random.normal(0.5, 0.15), 0, 1)
        historial_real = np.clip(np.random.normal(0.5, 0.15), 0, 1)
        
        # Probabilidad justa basada solo en caracteristicas objetivas
        prob_justa = 0.4 * ingresos + 0.4 * historial_real + 0.05
        
        # SESGO HISTORICO: el grupo 0 recibio credito con 35% menos
        # de frecuencia que el grupo 1, con identicas caracteristicas.
        # Este sesgo esta 'grabado' en los datos de entrenamiento.
        if grupo == 0:
            prob = prob_justa * 0.65   # penalizacion injusta
        else:
            prob = prob_justa
        
        # VARIABLE PROXY: el historial REGISTRADO esta sesgado a la baja
        # para el grupo 0 (mismo poder adquisitivo real, peor historial
        # observable por falta historica de acceso al credito).
        historial_registrado = historial_real * 0.7 if grupo == 0 else historial_real
        
        aprobado = int(np.random.random() < np.clip(prob, 0, 1))
        datos.append([grupo, ingresos, historial_registrado, aprobado])
    return np.array(datos)
 
 
datos = generar_dataset(1000)
grupo    = datos[:, 0].astype(int)
X        = datos[:, 1:3]            # ingresos, historial (SIN grupo)
y        = datos[:, 3].astype(int)
 
# ================================================================
# ENTRENAMIENTO: regression logistica sin usar 'grupo' como feature
# El desarrollador cree que su modelo es justo porque no usa el grupo
# ================================================================
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
 
w = np.zeros(2); b = 0.0
for _ in range(3000):
    pred = sigmoid(X @ w + b)
    err  = pred - y
    w   -= 0.5 * (X.T @ err) / len(y)   # lr aumentado de 0.05: asegura convergencia
    b   -= 0.5 * err.mean()
 
pred_bin = (sigmoid(X @ w + b) > 0.5).astype(int)
 
# ================================================================
# ANALISIS DE SESGO: comparar resultados por grupo
# ================================================================
print('=== ANALISIS DE SESGO EN EL MODELO DE CREDITO ===\n')
print('El modelo NO usa el grupo como variable de entrada.')
print('Aun asi, aprende el sesgo historico de los datos.\n')
 
print(f"{'Grupo':>8}  {'Aprobados reales':>18}  {'Aprobados modelo':>18}  {'Diferencia':>12}")
print('-' * 62)
for g in [0, 1]:
    mask = grupo == g
    real    = y[mask].mean()
    modelo  = pred_bin[mask].mean()
    dif     = modelo - real
    print(f'{g:>8}  {real:>18.1%}  {modelo:>18.1%}  {dif:>+12.1%}')
 
# Paridad demografica
t0 = pred_bin[grupo==0].mean()
t1 = pred_bin[grupo==1].mean()
print(f'\nParidad demografica: diferencia del {abs(t0-t1):.1%}')
print(f'(0% = paridad perfecta, > 5% = señal de alerta)')
print(f'\nPrecision global: {(pred_bin==y).mean():.1%}')
print('La precision global oculta la disparidad entre grupos.')
print('Un desarrollador desatento pensaria que el modelo es justo.')
