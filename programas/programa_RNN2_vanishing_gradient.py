# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(0)
 
# ================================================================
# DEMOSTRACION DEL VANISHING GRADIENT
# Simulamos como fluye el gradiente hacia atras en un RNN.
# En cada paso, el gradiente se multiplica por W_hh.T
# y por la derivada de tanh (entre 0 y 1).
# ================================================================
 
H    = 32   # dimension del estado oculto
T    = 60   # longitud de la secuencia
 
# Cuatro RNNs con distintas inicializaciones de W_hh
configs = {
    'W_hh pequena  (escala 0.05)':      np.random.randn(H,H) * 0.05,
    'W_hh moderada (escala 1/sqrt(H))': np.random.randn(H,H) * (1.0/np.sqrt(H)),
    'W_hh grande   (escala 0.3)':       np.random.randn(H,H) * 0.3,
    'W_hh muy grande (escala 0.5)':     np.random.randn(H,H) * 0.5,
}
 
def simular_gradiente(W_hh, T):
    """
    Simula el flujo del gradiente hacia atras en un RNN.
    En cada paso backward, el gradiente se multiplica por W_hh.T
    y por la derivada de tanh (simplificada a 0.5 de media).
    """
    grad   = np.ones(H) / np.sqrt(H)  # gradiente inicial normalizado
    normas = [float(np.linalg.norm(grad))]
    
    for t in range(T):
        # Derivada de tanh promediada: ~0.5 en un estado tipico
        d_tanh = 0.5 * np.ones(H)
        # Backprop un paso atras
        grad   = (W_hh.T @ grad) * d_tanh
        normas.append(float(np.linalg.norm(grad)))
    
    return normas
 
print('=== VANISHING / EXPLODING GRADIENT EN RNN ===')
print(f'Estado oculto H={H}, secuencia T={T}')
print()
print('Norma del gradiente a distintas distancias del final:\n')
print(f"{'Distancia':>10}", end='')
for nombre in configs: print(f'  {nombre.split("(")[0].strip():>17}', end='')
print()
print('-' * (10 + 19*len(configs)))
 
historicos = {}
for nombre, W in configs.items():
    historicos[nombre] = simular_gradiente(W, T)
 
pasos_mostrar = [0, 1, 2, 5, 10, 20, 30, 50, 60]
for dist in pasos_mostrar:
    print(f'{dist:>10}', end='')
    for nombre in configs:
        v = historicos[nombre][dist]
        if v < 1e-10:   fmt = '< 1e-10  (MUERTO)'
        elif v > 1e4:   fmt = f'{v:.1e}  (EXPLOTA)'
        else:           fmt = f'{v:.6f}'
        print(f'  {fmt:>17}', end='')
    print()
 
# Visualizacion ASCII del decaimiento
print('\nVisualizacion del gradiente (W_hh pequena — caso tipico):')
print('Norma del gradiente al retropropagar hacia atras:')
normas = historicos['W_hh pequena  (escala 0.05)']
vmax   = normas[0]
for dist in range(0, T+1, 5):
    v     = normas[dist]
    ratio = v / vmax if vmax > 0 else 0
    barra = '#' * int(ratio * 40)
    arrow = '<-- el gradiente es casi NULO' if v < 1e-4 else ''
    print(f'  Distancia {dist:>3}: {v:.2e}  {barra} {arrow}')
 
# Contraste: la derivada REAL de tanh, 1 - h^2, en una secuencia de verdad
# (entradas al azar ya proyectadas; mismo W_hh que arriba)
np.random.seed(1)
X_in = np.random.randn(T, H) * 0.5
 
def gradiente_real(W_hh):
    h, hs = np.zeros(H), []
    for t in range(T):                                # forward real
        h = np.tanh(W_hh @ h + X_in[t]); hs.append(h)
    grad, normas, dts = np.ones(H) / np.sqrt(H), [1.0], []
    for t in reversed(range(T)):                      # backward real
        d_tanh = 1 - hs[t]**2
        dts.append(d_tanh.mean())
        grad = W_hh.T @ (grad * d_tanh)
        normas.append(float(np.linalg.norm(grad)))
    return normas, float(np.mean(dts))
 
print('\nContraste con la derivada real de tanh (1 - h^2):')
print(f"{'Configuracion':>18}  {'media 1-h^2':>11}  {'dist 10 (0.5)':>14}  {'dist 10 (real)':>14}  {'dist 30 (real)':>14}")
for nombre, W in configs.items():
    reales, media = gradiente_real(W)
    print(f"{nombre.split('(')[0].strip():>18}  {media:>11.3f}  {historicos[nombre][10]:>14.2e}  {reales[10]:>14.2e}  {reales[30]:>14.2e}")
 
print()
print('Conclusion:')
print('  Con W_hh pequena o moderada, el gradiente desaparece en pocos pasos,')
print('  tanto con la derivada fija (0.5) como con la real (1 - h^2):')
print('  los tokens lejanos apenas contribuyen al aprendizaje.')
print()
print('  Con escalas mayores, la derivada importa: con 0.5 fijo, el gradiente')
print('  decae (0.3) o EXPLOTA (0.5); con la real, que baja cuando tanh se')
print('  satura, se mantiene en el mismo orden de magnitud a distancia 30.')
print('  Una RNN simple vive en un equilibrio fragil entre ambos extremos.')
print()
print('  Esta es la razon fundamental por la que el Transformer')
print('  supero a los RNNs: la atencion conecta directamente')
print('  cualquier par de tokens sin importar la distancia.')
