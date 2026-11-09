# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def metricas_equidad(y_real, y_pred, grupo):
    '''Calcula las principales metricas de equidad por grupo.'''
    resultados = {}
    for g in np.unique(grupo):
        mask = grupo == g
        yr, yp = y_real[mask], y_pred[mask]
        vp = ((yp==1)&(yr==1)).sum()  # verdadero positivo
        fp = ((yp==1)&(yr==0)).sum()  # falso positivo
        vn = ((yp==0)&(yr==0)).sum()  # verdadero negativo
        fn = ((yp==0)&(yr==1)).sum()  # falso negativo
        resultados[g] = {
            'tasa_seleccion': yp.mean(),
            # sin casos en el denominador la métrica no está definida: nan, no 0
            'TPR (sensibil.)': vp/(vp+fn) if (vp+fn) else float('nan'),
            'FPR (1-espec.)':  fp/(fp+vn) if (fp+vn) else float('nan'),
            'precision_PPV':   vp/(vp+fp) if (vp+fp) else float('nan'),
            'exactitud':       (vp+vn)/len(yr),
        }
    return resultados
 
 
# Simulacion: sistema de seleccion de candidatos sesgado
np.random.seed(7)
n = 1000
grupo  = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
y_real = np.random.binomial(1, 0.45, n)  # capacidad real similar
 
# Modelo sesgado: el grupo 0 necesita superar un umbral mas alto
y_pred = np.array([
    int(np.random.random() > (0.58 if grupo[i]==0 else 0.42))
    for i in range(n)
])
 
met = metricas_equidad(y_real, y_pred, grupo)
 
print('=== METRICAS DE EQUIDAD: SISTEMA DE SELECCION ===\n')
print(f"{'Metrica':>22}  {'Grupo 0':>10}  {'Grupo 1':>10}  {'Diferencia':>12}  {'Alerta?':>8}")
print('-' * 70)
 
UMBRAL_ALERTA = 0.05
metricas_mostrar = [
    'tasa_seleccion', 'TPR (sensibil.)', 'FPR (1-espec.)',
    'precision_PPV', 'exactitud'
]
for m in metricas_mostrar:
    v0  = met[0][m]
    v1  = met[1][m]
    dif = v0 - v1
    alerta = '!!! SI' if abs(dif) > UMBRAL_ALERTA else 'no'
    print(f'{m:>22}  {v0:>10.1%}  {v1:>10.1%}  {dif:>+12.1%}  {alerta:>8}')
 
print()
print('Interpretacion:')
print('  Paridad demografica: tasa_seleccion igual para todos los grupos')
print('  Equal Opportunity:   TPR igual (misma oportunidad si eres valido)')
print('  Equalized Odds:      TPR Y FPR iguales (simetria total)')
print()
print('Nota: estas metricas son INCOMPATIBLES entre si en muchos casos.')
print('Elegir cual garantizar es una decision etica, no tecnica.')
