# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
modelos = {
    'Modelo A': 1.2,
    'Modelo B': 2.5,
    'Modelo C': 0.3,
}
 
print('=== INTERPRETACION DE PERPLEJIDAD ===\n')
print(f"{'Modelo':>10}  {'Perdida':>9}  {'PPL':>8}  Interpretacion")
print('-' * 60)
 
for nombre, perdida in modelos.items():
    ppl = np.exp(perdida)
    if ppl < 2:
        interp = 'Excelente: casi siempre acierta el token correcto'
    elif ppl < 5:
        interp = 'Muy bueno: duda entre pocas opciones razonables'
    elif ppl < 15:
        interp = 'Aceptable: bastante incertidumbre'
    else:
        interp = 'Pobre: el modelo apenas ha aprendido patrones'
    print(f'{nombre:>10}  {perdida:>9.2f}  {ppl:>8.2f}  {interp}')
 
print()
print('Comparacion relativa:')
ppls = {k: np.exp(v) for k,v in modelos.items()}
mejor = min(ppls, key=ppls.get)
peor  = max(ppls, key=ppls.get)
print(f'  {mejor} es {ppls[peor]/ppls[mejor]:.1f}x mejor que {peor}')
