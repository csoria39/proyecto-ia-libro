# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 15.2 — RZ-2 con el resultado en orden natural
# 47+85=132 en lugar de 47+85=231: la primera cifra que
# escribe el modelo es ahora la de las centenas.
# Uso: python ejercicio_9t_2.py [semilla]
# =======================================================
fuente = open('programa_RZ2_suma.py', encoding='utf-8').read()
cambios = [
    ("str(a + b).zfill(3)[::-1] + '\\n'", "str(a + b).zfill(3) + '\\n'"),     # sin invertir
    ("pred = int(''.join(I2C[i] for i in r)[::-1])", "pred = int(''.join(I2C[i] for i in r))"),
]
for viejo, nuevo in cambios:
    assert viejo in fuente, viejo
    fuente = fuente.replace(viejo, nuevo)
fuente = fuente.split('# -- ¿Cómo lo hace?')[0]   # el análisis de atención supone el orden invertido
exec(fuente)
