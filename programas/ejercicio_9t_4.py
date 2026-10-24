# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 15.4 — RZ-3 con cadenas más largas
# 16 hechos por problema (antes 12). El examen llega hasta
# 14 pasos; se sigue entrenando SOLO con 1 a 4 pasos.
# Más hechos = prompt más largo: hay que ampliar CTX (y con
# él wpe) y el relleno, para que las posiciones finales
# también se vean al entrenar.
# Uso: python ejercicio_9t_4.py [pasos]   (por defecto 2000)
# =======================================================
import sys
PASOS_EJ = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
sys.argv = sys.argv[:1]          # RZ-3 lee sus propios argumentos
fuente = open('programa_RZ3_deduce_letras.py', encoding='utf-8').read()
cambios = [
    ("N_HECHOS, PASOS_MAX_ENTRENO = 12, 4", "N_HECHOS, PASOS_MAX_ENTRENO = 16, 4"),
    ("CTX, D, D_FF, N_BLQ, H = 56, 48, 192, 2, 2", "CTX, D, D_FF, N_BLQ, H = 72, 48, 192, 2, 2"),
    ("LR, PASOS, BATCH, CLIP, RELLENO_MAX = 3e-3, 2000, 64, 1.0, 8",
     f"LR, PASOS, BATCH, CLIP, RELLENO_MAX = 3e-3, {PASOS_EJ}, 64, 1.0, 12"),
    ("def resolver(prompts, max_chars=12):", "def resolver(prompts, max_chars=16):"),
    ("for L in (2, 6, 9):", "for L in (3, 9, 13):"),
]
for viejo, nuevo in cambios:
    assert viejo in fuente, viejo
    fuente = fuente.replace(viejo, nuevo)
exec(fuente)
