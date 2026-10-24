# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 15.3 — RZ-2 aprende a RESTAR
# a − b con a ≥ b (dos cifras), resultado de 2 cifras escrito
# al revés: 85−47=38 se escribe «85-47=83». Hay 5 050 restas
# posibles; 1 000 se apartan para el examen.
# Uso: python ejercicio_9t_3.py [semilla]
# =======================================================
fuente = open('programa_RZ2_suma.py', encoding='utf-8').read()
cambios = [
    ("VOCAB = list('0123456789+=\\n')", "VOCAB = list('0123456789-=\\n')"),
    ("return f'{a:02d}+{b:02d}=' + str(a + b).zfill(3)[::-1] + '\\n'",
     "return f'{a:02d}-{b:02d}=' + str(a - b).zfill(2)[::-1] + '\\n'"),
    ("pares = [(a, b) for a in range(100) for b in range(100)]",
     "pares = [(a, b) for a in range(100) for b in range(a + 1)]"),
    ("TEST, TRAIN = pares[:2000], pares[2000:2000 + N_ENTRENO]",
     "TEST, TRAIN = pares[:1000], pares[1000:1000 + N_ENTRENO]"),
    ("PESO = np.zeros(9); PESO[5:8] = 1", "PESO = np.zeros(8); PESO[5:7] = 1"),
    ("CTX, D, D_FF, N_BLQ = 9, 48, 192, 2", "CTX, D, D_FF, N_BLQ = 8, 48, 192, 2"),
    ("    for _ in range(3):", "    for _ in range(2):"),
    ("    return seq[:, 6:9]", "    return seq[:, 6:8]"),
    ("return (resolver(X) == Y[:, 5:8]).all(1).mean()", "return (resolver(X) == Y[:, 5:7]).all(1).mean()"),
    ("print(f'{a:2d} + {b:2d} = {pred:3d}   {\"✓\" if pred == a + b else \"✗ (correcto: \" + str(a + b) + \")\"}')",
     "print(f'{a:2d} - {b:2d} = {pred:3d}   {\"✓\" if pred == a - b else \"✗ (correcto: \" + str(a - b) + \")\"}')"),
    ("=== EXAMEN CON SUMAS QUE EL MODELO NUNCA VIO ===", "=== EXAMEN CON RESTAS QUE EL MODELO NUNCA VIO ==="),
    ("print(f'\\nAciertos en las {len(TEST)} sumas no vistas", "print(f'\\nAciertos en las {len(TEST)} restas no vistas"),
    ("sumas de entrenamiento: {len(TRAIN)}", "restas de entrenamiento: {len(TRAIN)}"),
    ("sumas de examen (no vistas)", "restas de examen (no vistas)"),
    ("(exactitud = suma entera correcta)", "(exactitud = resta entera correcta)"),
    ("| sumas NUNCA vistas", "| restas NUNCA vistas"),
]
for viejo, nuevo in cambios:
    assert viejo in fuente, viejo
    fuente = fuente.replace(viejo, nuevo)
fuente = fuente.split('# -- ¿Cómo lo hace?')[0]
exec(fuente)
