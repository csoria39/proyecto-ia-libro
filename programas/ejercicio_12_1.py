# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
def params_transformer(d, d_ff=None, n_heads=1):
    d_ff = d_ff or d * 4
    """Cuenta parámetros de un bloque transformer"""
    aten  = 4 * d * d + 4 * d      # Wq, Wk, Wv, Wo y sus sesgos
    ffn   = d*d_ff + d_ff + d_ff*d + d  # W1+b1 + W2+b2
    lnorm = 2 * 2 * d              # 2 LayerNorm x (gamma + beta)
    total = aten + ffn + lnorm
    return {'atención': aten, 'FFN': ffn, 'LayerNorm': lnorm, 'total': total}
 
configs = [
    ('Nuestro bloque (d=16)',     16,   64,  1),
    ('Nano-LLM completo (2 bl.)', 48,  192,  2),   # 28 272 por bloque, como el Programa 8
    ('GPT-2 small (d=768, 12bl)',768, 3072, 12),
    ('GPT-3 (d=12288, 96bl)',  12288,49152, 96),
]
 
for nombre, d, d_ff, n_bl in configs:
    r = params_transformer(d, d_ff)
    total_modelo = r['total'] * n_bl
    print(f'{nombre}')
    print(f'  Atención: {r["atención"]:>10,}  FFN: {r["FFN"]:>10,}  LN: {r["LayerNorm"]:>6,}')
    print(f'  Por bloque: {r["total"]:>10,}   x {n_bl} bloques = {total_modelo:>15,} total')
    print()
