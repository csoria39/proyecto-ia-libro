# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
def tokens_optimos(n_params):
    """Tokens óptimos según Chinchilla: 20 x n_params"""
    return 20 * n_params
 
TOKENS_POR_LIBRO = 130_000    # tokens BPE aprox.
TOKENS_POR_WEB   = 500        # tokens por página web típica
 
modelos = [
    ('LLaMA-7B',     7_000_000_000),
    ('LLaMA-13B',   13_000_000_000),
    ('LLaMA-70B',   70_000_000_000),
    ('Hipotético-700B', 700_000_000_000),
]
 
print('Tokens óptimos según Chinchilla (Hoffmann et al., 2022)\n')
print(f"{'Modelo':>20}  {'Tokens óptimos':>16}  {'Libros equiv.':>15}")
print('-' * 58)
for nombre, n in modelos:
    t = tokens_optimos(n)
    libros = t / TOKENS_POR_LIBRO
    print(f'{nombre:>20}  {t:>16,.0f}  {libros:>15,.0f}')
 
print()
print('Comparación: GPT-3 fue entrenado con ~300B tokens')
print('pero tiene 175B params -> solo 1.7x (infraentrenado)')
print('LLaMA-2-7B fue entrenado con 2T tokens -> 285x (sobreentrenado')
print('para tamaño, pero mejor en inference!)')
