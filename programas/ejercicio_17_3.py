# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
def costo_atencion(T, d, n_heads, bytes_por=2):
    """Memoria en GB y GFLOPs de la atención"""
    mem_gb = T * T * n_heads * bytes_por / (1024**3)
    gflops = 4 * T * T * d / 1e9
    return mem_gb, gflops
 
d, n_heads = 768, 12  # GPT-2 / BERT configuración
 
print(f'd={d}, n_heads={n_heads} (estilo GPT-2 small)\n')
print(f"{'Contexto T':>12}  {'Mem. aten. (GB)':>17}  {'GFLOPs':>10}  Factor vs ant.")
print('-' * 60)
 
T_vals = [512, 2048, 8192, 32768, 131072]
prev_mem = None
for T in T_vals:
    mem, gflops = costo_atencion(T, d, n_heads)
    factor = f'{mem/prev_mem:.1f}x' if prev_mem else '—'
    print(f'{T:>12,}  {mem:>17.3f}  {gflops:>10.2f}  {factor:>13}')
    prev_mem = mem
 
print()
print('Al doblar T, el costo se CUADRUPLICA (x4).')
print('Por eso ampliar la ventana de contexto es tan caro.')
print('Flash Attention: evita materializar la matriz TxT completa,')
print('calculándola por bloques. Mismo resultado, mucha menos memoria.')
