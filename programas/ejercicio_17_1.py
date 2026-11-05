# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
def memoria_gb(n_params, precision='float16'):
    bytes_por = {'float32':4, 'float16':2, 'int8':1, 'int4':0.5}
    return n_params * bytes_por[precision] / (1024**3)
 
modelos = [
    ('Nano-LLM (nuestro)',     60_000),
    ('GPT-2 small',       124_000_000),
    ('GPT-2 XL',        1_558_000_000),
    ('LLaMA-7B',        7_000_000_000),
    ('LLaMA-13B',      13_000_000_000),
    ('GPT-3 175B',    175_000_000_000),
]
 
print(f"{'Modelo':>25}  {'FP32 (GiB)':>10}  {'FP16 (GiB)':>10}  {'INT4 (GiB)':>10}")
print('-' * 60)
for nombre, n in modelos:
    g32 = memoria_gb(n, 'float32')
    g16 = memoria_gb(n, 'float16')
    g4  = memoria_gb(n, 'int4')
    print(f'{nombre:>25}  {g32:>10.3f}  {g16:>10.3f}  {g4:>10.3f}')
 
print()
print('Una GPU A100: 80 GB  |  RTX 4090 (gaming): 24 GB')
print('INT4 permite ejecutar LLaMA-7B en una GPU de 4 GB!')
