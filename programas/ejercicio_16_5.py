# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# EJERCICIO 16.5 — ¿En qué se va el tiempo de un token con KV-cache?
# Con el cache, cada token nuevo atraviesa los 12 bloques y la proyección
# final sobre los 50 257 tokens. Medimos cada parte por separado.
import time
import numpy as np

fuente = open('programa_G3_kvcache.py', encoding='utf-8').read()
exec(fuente.split('# 1. ¿Da exactamente lo mismo')[0])     # funciones del Programa G-3

ids = codificar('The quick brown fox jumps over the lazy dog because it')
kv = nuevo_cache()
prefill(ids, kv)
x = params['wte'][ids[-1]] + params['wpe'][len(ids) - 1]


def medir(escala, rep=20):
    """Un token con el cache lleno: tiempo de los 12 bloques y de la salida."""
    t_bloques = t_salida = 0.0
    for _ in range(rep):
        t0 = time.time()
        y = x
        for k in range(N_BLQ):              # como forward_un_token, sin guardar nada
            a, _ = layer_norm_fwd(y, params[f'ln1_g{k}'], params[f'ln1_b{k}'])
            q, kk, v = ((a @ params[f'W{n}{k}'] + params[f'b{n}{k}']).reshape(H, DH) for n in 'qkv')
            K, Vm = kv[k]['K'][:, :len(ids)], kv[k]['V'][:, :len(ids)]
            pesos = softmax(np.einsum('hd,htd->ht', q, K) / escala)
            att = np.einsum('ht,htd->hd', pesos, Vm).reshape(D)
            y = y + att @ params[f'Wo{k}'] + params[f'bo{k}']
            m, _ = layer_norm_fwd(y, params[f'ln2_g{k}'], params[f'ln2_b{k}'])
            y = y + gelu(m @ params[f'W1{k}'] + params[f'b1{k}']) @ params[f'W2{k}'] + params[f'b2{k}']
        t1 = time.time()
        xf, _ = layer_norm_fwd(y, params['lnf_g'], params['lnf_b'])
        logits = xf @ params['wte'].T        # la proyección sobre el vocabulario
        t2 = time.time()
        t_bloques += t1 - t0
        t_salida += t2 - t1
    print(f'   pesos de atención en {pesos.dtype}, vía residual en {y.dtype}')
    print(f'   12 bloques:    {t_bloques / rep * 1000:6.1f} ms por token')
    print(f'   salida (wteᵀ): {t_salida / rep * 1000:6.1f} ms por token')
    return logits


n_bloques = N_BLQ * (4 * D * D + 2 * D * 4 * D)
print(f'Pesos de los 12 bloques: {n_bloques / 1e6:.1f} millones; '
      f'de la salida: {D * V / 1e6:.1f} millones')
print('\nCon np.sqrt(DH), como en el Programa 8:')
l64 = medir(np.sqrt(DH))
# np.sqrt() devuelve un número float64 de NumPy, y NumPy convierte a float64 todo
# lo que se opera con él. Pasa con np.sqrt(DH) y con C_GELU = np.sqrt(2 / np.pi).
print('\nCon esas dos constantes en float32:')
C_GELU = np.float32(C_GELU)                  # gelu() la lee de aquí
l32 = medir(np.float32(np.sqrt(DH)))
print(f'\nDiferencia máxima entre los logits: {np.abs(l64 - l32).max():.1e}; '
      f'mismo token elegido: {"sí" if l64.argmax() == l32.argmax() else "no"}')
