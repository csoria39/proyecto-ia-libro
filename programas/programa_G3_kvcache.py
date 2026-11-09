# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA G-3 — El KV-cache, implementado y medido
# Capítulo 16, sección 16.7
#
# generar() del Programa 8 bis recalcula, para cada token nuevo,
# el forward de TODA la secuencia. Pero con la máscara causal las
# claves (K) y los valores (V) de los tokens anteriores no cambian
# nunca: se pueden guardar y reutilizar. Así, cada paso solo
# procesa UN token. Es el «KV-cache» que usan todos los LLM.
# Requisitos: programa_8bis_gpt2_real.py y la carpeta gpt2/.
# =======================================================
import time
import numpy as np

fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])     # pesos, tokenizador, forward() y generar()


def nuevo_cache():
    """Espacio para las K y V de hasta CTX tokens en cada bloque: (H, CTX, DH)."""
    return [{'K': np.zeros((H, CTX, DH), np.float32), 'V': np.zeros((H, CTX, DH), np.float32)}
            for _ in range(N_BLQ)]


def forward_un_token(token, t, kv):
    """Logits del token que sigue al token `token`, que está en la posición t.
    Usa las K y V guardadas de las posiciones 0..t-1 y guarda las suyas."""
    x = params['wte'][token] + params['wpe'][t]                     # (D,)
    for k in range(N_BLQ):
        a, _ = layer_norm_fwd(x, params[f'ln1_g{k}'], params[f'ln1_b{k}'])
        q, kk, v = ((a @ params[f'W{n}{k}'] + params[f'b{n}{k}']).reshape(H, DH) for n in 'qkv')
        kv[k]['K'][:, t], kv[k]['V'][:, t] = kk, v                  # se guardan para siempre
        K, Vm = kv[k]['K'][:, :t + 1], kv[k]['V'][:, :t + 1]        # (H, t+1, DH)
        pesos = softmax(np.einsum('hd,htd->ht', q, K) / np.sqrt(DH))  # (H, t+1): sin máscara:
        att = np.einsum('ht,htd->hd', pesos, Vm).reshape(D)         # solo existe el pasado
        x = x + att @ params[f'Wo{k}'] + params[f'bo{k}']
        m, _ = layer_norm_fwd(x, params[f'ln2_g{k}'], params[f'ln2_b{k}'])
        x = x + gelu(m @ params[f'W1{k}'] + params[f'b1{k}']) @ params[f'W2{k}'] + params[f'b2{k}']
    xf, _ = layer_norm_fwd(x, params['lnf_g'], params['lnf_b'])
    return xf @ params['wte'].T                                      # (V,)


def prefill(ids, kv):
    """Procesa la semilla ENTERA de una vez (como forward) y guarda sus K y V.
    Los LLM reales lo hacen así: «prefill» de la pregunta y después «decode»,
    token a token, de la respuesta."""
    T = len(ids)
    X = params['wte'][ids] + params['wpe'][:T]                       # (T, D)
    for k in range(N_BLQ):
        a, _ = layer_norm_fwd(X, params[f'ln1_g{k}'], params[f'ln1_b{k}'])
        Q, K, Vm = ((a @ params[f'W{n}{k}'] + params[f'b{n}{k}']).reshape(T, H, DH).transpose(1, 0, 2)
                    for n in 'qkv')                                  # (H, T, DH)
        kv[k]['K'][:, :T], kv[k]['V'][:, :T] = K, Vm
        pesos = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(DH) + MASCARA_CAUSAL[:T, :T])
        att = (pesos @ Vm).transpose(1, 0, 2).reshape(T, D)
        X = X + att @ params[f'Wo{k}'] + params[f'bo{k}']
        m, _ = layer_norm_fwd(X, params[f'ln2_g{k}'], params[f'ln2_b{k}'])
        X = X + gelu(m @ params[f'W1{k}'] + params[f'b1{k}']) @ params[f'W2{k}'] + params[f'b2{k}']
    xf, _ = layer_norm_fwd(X[-1], params['lnf_g'], params['lnf_b'])
    return xf @ params['wte'].T                                       # logits del último


def generar_sin_cache(ids, n_tokens):
    """Greedy como generar() del Programa 8 bis: un forward completo por token."""
    ids = list(ids)
    for _ in range(n_tokens):
        ids.append(int(np.argmax(forward(np.array([ids[-CTX:]]))[0][0, -1])))
    return ids


def generar_con_cache(ids, n_tokens):
    """Greedy con KV-cache: prefill de la semilla y después un token cada vez."""
    if len(ids) + n_tokens > CTX:                 # el cache tiene CTX posiciones
        raise ValueError(f'semilla + tokens nuevos = {len(ids) + n_tokens} > {CTX}')
    ids, kv = list(ids), nuevo_cache()
    logits = prefill(ids, kv)
    for k in range(n_tokens):
        ids.append(int(np.argmax(logits)))
        if k < n_tokens - 1:                      # el último token no necesita otro paso
            logits = forward_un_token(ids[-1], len(ids) - 1, kv)
    return ids


# 1. ¿Da exactamente lo mismo que el forward completo?
ids = codificar('The quick brown fox jumps over the lazy dog because it')
kv, dif = nuevo_cache(), 0.0
completo = forward(np.array([ids]))[0][0]
for t, tok in enumerate(ids):
    dif = max(dif, np.abs(forward_un_token(tok, t, kv) - completo[t]).max())
dif = max(dif, np.abs(prefill(ids, nuevo_cache()) - completo[-1]).max())
print(f'Diferencia máxima entre los logits con y sin cache: {dif:.1e} '
      f'(los logits valen hasta {np.abs(completo).max():.0f})')

# 2. ¿Cuánto ahorra? Depende de la longitud del contexto.
texto = open('programa_8_nano_llm.py', encoding='utf-8').read().split('TEXTO = """')[1].split('"""')[0]
largo = codificar(texto * 3)                     # un texto largo del que cortar contextos
N_GEN = 10
print(f'\nTiempo por token generado (greedy, {N_GEN} tokens), según la longitud del contexto:')
print(' contexto   sin cache   con cache   ahorro   mismos tokens')
for L in (50, 200, 800):
    semilla = largo[:L]
    t0 = time.time(); a = generar_sin_cache(semilla, N_GEN); t_sin = (time.time() - t0) / N_GEN
    t0 = time.time(); b = generar_con_cache(semilla, N_GEN); t_con = (time.time() - t0) / N_GEN
    print(f'{L:7d}   {t_sin:7.2f} s   {t_con:7.2f} s   {t_sin / t_con:5.1f}x   '
          f'{"sí" if a == b else "NO"}')
print('(el tiempo «con cache» incluye el prefill de la semilla, repartido entre los tokens generados)')
