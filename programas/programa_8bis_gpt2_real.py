# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA 8 bis — GPT-2 de verdad, con el forward del Programa 8
# Carga los pesos REALES de GPT-2 small (124 millones de
# parámetros, OpenAI 2019) y los pasa por EXACTAMENTE la misma
# función forward() del Programa 8. Solo cambian los tamaños y
# el tokenizador (BPE en lugar de caracteres).
#
# Requisitos: NumPy y una carpeta gpt2/ junto a este archivo con
# los cuatro archivos de huggingface.co/openai-community/gpt2:
#   model.safetensors  (pesos, ~548 MB)
#   vocab.json, merges.txt  (tokenizador BPE)
#   config.json  (tamaños del modelo)
# No necesita PyTorch ni la librería transformers.
# =======================================================
import json
import re
import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')   # para que Windows muestre «Ġ», «␣», ñ...

CARPETA = 'gpt2'

# -- 1. Leer los pesos: el formato safetensors, a mano ------------
# Un archivo .safetensors es muy simple:
#   8 bytes  -> N, la longitud de la cabecera (entero sin signo)
#   N bytes  -> cabecera en JSON: para cada tensor, su tipo, su
#               forma y en qué bytes del resto del archivo está
#   resto    -> los números de todos los tensores, uno detrás de otro
def leer_safetensors(ruta):
    with open(ruta, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        cabecera = json.loads(f.read(n))
        datos = f.read()
    tipos = {'F32': np.float32, 'F16': np.float16}
    tensores = {}
    for nombre, info in cabecera.items():
        if nombre == '__metadata__':
            continue
        ini, fin = info['data_offsets']
        t = np.frombuffer(datos[ini:fin], dtype=tipos[info['dtype']])
        tensores[nombre.removeprefix('transformer.')] = t.reshape(info['shape'])
    return tensores


t0 = time.time()
cfg = json.load(open(f'{CARPETA}/config.json'))
gpt2 = leer_safetensors(f'{CARPETA}/model.safetensors')
print(f'Pesos leídos en {time.time() - t0:.1f} s: {len(gpt2)} tensores')

# -- 2. Los tamaños de GPT-2 small (en vez de los del Programa 8) --
N_BLQ = cfg['n_layer']          # 12 bloques     (Programa 8: 2)
H     = cfg['n_head']           # 12 cabezas     (Programa 8: 4)
D     = cfg['n_embd']           # 768            (Programa 8: 48)
CTX   = cfg['n_positions']      # 1024 tokens    (Programa 8: 24)
DH    = D // H                  # 64 por cabeza  (Programa 8: 12)
EPS   = cfg.get('layer_norm_epsilon', 1e-5)

# -- 3. Los pesos con los nombres del Programa 8 --------------------
# GPT-2 guarda Q, K y V juntas en una matriz c_attn de 768 × 2304;
# la cortamos en tres de 768 × 768. Sus matrices ya están en la
# orientación del Programa 8 (x @ W), así que no hay que trasponer.
params = {'wte': gpt2['wte.weight'], 'wpe': gpt2['wpe.weight'],
          'lnf_g': gpt2['ln_f.weight'], 'lnf_b': gpt2['ln_f.bias']}
for k in range(N_BLQ):
    b = f'h.{k}.'
    params[f'ln1_g{k}'], params[f'ln1_b{k}'] = gpt2[b + 'ln_1.weight'], gpt2[b + 'ln_1.bias']
    W, bias = gpt2[b + 'attn.c_attn.weight'], gpt2[b + 'attn.c_attn.bias']
    for i, n in enumerate('qkv'):
        params[f'W{n}{k}'] = W[:, i * D:(i + 1) * D]
        params[f'b{n}{k}'] = bias[i * D:(i + 1) * D]
    params[f'Wo{k}'], params[f'bo{k}'] = gpt2[b + 'attn.c_proj.weight'], gpt2[b + 'attn.c_proj.bias']
    params[f'ln2_g{k}'], params[f'ln2_b{k}'] = gpt2[b + 'ln_2.weight'], gpt2[b + 'ln_2.bias']
    params[f'W1{k}'], params[f'b1{k}'] = gpt2[b + 'mlp.c_fc.weight'], gpt2[b + 'mlp.c_fc.bias']
    params[f'W2{k}'], params[f'b2{k}'] = gpt2[b + 'mlp.c_proj.weight'], gpt2[b + 'mlp.c_proj.bias']
V = params['wte'].shape[0]
print(f'Vocabulario: {V} tokens | Bloques: {N_BLQ} | Cabezas: {H} | D: {D} | Contexto: {CTX}')
print(f'Total de parámetros: {sum(p.size for p in params.values()):,}')
MASCARA_CAUSAL = np.triu(np.full((CTX, CTX), -1e9, dtype=np.float32), k=1)

# -- 4. Las funciones del Programa 8, sin cambiar una línea ---------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)
def layer_norm_fwd(x, g, b, eps=EPS):
    mu = x.mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv)
C_GELU = np.sqrt(2 / np.pi)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(C_GELU * (x + 0.044715 * x * x * x)))
def separar(x):   # (B, T, D) -> (B, H, T, DH)
    B, T, _ = x.shape
    return x.reshape(B, T, H, DH).transpose(0, 2, 1, 3)
def unir(x):      # (B, H, T, DH) -> (B, T, D)
    B, _, T, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, D)

def forward(tokens):
    """tokens (B, T) -> logits (B, T, V). El mismo código del Programa 8."""
    B, T = tokens.shape
    X = params['wte'][tokens] + params['wpe'][:T]
    cache = []
    for k in range(N_BLQ):
        Xin = X
        a, ln1 = layer_norm_fwd(Xin, params[f'ln1_g{k}'], params[f'ln1_b{k}'])
        Q, K, Vm = (separar(a @ params[f'W{n}{k}'] + params[f'b{n}{k}']) for n in 'qkv')
        pesos = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(DH) + MASCARA_CAUSAL[:T, :T])
        att = unir(pesos @ Vm)
        X1 = Xin + att @ params[f'Wo{k}'] + params[f'bo{k}']
        m, ln2 = layer_norm_fwd(X1, params[f'ln2_g{k}'], params[f'ln2_b{k}'])
        h_pre = m @ params[f'W1{k}'] + params[f'b1{k}']
        h = gelu(h_pre)
        X = X1 + h @ params[f'W2{k}'] + params[f'b2{k}']
        cache.append(dict(X=X, pesos=pesos))   # salida y atención de cada bloque (16.5, 16.6)
    xf, lnf = layer_norm_fwd(X, params['lnf_g'], params['lnf_b'])
    logits = xf @ params['wte'].T          # pesos compartidos, como en el Programa 8
    return logits, (xf, lnf, cache)

def top_p_filtrar(probs, p):
    indices_ord = np.argsort(probs)[::-1]
    suma_acum = np.cumsum(probs[indices_ord])
    n_tokens = int(np.searchsorted(suma_acum, p)) + 1
    nueva = np.zeros_like(probs)
    nueva[indices_ord[:n_tokens]] = probs[indices_ord[:n_tokens]]
    return nueva / nueva.sum()

# -- 5. El tokenizador BPE de GPT-2 (Capítulo 9) ----------------
# GPT-2 trabaja con BYTES: cada uno de los 256 valores posibles de
# un byte se representa con un carácter imprimible (el espacio, por
# ejemplo, se convierte en «Ġ»). Sobre esos caracteres se aplican,
# por orden, las 50.000 fusiones aprendidas que hay en merges.txt.
def bytes_a_unicode():
    bs = (list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1))
          + list(range(ord('®'), ord('ÿ') + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

B2U = bytes_a_unicode()
U2B = {u: b for b, u in B2U.items()}
VOCAB = json.load(open(f'{CARPETA}/vocab.json', encoding='utf-8'))
ID2TOK = {i: t for t, i in VOCAB.items()}
lineas = open(f'{CARPETA}/merges.txt', encoding='utf-8').read().split('\n')
RANGO = {tuple(l.split()): i for i, l in enumerate(l for l in lineas[1:] if l.strip())}
# Cómo trocea GPT-2 el texto antes de aplicar BPE (palabras con su
# espacio delante, números, signos, espacios). Versión con el módulo
# re estándar de Python: [^\W\d_] son las letras.
TROCEO = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")

def bpe(palabra):
    """Aplica las fusiones de merges.txt, siempre la de menor rango primero."""
    partes = list(palabra)
    while len(partes) > 1:
        pares = [(RANGO.get((a, b), 1e18), i) for i, (a, b) in enumerate(zip(partes, partes[1:]))]
        rango, i = min(pares)
        if rango == 1e18:
            break
        partes = partes[:i] + [partes[i] + partes[i + 1]] + partes[i + 2:]
    return partes

def codificar(texto):
    ids = []
    for trozo in TROCEO.findall(texto):
        palabra = ''.join(B2U[b] for b in trozo.encode('utf-8'))
        ids += [VOCAB[p] for p in bpe(palabra)]
    return ids

def decodificar(ids):
    texto = ''.join(ID2TOK[i] for i in ids)
    return bytes(U2B[c] for c in texto).decode('utf-8', errors='replace')

def generar(semilla, n_tokens=40, temperatura=0.8, top_p=0.9):
    """Como generar() del Programa 8, pero con tokens BPE.
    temperatura=0 -> greedy (siempre el más probable)."""
    ids = codificar(semilla)
    for _ in range(n_tokens):
        logits = forward(np.array([ids[-CTX:]]))[0][0, -1].astype(np.float64)
        if temperatura == 0:
            siguiente = int(np.argmax(logits))
        else:
            probs = softmax(logits / temperatura)
            if top_p < 1.0:
                probs = top_p_filtrar(probs, top_p)
            siguiente = int(np.random.choice(V, p=probs))
        ids.append(siguiente)
    return decodificar(ids)

# =======================================================
# EXPERIMENTOS
# =======================================================
print('\n=== 1. EL TOKENIZADOR BPE (Ġ = espacio delante de la palabra) ===')
for frase in ['Hello world', 'The capital of France is',
              'la inteligencia artificial aprende de los datos.']:
    ids = codificar(frase)
    print(f'{frase!r}: {len(frase)} caracteres -> {len(ids)} tokens')
    print('   ', [ID2TOK[i] for i in ids])
    print('   ', ids)

print('\n=== 2. ¿QUÉ TOKEN VIENE DESPUÉS? (las 5 opciones más probables) ===')
for frase in ['The capital of France is', 'Paris is the capital of',
              'The cat sat on the', 'One, two, three, four,']:
    t0 = time.time()
    logits = forward(np.array([codificar(frase)]))[0][0, -1].astype(np.float64)
    probs = softmax(logits)
    print(f'{frase!r}   ({time.time() - t0:.2f} s)')
    for i in np.argsort(probs)[::-1][:5]:
        print(f'    {decodificar([i]).replace(" ", "␣").replace(chr(10), "↵"):>12s}  {probs[i]:.3f}')

print('\n=== 3. PÉRDIDA Y PERPLEJIDAD EN UN TEXTO QUE NO ESTÁ EN SU CORPUS ===')
texto = ('Neural networks learn by adjusting their weights. During training, the network makes a '
         'prediction, measures its error with a loss function, and uses backpropagation to compute '
         'how each weight should change to reduce that error.')
ids = codificar(texto)
logits = forward(np.array([ids]))[0][0].astype(np.float64)
probs = softmax(logits[:-1])
perdida = -np.mean(np.log(probs[np.arange(len(ids) - 1), ids[1:]]))
print(f'{len(ids)} tokens | pérdida media = {perdida:.3f} | perplejidad = {np.exp(perdida):.1f}')
print(f'Referencia «al azar»: ln({V}) = {np.log(V):.2f}, perplejidad {V}')

print('\n=== 4. GENERACIÓN GREEDY (comprobación) ===')
t0 = time.time()
print(generar('I enjoy walking with my cute dog', n_tokens=30, temperatura=0))
print(f'({time.time() - t0:.1f} s)')

print('\n=== 5. GENERACIÓN CON TEMPERATURA 0,8 Y TOP-P 0,9 ===')
np.random.seed(42)
for semilla in ['The meaning of life is', 'In a shocking finding, scientists discovered',
                'La inteligencia artificial es']:
    print(f'\n--- {semilla!r} ---')
    print(generar(semilla, n_tokens=40))
