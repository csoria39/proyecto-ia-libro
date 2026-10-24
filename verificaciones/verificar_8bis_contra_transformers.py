# Verificación (repositorio, no libro): el Programa 8 bis y el backward del
# Programa 8 frente a la implementación oficial de Hugging Face (transformers).
#   1. Tokenizador: mismos tokens en 50 textos.
#   2. Forward: mismos logits en 5 frases.
#   3. Backward: mismos gradientes en un mini-lote (forward_backward del Programa 8).
# Se ejecuta en la carpeta que contiene programa_8bis_gpt2_real.py, programa_8_nano_llm.py y gpt2/.
import numpy as np, torch, json
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])
p8 = open('programa_8_nano_llm.py', encoding='utf-8').read()
exec('# -- Utilidades' + p8.split('# -- Utilidades')[1].split('# -- Learning rate')[0])
params = {k: np.array(v, dtype=np.float32) for k, v in params.items()}

# Modelo oficial con los mismos pesos (nombres originales del archivo)
modelo = GPT2LMHeadModel(GPT2Config(**{k: cfg[k] for k in ('n_layer', 'n_head', 'n_embd', 'n_positions', 'vocab_size') if k in cfg},
                                    activation_function='gelu_new', resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
sd = {'transformer.' + k: torch.tensor(np.array(v)) for k, v in gpt2.items() if not k.endswith(('.attn.bias', '.attn.masked_bias'))}
falta = modelo.load_state_dict(sd, strict=False)
print('Claves no cargadas:', [k for k in falta.missing_keys if not k.endswith(('attn.bias', 'masked_bias', 'lm_head.weight'))], falta.unexpected_keys)
modelo.tie_weights(); modelo.eval()

# 1. Tokenizador
tok = GPT2TokenizerFast.from_pretrained(f'./{CARPETA}')
textos = ['Hello world', 'The capital of France is', 'la inteligencia artificial aprende de los datos.',
          "It's 3.14159 -- isn't it? He'll go; they've   gone!\n\nNew   para", '¡Hola, señor! ¿Qué tal? Ñandú, pingüino.',
          'def f(x):\n    return x ** 2  # comentario', 'café_au_lait  x__y  中文 emoji 😀', "CAPS'S don't I'M",
          '  leading spaces\ttabs\r\nwin', 'GPT-2 was released in 2019 by OpenAI.'] 
textos += [' '.join(np.random.RandomState(i).choice(['the', 'modelo', '42', '!', 'é', 'naïve', '\n', 'x_y', "'s", '  '], 8)) for i in range(40)]
malos = [t for t in textos if codificar(t) != tok.encode(t)]
print(f'1. Tokenizador: {len(textos) - len(malos)} de {len(textos)} textos idénticos', malos[:3])

# 2. Forward
dif = 0
for f in ['The capital of France is', 'I enjoy walking with my cute dog', 'One, two, three, four,',
          'la inteligencia artificial aprende de los datos.', 'Neural networks learn by adjusting their weights.']:
    ids = codificar(f)
    with torch.no_grad():
        ref = modelo(torch.tensor([ids])).logits[0].numpy()
    dif = max(dif, np.abs(forward(np.array([ids]))[0][0] - ref).max())
print(f'2. Forward: diferencia máxima de logits = {dif:.1e}')

# 3. Backward
datos = np.array(codificar('The quick brown fox jumps over the lazy dog. ' * 6))
xb = np.stack([datos[0:32], datos[5:37]]); yb = np.stack([datos[1:33], datos[6:38]])
grads, loss = forward_backward(xb, yb)
modelo.zero_grad()
out = modelo(torch.tensor(xb), labels=None).logits
lt = torch.nn.functional.cross_entropy(out.reshape(-1, out.shape[-1]), torch.tensor(yb).reshape(-1))
lt.backward()
P = dict(modelo.named_parameters())
pares = [('wte', P['transformer.wte.weight'].grad), ('wpe', P['transformer.wpe.weight'].grad),
         ('Wq0', P['transformer.h.0.attn.c_attn.weight'].grad[:, :D]), ('Wv5', P['transformer.h.5.attn.c_attn.weight'].grad[:, 2 * D:] if N_BLQ > 5 else None),
         ('W1' + str(N_BLQ - 1), P[f'transformer.h.{N_BLQ - 1}.mlp.c_fc.weight'].grad), ('ln1_g0', P['transformer.h.0.ln_1.weight'].grad),
         ('lnf_b', P['transformer.ln_f.bias'].grad)]
print(f'3. Backward: pérdida {loss:.5f} (nuestra) frente a {lt.item():.5f} (transformers)')
for n, g in pares:
    if g is None: continue
    g = g.numpy(); e = np.linalg.norm(grads[n] - g) / (np.linalg.norm(g) + 1e-30)
    print(f'   {n:6s} error relativo {e:.1e}')
