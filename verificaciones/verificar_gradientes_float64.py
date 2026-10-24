import numpy as np, torch, gc
from transformers import GPT2Config, GPT2LMHeadModel
fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])
p8 = open('programa_8_nano_llm.py', encoding='utf-8').read()
exec('# -- Utilidades' + p8.split('# -- Utilidades')[1].split('# -- Learning rate')[0])
params = {k: np.array(v, dtype=np.float64) for k, v in params.items()}
MASCARA_CAUSAL = MASCARA_CAUSAL.astype(np.float64)
datos = np.array(codificar('The quick brown fox jumps over the lazy dog. ' * 6))
xb = np.stack([datos[0:32], datos[5:37]]); yb = np.stack([datos[1:33], datos[6:38]])
grads, loss = forward_backward(xb, yb)
nuestros = forward(xb)[0]                      # logits en float64
sel = {n: grads[n].copy() for n in ['wte', 'wpe', 'Wq0', 'Wv5', 'W111', 'ln1_g0', 'lnf_b']}
del grads, params; gc.collect()
modelo = GPT2LMHeadModel(GPT2Config(n_layer=12, n_head=12, n_embd=768, n_positions=1024, vocab_size=50257,
                                    activation_function='gelu_new', resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
modelo.load_state_dict({'transformer.' + k: torch.tensor(np.array(v, dtype=np.float64)) for k, v in gpt2.items() if not k.endswith('.attn.bias')}, strict=False)
del gpt2; gc.collect()
modelo.tie_weights(); modelo.double()
out = modelo(torch.tensor(xb)).logits
lt = torch.nn.functional.cross_entropy(out.reshape(-1, out.shape[-1]), torch.tensor(yb).reshape(-1)); lt.backward()
P = dict(modelo.named_parameters()); D = 768
ref = {'wte': P['transformer.wte.weight'].grad, 'wpe': P['transformer.wpe.weight'].grad, 'Wq0': P['transformer.h.0.attn.c_attn.weight'].grad[:, :D],
       'Wv5': P['transformer.h.5.attn.c_attn.weight'].grad[:, 2 * D:], 'W111': P['transformer.h.11.mlp.c_fc.weight'].grad,
       'ln1_g0': P['transformer.h.0.ln_1.weight'].grad, 'lnf_b': P['transformer.ln_f.bias'].grad}
print(f'float64: diferencia máxima de logits = {np.abs(nuestros - out.detach().numpy()).max():.1e}')
print(f'float64: pérdida {loss:.12f} frente a {lt.item():.12f}')
for n in sel:
    g = ref[n].numpy(); print(f'   {n:6s} error relativo {np.linalg.norm(sel[n] - g) / np.linalg.norm(g):.1e}')
