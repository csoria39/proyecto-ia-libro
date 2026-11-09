# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# ================================================================
# FUNCIONES DE MUESTREO
# ================================================================
 
def softmax(x, tau=1.0):
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum()
 
def top_k_filtrar(probs, k):
    idx  = np.argsort(probs)[::-1]
    mask = np.zeros_like(probs)
    mask[idx[:k]] = probs[idx[:k]]
    return mask / mask.sum()
 
def top_p_filtrar(probs, p):
    idx       = np.argsort(probs)[::-1]
    acum      = np.cumsum(probs[idx])
    n         = int(np.searchsorted(acum, p)) + 1
    mask      = np.zeros_like(probs)
    mask[idx[:n]] = probs[idx[:n]]
    return mask / mask.sum()
 
def muestrear(probs, n=10000):
    """Muestrea n tokens y devuelve las frecuencias observadas."""
    muestras = np.random.choice(len(probs), size=n, p=probs)
    freqs    = np.bincount(muestras, minlength=len(probs)) / n
    return freqs
 
def visualizar_dist(probs, nombre, tokens):
    """Muestra la distribucion como diagrama de barras ASCII."""
    print(f'  {nombre}:')
    activos = sum(p > 0.001 for p in probs)
    print(f'  Tokens activos (prob > 0.1%): {activos}/{len(probs)}')
    for i, (tok, p) in enumerate(zip(tokens, probs)):
        if p < 0.001: continue
        barra  = int(p * 50)
        bloque = '#' * barra
        print(f'    {tok:>12}: {p:>6.3f}  {bloque}')
    print()
 
 
np.random.seed(42)
 
# ================================================================
# DISTRIBUCION DE EJEMPLO: vocabulario de 12 tokens
# Simula la salida de un LLM en un contexto concreto
# ================================================================
tokens = ['el', 'la', 'un', 'los', 'de',
          'que', 'en', 'y', 'a', 'por', 'extraño', 'xyz']
 
# Logits tipicos: unos pocos tokens muy probables, muchos improbables
logits = np.array([3.8, 3.2, 2.1, 1.8, 1.5,
                   1.0, 0.5, 0.2, -0.3, -0.8, -2.0, -4.0])
 
print('='*60)
print('VISUALIZACION DE ESTRATEGIAS DE MUESTREO')
print('='*60)
print()
 
# --- DISTRIBUCION BASE (temperatura 1.0) ---
probs_base = softmax(logits, tau=1.0)
visualizar_dist(probs_base, 'BASE  (tau=1.0, sin filtros)', tokens)
 
# --- TEMPERATURA BAJA ---
probs_fria = softmax(logits, tau=0.5)
visualizar_dist(probs_fria, 'TEMPERATURA BAJA  (tau=0.5)', tokens)
 
# --- TOP-K ---
probs_k3 = top_k_filtrar(probs_base, k=3)
visualizar_dist(probs_k3, 'TOP-K=3  (solo los 3 mejores)', tokens)
 
probs_k6 = top_k_filtrar(probs_base, k=6)
visualizar_dist(probs_k6, 'TOP-K=6  (los 6 mejores)', tokens)
 
# --- TOP-P ---
probs_p80 = top_p_filtrar(probs_base, p=0.80)
n_tokens_80 = sum(probs_p80 > 0)
visualizar_dist(probs_p80, f'TOP-P=0.80  ({n_tokens_80} tokens cubren 80%)', tokens)
 
probs_p95 = top_p_filtrar(probs_base, p=0.95)
n_tokens_95 = sum(probs_p95 > 0)
visualizar_dist(probs_p95, f'TOP-P=0.95  ({n_tokens_95} tokens cubren 95%)', tokens)
 
# --- COMPARATIVA: tokens incluidos por cada estrategia ---
print('COMPARATIVA: Tokens incluidos por cada estrategia')
print(f"{'Token':>12}  {'Base':>8}  {'tau=0.5':>8}  ",
      f"{'Top-k=3':>8}  {'Top-k=6':>8}  {'Top-p.80':>9}  {'Top-p.95':>9}")
print('-' * 75)
for i, tok in enumerate(tokens):
    p0 = probs_base[i]
    p1 = probs_fria[i]
    p2 = probs_k3[i]
    p3 = probs_k6[i]
    p4 = probs_p80[i]
    p5 = probs_p95[i]
    def fmt(p): return f'{p:.3f}' if p > 0.001 else '  ---'
    print(f'{tok:>12}  {fmt(p0):>8}  {fmt(p1):>8}  ',
          f'{fmt(p2):>8}  {fmt(p3):>8}  {fmt(p4):>9}  {fmt(p5):>9}')
 
print()
print('Observacion clave:')
print('  Top-k=3: exactamente 3 tokens siempre, sin importar la distribucion.')
print('  Top-p=0.80: el numero de tokens se adapta a la forma de la dist.')
print('  Con dist. concentrada: Top-p usara MENOS tokens que Top-k=6.')
print('  Con dist. plana: Top-p usara MAS tokens que Top-k=6.')
