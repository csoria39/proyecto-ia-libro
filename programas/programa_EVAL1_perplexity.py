# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# ================================================================
# CALCULO CORRECTO DE PERPLEXITY
# El modelo NUNCA debe ver los datos de test durante el entreno.
# ================================================================
 
def softmax(x): e=np.exp(x-x.max()); return e/e.sum()
 
def calcular_perplexity(modelo_forward, tokens, CTX, nombre=''):
    """
    Calcula la perplejidad de un modelo sobre una secuencia.
    modelo_forward: funcion que recibe CTX tokens y devuelve logits
    tokens:         lista de indices enteros del texto
    CTX:            longitud del contexto del modelo
    """
    perdidas = []
    n_tokens = 0
    
    for i in range(CTX, len(tokens)):
        ctx    = tokens[i-CTX:i]      # contexto
        target = tokens[i]             # token a predecir
        logits = modelo_forward(ctx)   # logits del modelo
        # Tomar los logits del ULTIMO paso (prediccion del siguiente)
        if logits.ndim == 2: logits = logits[-1]
        probs  = softmax(logits)
        # -log P(token correcto | contexto)
        loss   = -np.log(probs[target] + 1e-10)
        perdidas.append(float(loss))
        n_tokens += 1
    
    perdida_media = np.mean(perdidas)
    ppl           = np.exp(perdida_media)
    bits_por_tok  = perdida_media / np.log(2)   # nats -> bits (aqui cada token es un caracter)
    
    print(f'  {nombre or "Modelo"}')
    print(f'    Tokens evaluados:  {n_tokens}')
    print(f'    Perdida media:     {perdida_media:.4f}')
    print(f'    Perplejidad:       {ppl:.2f}')
    print(f'    Bits por token:    {bits_por_tok:.3f}')
    return ppl, perdida_media
 
 
# Simulacion: entrenamiento correcto con split train/test
# (El Programa RZ-1 del Cap 15 lo hace con el modelo real del Cap 13)
TEXTO_COMPLETO = '''
la inteligencia artificial aprende de los datos de entrenamiento.
los modelos de lenguaje predicen el siguiente token del texto.
las redes neuronales tienen capas de neuronas interconectadas.
el aprendizaje profundo usa muchas capas ocultas y atencion.
los transformers procesan secuencias en paralelo con atencion.
entrenar un modelo requiere datos etiquetados y potencia de calculo.
la generacion de texto usa estrategias de muestreo como top-p.
evaluar un modelo requiere datos de test que no vio en entrenamiento.
'''
 
vocab  = sorted(set(TEXTO_COMPLETO))
V      = len(vocab)
c2i    = {c:i for i,c in enumerate(vocab)}
i2c    = dict(enumerate(vocab))
tokens = [c2i[c] for c in TEXTO_COMPLETO]
 
# Division: 80% train, 20% test
corte      = int(0.8 * len(tokens))
tok_train  = tokens[:corte]
tok_test   = tokens[corte:]
 
print(f'Corpus total: {len(tokens)} tokens (vocab={V})')
print(f'Entrenamiento: {len(tok_train)} tokens ({len(tok_train)/len(tokens):.0%})')
print(f'Test:          {len(tok_test)}  tokens ({len(tok_test)/len(tokens):.0%})')
print(f'Perplejidad aleatoria esperada: {V:.0f} (= tamano vocabulario)')
print()
 
# Modelo ficticio para demo: distribuciones variadas
np.random.seed(42)
 
def modelo_aleatorio(ctx):
    """Distribucion uniforme: la referencia del azar (PPL = V)"""
    return np.zeros(V)    # logits uniformes -> prob 1/V cada token
 
def modelo_bigrama(ctx, tokens_ref=tok_train):
    """Bigrama: solo recuerda el ultimo token"""
    ultimo = ctx[-1]
    logits = np.zeros(V)
    # Buscar cuales tokens siguen a 'ultimo' en los datos de entreno
    for i in range(len(tokens_ref)-1):
        if tokens_ref[i] == ultimo:
            logits[tokens_ref[i+1]] += 1
    if logits.sum() == 0: logits[:] = 1  # fallback uniforme
    return np.log(logits / logits.sum() + 1e-10)  # log-probs
 
def modelo_uniforme_top3(ctx):
    """Simula un modelo que conoce los 3 tokens mas probables"""
    # El siguiente token es el ultimo caracter del contexto + algo
    # (simulacion simplificada para la demo)
    logits = np.random.randn(V) * 0.1
    # Dar algo de peso al token correcto con probabilidad variable
    logits[np.random.randint(V)] += 2.0
    return logits
 
CTX = 8
print('=== PERPLEJIDAD EN DISTINTOS CONJUNTOS ===')
print()
 
print('Modelo aleatorio (baseline):')
calcular_perplexity(modelo_aleatorio, tok_test, CTX, 'Test')
print()
 
print('Modelo bigrama (aprende pares de caracteres):')
calcular_perplexity(modelo_bigrama, tok_train, CTX, 'Train')
calcular_perplexity(modelo_bigrama, tok_test, CTX,  'Test ')
print()
 
# Demostrar el diagnostico de overfitting
print('=== DIAGNOSTICO DE OVERFITTING VIA PERPLEJIDAD ===')
print()
print('CASO 1 — Buen modelo (generaliza):')
print('  PPL train = 4.2,  PPL test = 4.8  -> diferencia pequeña: OK')
print()
print('CASO 2 — Overfitting leve:')
print('  PPL train = 2.1,  PPL test = 8.3  -> test 4x peor: señal de alerta')
print()
print('CASO 3 — Overfitting grave (memorizacion):')
print('  PPL train = 1.0,  PPL test = 45.2 -> test 45x peor: el modelo')
print('  memorizo el corpus de entrenamiento pero no generaliza.')
print()
print('Una pista, no una regla: un cociente PPL_test / PPL_train alto avisa de')
print('sobreajuste, pero no hay umbrales universales (un modelo que no aprende')
print('nada tiene cociente 1). Compara siempre con una linea base.')
