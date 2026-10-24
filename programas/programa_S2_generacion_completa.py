# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
exec(fuente.split('# -- Learning rate')[0])     # tokenizador y forward() del Programa 8
params = dict(np.load('nano_llm_pesos.npz'))   # los pesos que guardó el Programa 8
 
# ================================================================
# GENERADOR UNIFICADO CON TODAS LAS ESTRATEGIAS
# Este codigo amplia la funcion generar() del Nano-LLM
# con soporte para todas las estrategias de muestreo.
# ================================================================
 
def softmax(x, axis=-1): e=np.exp(x-x.max(axis, keepdims=True)); return e/e.sum(axis, keepdims=True)
def top_k_filtrar(p, k): idx=np.argsort(p)[::-1]; m=np.zeros_like(p); m[idx[:k]]=p[idx[:k]]; return m/m.sum()
def top_p_filtrar(p, pp): idx=np.argsort(p)[::-1]; ac=np.cumsum(p[idx]); n=int(np.searchsorted(ac,pp))+1; m=np.zeros_like(p); m[idx[:n]]=p[idx[:n]]; return m/m.sum()
 
def generar_avanzado(
    modelo_forward,  # funcion forward del Nano-LLM
    c2i, i2c,        # diccionarios del tokenizador
    semilla,         # texto inicial
    n       = 80,    # tokens a generar
    CTX     = 24,    # longitud de contexto
    # Hiperparametros de muestreo:
    tau     = 1.0,   # temperatura
    k       = 0,     # top-k (0 = desactivado)
    p       = 0.0,   # top-p (0 = desactivado)
    rep_pen = 1.0,   # repetition penalty (1.0 = sin penalizacion)
):
    """
    Generacion flexible con todas las estrategias combinables.
    El orden de aplicacion es: temperatura -> top-k -> top-p -> muestreo
    """
    V       = len(c2i)
    tokens  = [c2i.get(c, 0) for c in semilla[-CTX:]]
    resultado = semilla
    generados = list(tokens)   # para repetition penalty
    
    for _ in range(n):
        ctx    = tokens[-CTX:]   # SIN relleno: longitud real (ver 9.3)
        
        # 1. Obtener logits del modelo
        logits = modelo_forward(ctx)[-1]
        
        # 2. Aplicar repetition penalty
        if rep_pen != 1.0:
            for t in set(generados[-20:]):   # solo los ultimos 20
                if logits[t] > 0: logits[t] /= rep_pen
                else:             logits[t] *= rep_pen
        
        # 3. Aplicar temperatura
        probs = softmax(logits / max(tau, 1e-5))
        
        # 4. Aplicar top-k (si esta activado)
        if k > 0:
            probs = top_k_filtrar(probs, k)
        
        # 5. Aplicar top-p (si esta activado)
        if 0 < p < 1.0:
            probs = top_p_filtrar(probs, p)
        
        # 6. Muestrear
        siguiente = int(np.random.choice(V, p=probs))
        tokens.append(siguiente)
        generados.append(siguiente)
        resultado += i2c[siguiente]
    
    return resultado
 
 
# ================================================================
# COMPARATIVA: misma semilla, distintas estrategias
# (Nano-LLM del Cap 13 ya entrenado: params = dict(np.load('nano_llm_pesos.npz')))
# ================================================================
 
estrategias = [
    {'nombre': 'Greedy (tau=0.01)',
     'tau': 0.01, 'k': 0, 'p': 0.0, 'rep_pen': 1.0},
 
    {'nombre': 'Solo temperatura (tau=0.8)',
     'tau': 0.8,  'k': 0, 'p': 0.0, 'rep_pen': 1.0},
 
    {'nombre': 'Top-k=5, tau=0.8',
     'tau': 0.8,  'k': 5, 'p': 0.0, 'rep_pen': 1.0},
 
    {'nombre': 'Top-p=0.90, tau=0.8',
     'tau': 0.8,  'k': 0, 'p': 0.9, 'rep_pen': 1.0},
 
    {'nombre': 'Top-p=0.90 + rep_pen=1.2',
     'tau': 0.8,  'k': 0, 'p': 0.9, 'rep_pen': 1.2},
]
 
SEMILLA = 'la inteligencia'
print(f'=== COMPARATIVA DE ESTRATEGIAS ===')
print(f'Semilla: "{SEMILLA}"  |  70 tokens a generar')
print()
 
# El forward del Programa 8 trabaja con lotes: le pasamos un lote de una
# sola secuencia y nos quedamos con sus logits, de forma (T, V)
def logits_nano(ctx):
    return forward(np.array([ctx]))[0][0]
 
np.random.seed(7)
for conf in estrategias:
    np.random.seed(7)   # misma semilla aleatoria para comparar
    nombre = conf.pop('nombre')
    texto = generar_avanzado(logits_nano, C2I, I2C, SEMILLA, n=70, CTX=CTX, **conf)
    # (C2I, I2C y forward son los del Nano-LLM ya entrenado, Programa 8)
    print(f'  [{nombre}]')
    print(f'  -> {texto}')
    print()
 
# ================================================================
# ANALISIS: efecto de tau y top-p sobre la diversidad
# ================================================================
print('=== EFECTO EN LA DIVERSIDAD DEL TEXTO ===')
print()
print('Tau muy bajo (< 0.3):')
print('  El modelo siempre elige el token mas probable.')
print('  Texto coherente pero repetitivo. Mismo texto siempre.')
print()
print('Tau 0.7-0.9 + Top-p=0.90 (configuracion estandar):')
print('  Balance entre coherencia y variedad.')
print('  Cada ejecucion produce texto distinto pero sensato.')
print('  Es la configuracion por defecto de la mayoria de LLMs.')
print()
print('Tau muy alto (> 1.5):')
print('  El texto se vuelve incoherente. Palabras aleatorias.')
print('  Util para explorar espacios creativos extremos.')
print()
print('Top-p=0.90 + rep_pen=1.2:')
print('  Evita la repeticion. Necesario en textos largos.')
print('  Riesgo: penalizacion muy alta puede evitar palabras necesarias.')
