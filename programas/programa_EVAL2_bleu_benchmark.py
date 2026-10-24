# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
import math
from collections import Counter
 
# ================================================================
# PARTE 1: BLEU DESDE CERO
# ================================================================
 
def ngramas(tokens, n):
    """Extrae todos los n-gramas de una lista de tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
 
def bleu(candidato, referencia, n_max=4):
    """
    Calcula el BLEU score entre un texto candidato y una referencia.
    candidato, referencia: listas de palabras (tokens)
    n_max:                 maximo orden de n-gramas (tipicamente 4)
    """
    cand = candidato.lower().split()
    ref  = referencia.lower().split()
    
    precisiones = []
    for n in range(1, n_max + 1):
        cand_ng = Counter(ngramas(cand, n))
        ref_ng  = Counter(ngramas(ref,  n))
        
        # Coincidencias con clipping (no contar mas que en la referencia)
        coincidencias = sum(
            min(cnt, ref_ng[ng]) for ng, cnt in cand_ng.items()
        )
        total = max(len(cand) - n + 1, 0)
        
        if total == 0:
            precisiones.append(0.0)
        else:
            precisiones.append(coincidencias / total)
    
    # Media geometrica de las precisiones
    if any(p == 0 for p in precisiones):
        bleu_score = 0.0
    else:
        log_avg    = sum(math.log(p) for p in precisiones) / n_max
        bleu_score = math.exp(log_avg)
    
    # Penalizacion por brevedad
    bp = math.exp(min(0, 1 - len(ref)/max(len(cand), 1)))
    
    return bp * bleu_score, precisiones, bp
 
 
# --- Ejemplos de calculo BLEU ---
print('=== BLEU SCORE: EJEMPLOS ===')
print()
 
casos = [
    ('Traduccion perfecta',
     'el modelo aprende de los datos',
     'el modelo aprende de los datos'),
    ('Traduccion buena',
     'el modelo aprende usando los datos',
     'el modelo aprende de los datos'),
    ('Traduccion parcial',
     'el sistema procesa informacion',
     'el modelo aprende de los datos'),
    ('Texto sin relacion',
     'la tarde cae sobre el monte',
     'el modelo aprende de los datos'),
]
 
for nombre, cand, ref in casos:
    score, precs, bp = bleu(cand, ref, n_max=2)
    print(f'  {nombre}:')
    print(f'    Candidato: "{cand}"')
    print(f'    Referencia: "{ref}"')
    print(f'    BLEU-2: {score:.4f}  (BLEU-1: {precs[0]:.3f}, BLEU-2: {precs[1]:.3f}, BP: {bp:.3f})')
    print()
 
# ================================================================
# PARTE 2: FRAMEWORK DE EVALUACION CON OPCION MULTIPLE
# ================================================================
 
print('=== FRAMEWORK DE EVALUACION: OPCION MULTIPLE ===')
print()
 
# Mini-benchmark sobre IA (preguntas de ejemplo)
BENCHMARK = [
    {
        'pregunta': 'Que funcion de activacion se usa mas en capas ocultas modernas?',
        'opciones': ['a) Sigmoid', 'b) Tanh', 'c) ReLU', 'd) Softmax'],
        'correcta': 'c',
        'explicacion': 'ReLU es la mas usada por eficiencia y por evitar el vanishing gradient'
    },
    {
        'pregunta': 'Que significa PPL = 37 si el vocabulario tiene 37 caracteres?',
        'opciones': ['a) El modelo es excelente', 'b) El modelo no ha aprendido nada',
                    'c) El modelo tiene overfitting', 'd) El modelo necesita mas capas'],
        'correcta': 'b',
        'explicacion': 'PPL = V significa distribucion uniforme, equivalente a adivinar al azar'
    },
    {
        'pregunta': 'Que optimizador usa weight decay desacoplado?',
        'opciones': ['a) SGD', 'b) RMSprop', 'c) Adam', 'd) AdamW'],
        'correcta': 'd',
        'explicacion': 'AdamW desacopla el weight decay del gradiente, corrigiendo un error de Adam'
    },
    {
        'pregunta': 'Cual es la ventaja principal de Top-p sobre Top-k en muestreo?',
        'opciones': ['a) Es mas rapido', 'b) Se adapta a la forma de la distribucion',
                    'c) Siempre incluye mas tokens', 'd) No necesita temperatura'],
        'correcta': 'b',
        'explicacion': 'Top-p adapta el numero de tokens al nivel de concentracion de la distribucion'
    },
    {
        'pregunta': 'Que es el Vanishing Gradient en RNNs?',
        'opciones': ['a) Los gradientes crecen sin control',
                    'b) Los gradientes se vuelven nulos al retropropagar muchos pasos',
                    'c) Los pesos convergen a cero', 'd) La perdida no baja'],
        'correcta': 'b',
        'explicacion': 'El gradiente se multiplica por valores < 1 en cada paso, decayendo exponencialmente'
    },
]
 
def evaluar_modelo_simulado(benchmark, tasa_acierto=0.80, semilla=42):
    """
    Simula la evaluacion de un modelo en un benchmark de opcion multiple.
    En la practica, se le daria la pregunta+opciones al modelo y se
    miraria cual opcion asigna mayor probabilidad.
    """
    np.random.seed(semilla)
    resultados = []
    for i, q in enumerate(benchmark):
        # Simular respuesta del modelo
        acierta = np.random.random() < tasa_acierto
        opciones = [o[0] for o in q['opciones']]
        if acierta:
            respuesta = q['correcta']
        else:
            otras = [o for o in opciones if o != q['correcta']]
            respuesta = np.random.choice(otras)
        resultados.append({
            'correcto': acierta,
            'respuesta': respuesta,
            'esperada': q['correcta']
        })
    return resultados
 
print('Mini-benchmark de IA (5 preguntas):')
print()
 
modelos = [
    ('Modelo aleatorio',  0.25),
    ('Modelo basico',     0.60),
    ('Modelo avanzado',   0.80),
    ('Modelo experto',    1.00),
]
 
print(f"{'Modelo':>20}  {'Aciertos':>10}  {'Precision':>12}  Interpretacion")
print('-' * 68)
for nombre, tasa in modelos:
    res = evaluar_modelo_simulado(BENCHMARK, tasa_acierto=tasa)
    n_ok = sum(r['correcto'] for r in res)
    prec = n_ok / len(BENCHMARK)
    if prec <= 0.25:   interp = 'Equivale a adivinar al azar (4 opciones)'
    elif prec <= 0.50: interp = 'Por debajo del azar corregido'
    elif prec <= 0.75: interp = 'Conocimiento basico'
    elif prec < 1.0:   interp = 'Buen dominio del tema'
    else:              interp = 'Dominio completo'
    print(f'{nombre:>20}  {n_ok:>4}/{len(BENCHMARK):>4}  {prec:>12.1%}  {interp}')
 
print()
print('En benchmarks reales como MMLU (57 materias, 14.000 preguntas):')
print('  GPT-3.5: ~70%   GPT-4: ~86%   Claude 3 Opus: ~87%')
print('  Humano experto: ~90%  Humano promedio: ~57%')
