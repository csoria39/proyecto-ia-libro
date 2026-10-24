# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)

# =============================================================================
#  PROGRAMA 8 — NANO-LLM: UN GPT-2 EN MINIATURA (versión extensamente comentada)
#  ----------------------------------------------------------------------------
#  Modelo de lenguaje (LLM) de nivel CARÁCTER, implementado ÚNICAMENTE
#  con NumPy — sin PyTorch, sin TensorFlow, sin ninguna librería de
#  aprendizaje profundo. Todo lo que hace un framework "por debajo"
#  (propagar hacia adelante, calcular gradientes, actualizar pesos)
#  está escrito aquí a mano, línea por línea, para que se pueda ver
#  y entender exactamente qué ocurre en cada paso.
#
#  Hace lo mismo, y da exactamente la misma salida, que la versión
#  compacta del Programa 8 impresa en el Capítulo 13 (13.5.1).
#
#  ¿QUÉ ES UN "NANO-LLM"?
#  Es GPT-2 (OpenAI, 2019) en miniatura: la MISMA arquitectura, pieza
#  a pieza, con tamaños diminutos. Aprende a predecir el SIGUIENTE
#  CARÁCTER de un texto, dado un fragmento de contexto. Repitiendo ese
#  proceso carácter a carácter, "escribe" frases que se parecen a las
#  del texto con el que fue entrenado.
#
#                       este modelo     GPT-2 small
#      contexto (CTX)        24            1024
#      dimensión (D)         48             768
#      cabezas (H)            4              12
#      bloques (N_BLQ)        2              12
#      vocabulario (V)       46          50 257 (BPE)
#      parámetros        60 000      124 millones
#
#  ARQUITECTURA (de abajo hacia arriba):
#
#      texto de entrada (caracteres)
#              │
#              ▼
#        Tokenizador (carácter → número entero)
#              │
#              ▼
#        wte[token] + wpe[posición]          (número → vector de D = 48)
#              │
#              ▼
#      ┌──────────────────────────────────┐
#      │ Bloque Transformer (Pre-LN)        │   × N_BLQ  (aquí, 2 veces)
#      │  X = X + Atención(LayerNorm(X))     │   4 cabezas, máscara causal
#      │  X = X + FFN(LayerNorm(X))          │   48 → 192 (GELU) → 48
#      └──────────────────────────────────┘
#              │
#              ▼
#        LayerNorm final
#              │
#              ▼
#        logits = vector · wteᵀ   (la MISMA tabla de la entrada)
#              │
#              ▼
#        Carácter muestreado  →  se añade al texto  →  se repite
#
#  FLUJO GENERAL DEL PROGRAMA (en el orden en que aparece el código):
#    1. Se define el texto de entrenamiento (TEXTO).
#    2. Se construye el tokenizador (carácter <-> índice entero).
#    3. Se fijan los hiperparámetros (tamaños, tasa de aprendizaje...).
#    4. Se inicializan aleatoriamente todos los parámetros (pesos).
#    5. Se definen el forward (predicción) y el forward + backward
#       (pérdida y gradiente exacto de todos los pesos, con
#       backpropagation escrito a mano).
#    6. Se entrena con Adam durante PASOS pasos, con mini-lotes,
#       warmup + coseno y recorte del gradiente (gradient clipping).
#    7. Se genera texto nuevo usando el modelo ya entrenado.
#    8. Se guardan los pesos entrenados en nano_llm_pesos.npz.
#
#  Todas las matrices que el modelo "aprende" están en el diccionario
#  `params`. Entrenar consiste, únicamente, en ir ajustando poco a
#  poco los números de esas matrices para que el modelo se equivoque
#  cada vez menos al predecir el siguiente carácter.
# =============================================================================

import numpy as np   # única dependencia externa: cálculo numérico con arrays/matrices

# -----------------------------------------------------------------------------
# TEXTO DE ENTRENAMIENTO
# -----------------------------------------------------------------------------
# Es el "libro" del que el modelo aprenderá. Cuanto más texto y más
# variado, más patrones distintos puede aprender — pero también hace
# falta más capacidad de modelo (ver D, D_FF más abajo) para no perder
# calidad, un compromiso conocido como "leyes de escalado".
#
# 93 líneas en español, con tildes, la letra "ñ" y signos de puntuación
# variados (¿ ? ¡ ! , . ; : ( ) -), pensadas para que el vocabulario
# resultante sea exactamente de 46 caracteres distintos (ver más abajo).
# El contenido cubre, a propósito, muchos conceptos de IA/ML para que
# el propio texto de entrenamiento sirva como resumen del tema.
TEXTO = """
la inteligencia artificial aprende de los datos.
los modelos de lenguaje predicen el siguiente token.
las redes neuronales tienen capas de neuronas.
el aprendizaje profundo usa muchas capas ocultas.
la atención permite ver todo el contexto a la vez.
los transformers son la base de los llm modernos.
entrenar un modelo requiere muchos datos y cálculo.
el futuro pertenece a quienes comprenden la ia.
un modelo pequeño puede aprender patrones simples.
la tokenización convierte texto en números enteros.
cada carácter del vocabulario tiene un índice único.
el embedding transforma índices en vectores densos.
la codificación posicional indica el orden del texto.
la función softmax convierte logits en probabilidades.
el optimizador adam ajusta los pesos, paso a paso.
una red profunda combina atención y capas ocultas.
generar texto significa predecir, uno a uno, cada token.
la temperatura controla cuánta variedad hay al generar.
un buen modelo equilibra memoria y generalización.
¿comprender la ia? requiere práctica, paciencia y datos.
también existe el software: gpt-2 fue un hito clave.
un modelo (bien entrenado) generaliza; uno malo, no.
¡el aprendizaje nunca termina! sigue explorando.
un vector es una lista ordenada de números reales.
una matriz agrupa vectores en filas y columnas.
el producto punto mide cuánto se parecen dos vectores.
la retropropagación calcula gradientes hacia atrás.
el descenso de gradiente ajusta los pesos poco a poco.
una época recorre todo el conjunto de entrenamiento.
el sobreajuste ocurre cuando el modelo memoriza demasiado.
la regularización ayuda a que el modelo generalice mejor.
un lote agrupa varios ejemplos para entrenar juntos.
la normalización estabiliza las activaciones internas.
una neurona combina entradas, pesos y un sesgo.
la función de pérdida mide el error del modelo.
el aprendizaje supervisado usa ejemplos etiquetados.
una capa oculta transforma la información paso a paso.
el descenso estocástico usa lotes pequeños y aleatorios.
la validación comprueba si el modelo generaliza bien.
un hiperparámetro se ajusta antes de entrenar el modelo.
la arquitectura define cómo se conectan las capas.
el vocabulario agrupa todos los símbolos posibles.
una secuencia es un orden concreto de tokens.
el contexto limita cuántos tokens ve el modelo a la vez.
la generalización permite responder a datos nuevos.
comprender estos conceptos abre la puerta a la ia moderna.
un dataset agrupa todos los ejemplos disponibles.
el conjunto de test evalúa el modelo al final, una sola vez.
la precisión mide cuántas predicciones son correctas.
el error mide cuánto se aleja la predicción del valor real.
una etiqueta indica la respuesta correcta de un ejemplo.
la inferencia usa el modelo ya entrenado, sin ajustar pesos.
el preentrenamiento aprende patrones generales del lenguaje.
el ajuste fino especializa un modelo en una tarea concreta.
un token puede ser una letra, una parte de palabra o una palabra.
la ventana de contexto limita cuánto texto ve el modelo.
un residual conecta la entrada directamente con la salida.
la normalización evita que las activaciones crezcan sin control.
el ruido en los datos dificulta encontrar el patrón real.
una red ancha tiene muchas neuronas en cada capa.
una red profunda tiene muchas capas apiladas.
el paralelismo permite entrenar varias partes a la vez.
la memoria limita cuántos datos caben en una gráfica.
un acelerador gráfico multiplica matrices muy rápido.
la escala mide cuántos parámetros tiene un modelo.
un modelo grande necesita muchos más datos para aprender bien.
la eficiencia importa tanto como la precisión final.
el costo de entrenar crece con el tamaño del modelo.
la interpretabilidad explica por qué el modelo decide algo.
un sesgo en los datos puede producir un modelo injusto.
la ética en la ia estudia el impacto social de los modelos.
la privacidad protege los datos personales de los usuarios.
un modelo robusto resiste pequeños cambios en la entrada.
la incertidumbre mide cuánta confianza tiene el modelo.
un ejemplo difícil confunde incluso a un buen modelo.
la curva de aprendizaje muestra el progreso del entrenamiento.
un mínimo local complica encontrar la mejor solución.
el paisaje de pérdida describe cómo cambia el error.
una meseta ralentiza el progreso del entrenamiento.
el ruido ayuda a veces a escapar de mínimos malos.
la exploración prueba caminos nuevos durante el entrenamiento.
la explotación aprovecha lo que ya funciona bien.
un modelo generativo produce contenido nuevo y original.
un modelo discriminativo distingue entre categorías distintas.
la clasificación asigna una categoría a cada ejemplo.
la regresión predice un número continuo, no una categoría.
el aprendizaje por refuerzo premia las buenas decisiones.
un agente aprende actuando dentro de un entorno.
la recompensa guía el aprendizaje del agente.
la política define cómo actúa el agente en cada estado.
un entorno simulado permite entrenar sin riesgo real.
la simulación acelera enormemente el aprendizaje del agente.
comprender bien lo básico facilita entender lo avanzado.
"""

# -----------------------------------------------------------------------------
# TOKENIZADOR A NIVEL DE CARÁCTER
# -----------------------------------------------------------------------------
# Una red neuronal solo entiende números, no letras. El "tokenizador"
# es el traductor entre texto y números en ambas direcciones.
#
# Este es el tokenizador MÁS SIMPLE posible: cada carácter distinto
# del texto (una letra, un espacio, un signo de puntuación...) se
# convierte en UN token. Los LLMs de verdad (GPT, Claude...) usan
# tokenizadores más sofisticados a nivel de "subpalabra" (por ejemplo,
# BPE), donde un token puede ser una palabra entera o un trozo de ella,
# lo que hace las secuencias más cortas y eficientes. Aquí, por
# claridad didáctica, se usa el nivel de carácter: más lento de
# entrenar, pero mucho más fácil de entender y depurar.
VOCAB  = sorted(set(TEXTO))
# `set(TEXTO)` recorre el string TEXTO y se queda con cada carácter
# ÚNICO que aparece en él (sin repeticiones). `sorted(...)` los ordena
# (por su código Unicode) para que el orden sea siempre el mismo cada
# vez que se ejecuta el programa — importante para reproducibilidad.
# Resultado: una lista de 46 caracteres distintos, p. ej.
# ['\n', ' ', '!', '(', ')', ',', ..., 'z', '¡', '¿', 'á', 'é', ...].

V      = len(VOCAB)
# V = tamaño del vocabulario = número de caracteres distintos = 46.
# Es la "V" que aparecerá en muchas fórmulas y tamaños de matrices
# más abajo (p. ej. la tabla de embeddings tiene forma (V, D)).

C2I    = {c: i for i, c in enumerate(VOCAB)}
# "Character To Index": diccionario que traduce cada carácter a su
# número entero (su posición en VOCAB). Ejemplo: C2I['a'] podría ser 12.
# `enumerate(VOCAB)` genera pares (índice, carácter); el diccionario
# por comprensión los invierte a {carácter: índice}.

I2C    = dict(enumerate(VOCAB))
# "Index To Character": el diccionario inverso, traduce cada número
# de vuelta a su carácter original. Se usa al GENERAR texto, para
# convertir los índices que predice el modelo de nuevo en letras.

DATOS  = np.array([C2I[c] for c in TEXTO], dtype=int)
# Se recorre el texto completo, carácter a carácter, y se sustituye
# cada uno por su índice entero (usando C2I). El resultado es un
# array de NumPy de enteros: la "traducción numérica" de todo el
# texto de entrenamiento. Es sobre este array (DATOS) de donde se
# extraerán, más adelante, las ventanas de contexto para entrenar.

# -----------------------------------------------------------------------------
# HIPERPARÁMETROS
# -----------------------------------------------------------------------------
# Los "hiperparámetros" son las decisiones de diseño que el programador
# fija ANTES de entrenar (a diferencia de los "parámetros", que son los
# pesos que el modelo aprende solo). Cambiarlos cambia la forma y el
# comportamiento del modelo.
CTX    = 24
# Longitud del "contexto": cuántos caracteres previos puede ver el
# modelo como máximo para predecir el siguiente. Con CTX=24, el modelo
# solo "recuerda" los últimos 24 caracteres — más o menos 4-5 palabras
# cortas en español. Los LLMs reales usan contextos de miles o cientos
# de miles de tokens; aquí se mantiene pequeño para que el modelo siga
# siendo manejable y rápido de entrenar.

D      = 48
# Dimensión del "embedding": cuántos números se usan para representar
# internamente cada carácter (su "vector de significado"). Cuanto
# mayor es D, más matices puede capturar el modelo sobre cada
# carácter y su contexto, pero también hay más parámetros que
# aprender y más lento va el entrenamiento. GPT-2 small usa D = 768.

H      = 4
# Número de "cabezas" de atención. En vez de una única atención sobre
# los 48 números de cada vector, se hacen 4 atenciones en paralelo,
# cada una sobre 12 de ellos (DH = D / H = 12). Cada cabeza puede
# especializarse en una relación distinta: una puede mirar el carácter
# anterior; otra, el principio de la palabra. GPT-2 small usa 12
# cabezas de 64 dimensiones.

DH     = D // H

D_FF   = 192
# Dimensión interna de la red "Feed-Forward" (FFN) que hay dentro de
# cada bloque transformer (ver más abajo). Por convención, suele ser
# 4 veces D (48 × 4 = 192): esa capa intermedia más ancha le da al
# modelo más "espacio de trabajo" para procesar cada posición antes
# de comprimir el resultado de vuelta a dimensión D.

N_BLQ  = 2
# Número de "bloques transformer" apilados uno tras otro (ver el
# diagrama de la cabecera). Cada bloque añade una ronda más de
# "prestar atención al contexto + procesar la información". Los LLMs
# grandes apilan decenas o cientos de bloques; aquí, con solo 2, el
# modelo ya es capaz de aprender patrones de lenguaje simples.

LR     = 3e-3
# "Learning Rate" o tasa de aprendizaje: de qué tamaño son los pasos
# con los que se ajustan los pesos en cada actualización. Un valor
# demasiado alto hace que el entrenamiento sea inestable (los pesos
# "saltan" de un lado a otro sin asentarse); demasiado bajo hace que
# el aprendizaje sea desesperantemente lento. 3e-3 (0.003) es un
# valor típico para el optimizador Adam con modelos de este tamaño.
# Nótese que este valor es el MÁXIMO: durante el entrenamiento se
# modula con un "warmup + decaimiento coseno" (ver más abajo).

PASOS  = 12000
# Número de veces que se actualizan los pesos: PASOS de optimización.
# Cada paso procesa un mini-lote de BATCH ventanas elegidas al azar,
# no todo el texto, así que un paso NO es una "época" (una época es
# una pasada completa por todos los datos; ver el glosario). Cuenta:
# 12000 pasos × 8 ventanas × 24 caracteres ≈ 2,3 millones de
# predicciones ≈ 454 pasadas por los 5076 caracteres del corpus.

BATCH  = 8
# Cuántas secuencias de entrenamiento distintas se procesan juntas
# antes de actualizar los pesos una vez. Promediar el gradiente de
# varias secuencias (en vez de actualizar tras cada una) suaviza el
# entrenamiento: reduce el "ruido" de que una sola secuencia rara
# empuje los pesos en una dirección poco representativa.

CLIP  = 1.0
# Recorte de la norma del gradiente (Capítulo 5, 5.8): si la
# norma global del gradiente (la raíz de la suma de los cuadrados de
# TODOS sus elementos) supera CLIP, el gradiente entero se reescala
# para que su norma valga exactamente CLIP. La dirección no cambia;
# solo se limita el tamaño. Es una red de seguridad contra un paso
# desmesurado provocado por un mini-lote anómalo.

np.random.seed(42)
# Fija la "semilla" del generador de números aleatorios de NumPy.
# Con la misma semilla, todos los números "aleatorios" que se generen
# a partir de aquí (inicialización de pesos, elección de qué ventana
# de texto usar en cada paso, muestreo al generar texto...) serán
# EXACTAMENTE los mismos cada vez que se ejecute el programa. Esto
# hace que el experimento sea reproducible: cualquiera que ejecute
# este mismo código debería obtener los mismos resultados.

# -----------------------------------------------------------------------------
# INICIALIZACIÓN DE LOS PARÁMETROS (PESOS) DEL MODELO
# -----------------------------------------------------------------------------
# Un modelo "sin entrenar" no es más que un conjunto de matrices
# llenas de números aleatorios pequeños. El entrenamiento consiste,
# únicamente, en ir corrigiendo esos números poco a poco (con
# backpropagation + Adam) para que el modelo acierte cada vez más.

def randn(shape, std=0.02):
    """
    Genera una matriz de la forma (shape) dada, con valores aleatorios
    siguiendo una distribución normal (campana de Gauss) de media 0 y
    desviación típica `std` (0.02 por defecto, como en GPT-2): la
    mayoría de los valores quedan entre -0.06 y 0.06.

    ¿Por qué valores tan pequeños? Si los pesos iniciales fueran muy
    grandes, las primeras predicciones del modelo serían caóticas y
    los gradientes iniciales podrían "explotar" (números enormes que
    desestabilizan el entrenamiento desde el primer paso). Empezar
    "casi en cero" da un punto de partida neutro y estable.
    """
    return np.random.randn(*shape) * std

# Las dos proyecciones cuya salida se SUMA a la vía residual (Wo, al
# final de la atención, y W2, al final de la FFN) se inician con una
# desviación todavía menor: 0.02 / sqrt(2·N_BLQ). Es la receta de
# GPT-2: cada bloque suma dos contribuciones a la vía residual, y con
# N_BLQ bloques hay 2·N_BLQ sumas; dividir por su raíz cuadrada evita
# que, al apilar muchos bloques, la vía residual crezca sin control
# antes de empezar a entrenar.
STD_RES = 0.02 / np.sqrt(2 * N_BLQ)

# `params` es un diccionario que contiene TODAS las matrices que el
# modelo puede aprender. Cada entrada es una pieza del "cerebro" del
# modelo. Los nombres son los mismos que usa GPT-2 (wte, wpe, ln_f...).
params = {
    # Tabla de embeddings de TOKEN ("word token embedding"): una fila
    # por cada carácter del vocabulario (V filas), cada una con D
    # números. Es la "traducción" de cada carácter a un vector que la
    # red puede procesar. Forma: (V, D) = (46, 48).
    # En GPT-2 esta MISMA matriz se usa también a la salida, para
    # convertir el vector final en una puntuación por carácter: se
    # dice que entrada y salida "comparten pesos" (weight tying). Por
    # eso no hay una matriz de salida aparte.
    'wte':  randn((V, D)),

    # Tabla de embeddings POSICIONALES ("word position embedding"):
    # una fila por cada posición de la ventana de contexto (CTX
    # filas). Le dice al modelo "en qué lugar de la secuencia" está
    # cada carácter, algo que la atención, por sí sola, no distingue.
    # Es APRENDIDA, como en GPT-2 (el Transformer original de 2017 la
    # calculaba con senos y cosenos; ver Capítulo 12). Forma: (CTX, D).
    'wpe':  randn((CTX, D)),

    # LayerNorm FINAL: se aplica una vez, después del último bloque y
    # antes de la salida. g (gamma, escala) empieza en 1 y b (beta,
    # desplazamiento) en 0: al principio la LayerNorm solo normaliza.
    'lnf_g': np.ones(D), 'lnf_b': np.zeros(D),
}

# A continuación se crean los parámetros de cada BLOQUE TRANSFORMER
# (hay N_BLQ = 2 bloques, numerados 0 y 1). Cada bloque tiene su
# propio juego de matrices, completamente independiente.
for k in range(N_BLQ):
    # --- LayerNorm previa a la atención (ln1) ---
    params[f'ln1_g{k}'] = np.ones(D);  params[f'ln1_b{k}'] = np.zeros(D)

    # --- Matrices de ATENCIÓN (self-attention) ---
    # Wq, Wk, Wv proyectan la entrada en tres "vistas" del mismo
    # vector, cada una con un papel distinto:
    #   Q (Query)  = "¿qué información estoy buscando?"
    #   K (Key)    = "¿qué información ofrezco yo a los demás?"
    #   V (Value)  = "el contenido real que se transmite si me
    #                 prestan atención"
    # Cada una es (D, D) y lleva su sesgo de tamaño D, como en GPT-2.
    # Las H = 4 cabezas NO tienen matrices propias: cada cabeza usa
    # una "rebanada" de DH = 12 columnas de estas matrices (ver
    # separar(), más abajo).
    for n in 'qkv':
        params[f'W{n}{k}'] = randn((D, D)); params[f'b{n}{k}'] = np.zeros(D)
    # Wo junta lo que han recogido las 4 cabezas y lo proyecta de
    # vuelta a dimensión D. Forma: (D, D), con sesgo.
    params[f'Wo{k}'] = randn((D, D), STD_RES); params[f'bo{k}'] = np.zeros(D)

    # --- LayerNorm previa a la FFN (ln2) ---
    params[f'ln2_g{k}'] = np.ones(D);  params[f'ln2_b{k}'] = np.zeros(D)

    # --- Matrices de la FEED-FORWARD NETWORK (FFN) ---
    # Dos capas lineales con una activación GELU en medio. Procesa
    # CADA posición por separado (la atención es la que mezcla
    # información entre posiciones).
    params[f'W1{k}'] = randn((D, D_FF)); params[f'b1{k}'] = np.zeros(D_FF)   # expande D -> D_FF
    params[f'W2{k}'] = randn((D_FF, D), STD_RES); params[f'b2{k}'] = np.zeros(D)  # comprime D_FF -> D

# Comprobación rápida en pantalla: tamaño del vocabulario, número
# total de parámetros (60.000 justos con estos tamaños) y cuánto texto
# se va a usar para entrenar.
total = sum(p.size for p in params.values())
print(f'Vocabulario: {V} caracteres')
print(f'Total de parámetros: {total:,}')
print(f'Texto de entrenamiento: {len(DATOS)} tokens, '
      f'{TEXTO.strip().count(chr(10)) + 1} líneas')

# -----------------------------------------------------------------------------
# MÁSCARA CAUSAL (para que el modelo no "haga trampa" mirando el futuro)
# -----------------------------------------------------------------------------
# Cuando el modelo predice el carácter en la posición t, NO debe
# poder "espiar" los caracteres en las posiciones t+1, t+2... (eso
# sería aprender a copiar la respuesta en vez de predecirla). La
# máscara causal se lo impide: es una matriz (CTX, CTX) de ceros,
# excepto por encima de la diagonal principal, donde vale -1e9.
#
# La máscara se SUMA a las puntuaciones de atención antes del
# softmax: sumar -1e9 hace que esa posición reciba un peso de
# atención prácticamente 0, es decir, "apaga" el futuro.
#
# Al entrenar, la longitud de la secuencia (T) es siempre CTX; al
# generar puede ser menor (al principio, con semillas cortas). Por eso
# la máscara se calcula UNA SOLA VEZ aquí, con el tamaño máximo, y se
# usa su esquina superior izquierda [:T, :T].
MASCARA_CAUSAL = np.triu(np.full((CTX, CTX), -1e9), k=1)


# -----------------------------------------------------------------------------
# UTILIDADES MATEMÁTICAS
# -----------------------------------------------------------------------------
def softmax(x, axis=-1):
    """
    Convierte un vector de números cualesquiera en una distribución
    de probabilidad: valores entre 0 y 1 que suman exactamente 1 a lo
    largo del eje `axis`.

    Fórmula: softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    TRUCO DE ESTABILIDAD NUMÉRICA: antes de exponenciar se resta el
    máximo. El resultado matemático no cambia, pero se evita calcular
    exponenciales enormes que desbordarían el float (inf o nan).
    """
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)


def layer_norm_fwd(x, g, b, eps=1e-5):
    """
    LayerNorm ("normalización de capa"), exactamente como en GPT-2:
    para cada fila (cada posición) de x, resta su media y divide por
    su desviación típica (con eps dentro de la raíz), y después
    REESCALA con g (gamma) y DESPLAZA con b (beta):

        xhat = (x − media) / sqrt(varianza + eps)
        y    = xhat · g + b

    ¿Para qué sirve? A medida que la información atraviesa bloques,
    sus valores pueden crecer o desplazarse sin control. LayerNorm
    devuelve cada fila a una escala estándar antes de que entre en la
    atención o en la FFN. Con g y b el modelo puede, si le conviene,
    aprender otra escala y otro centro distintos de (1, 0).

    Devuelve y y, en un "cache", lo que el backward necesitará:
    xhat (la fila normalizada) e inv = 1 / sqrt(varianza + eps).
    """
    mu = x.mean(-1, keepdims=True)                       # media de cada fila
    inv = 1.0 / np.sqrt(x.var(-1, keepdims=True) + eps)  # 1 / desviación típica
    xhat = (x - mu) * inv                                # fila normalizada
    return xhat * g + b, (xhat, inv)


def layer_norm_bwd(dy, g, cache):
    """
    Derivada (backward) EXACTA de la LayerNorm de arriba. Recibe
    `dy`, el gradiente de la pérdida respecto a la SALIDA y, y
    devuelve tres gradientes: respecto a la entrada x, a g y a b.

      · db = suma de dy sobre todas las filas   (y = ... + b)
      · dg = suma de dy·xhat sobre todas las filas   (y = xhat·g + ...)
      · dx: primero se deshace la escala, dxhat = dy·g, y después la
        normalización, con la fórmula exacta

        dx = inv · ( dxhat − media(dxhat) − xhat · media(dxhat·xhat) )

    Intuición de los tres términos dentro del paréntesis:
      · dxhat              -> el gradiente "tal cual", si la
                              normalización no acoplara las componentes.
      · − media(dxhat)     -> corrige que restar la media acopla TODAS
                              las componentes de la fila entre sí.
      · − xhat · media(...) -> corrige que dividir por la desviación
                              típica también depende de toda la fila.

    Es la misma matemática que usa la LayerNorm de PyTorch por
    debajo; aquí está escrita a la vista. El Ejercicio 13.6 la
    comprueba numéricamente.
    """
    xhat, inv = cache
    ejes = tuple(range(dy.ndim - 1))      # todas las filas (lote y posiciones)
    dg = (dy * xhat).sum(ejes)
    db = dy.sum(ejes)
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


C_GELU = np.sqrt(2 / np.pi)
def gelu(x):
    """
    GELU ("Gaussian Error Linear Unit"), la activación de GPT-2, con
    la misma aproximación por tanh que usa su código original:

        gelu(x) = 0.5 · x · (1 + tanh( sqrt(2/π) · (x + 0.044715·x³) ))

    Se parece a ReLU (≈ x para x grande y positivo, ≈ 0 para x muy
    negativo), pero es SUAVE: no tiene esquina en 0 y deja pasar un
    poco de los valores ligeramente negativos. x³ se escribe x*x*x
    porque, en NumPy, la potencia x**3 es unas quince veces más lenta.
    """
    return 0.5 * x * (1 + np.tanh(C_GELU * (x + 0.044715 * x * x * x)))


def gelu_bwd(x):
    """
    Derivada de gelu(x), por la regla del producto y la de la cadena.
    Con t = tanh(u) y u = sqrt(2/π)·(x + 0.044715·x³):

        gelu'(x) = 0.5·(1 + t) + 0.5·x·(1 − t²)·sqrt(2/π)·(1 + 3·0.044715·x²)

    (la derivada de tanh(u) es 1 − tanh²(u)). Sustituye a la derivada
    de ReLU (1 si x > 0, 0 si no) de la versión anterior del modelo.
    """
    t = np.tanh(C_GELU * (x + 0.044715 * x * x * x))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C_GELU * (1 + 3 * 0.044715 * x * x)


def separar(x):   # (B, T, D) -> (B, H, T, DH): una matriz por cabeza
    """
    Reparte las D = 48 columnas entre las H = 4 cabezas: la cabeza 0
    se queda con las columnas 0-11, la 1 con 12-23, etc. El resultado
    tiene un eje nuevo, H, para que las cuatro cabezas calculen su
    atención a la vez, con las mismas multiplicaciones de matrices.
    """
    B, T, _ = x.shape
    return x.reshape(B, T, H, DH).transpose(0, 2, 1, 3)


def unir(x):      # (B, H, T, DH) -> (B, T, D)
    """La operación inversa: vuelve a poner las 4 cabezas una al lado de otra."""
    B, _, T, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, D)


def tmm(a, b):
    """
    Gradiente de una capa lineal y = a @ W: dW = aᵀ · dy, SUMADO
    sobre todas las filas del lote y todas las posiciones. Para
    hacerlo con una sola multiplicación se aplanan los ejes del lote
    y de las posiciones: (B, T, n) -> (B·T, n).
    """
    return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])


# =============================================================================
# FORWARD: de los tokens a los logits
# =============================================================================
# Todo el programa trabaja con MINI-LOTES: en vez de una secuencia de
# T caracteres, se procesan B secuencias a la vez, apiladas en un
# tensor de forma (B, T). Las multiplicaciones de matrices de NumPy
# aceptan ejes extra delante, así que el código es el mismo que para
# una secuencia, y mucho más rápido que un bucle de Python.
def forward(tokens):
    """
    tokens : array de enteros de forma (B, T).
    Devuelve los logits (B, T, V) — una puntuación por carácter
    posible, para cada posición de cada secuencia — y un "cache" con
    los valores intermedios que el backward necesitará.
    """
    B, T = tokens.shape

    # --- Embeddings: qué carácter es + en qué posición está ---
    X = params['wte'][tokens] + params['wpe'][:T]       # (B, T, D)

    cache = []
    for k in range(N_BLQ):
        Xin = X   # entrada del bloque k (vía residual)

        # --- SUB-CAPA 1: LayerNorm -> atención causal (4 cabezas) -> suma residual ---
        # PRE-LN, como GPT-2: se normaliza ANTES de la subcapa, y la
        # salida de la subcapa se SUMA a Xin sin normalizarla. La vía
        # residual (Xin, X1, X...) nunca pasa por una LayerNorm dentro
        # del bloque: es un "carril" por el que la información, y en el
        # backward el gradiente, circula sin obstáculos.
        a, ln1 = layer_norm_fwd(Xin, params[f'ln1_g{k}'], params[f'ln1_b{k}'])

        # Q, K, V para las 4 cabezas a la vez: (B, H, T, DH)
        Q, K, Vm = (separar(a @ params[f'W{n}{k}'] + params[f'b{n}{k}']) for n in 'qkv')

        # Puntuaciones Q·Kᵀ de cada cabeza, divididas por sqrt(DH)
        # (12, la dimensión de CADA cabeza: es la que fija la escala de
        # los productos), más la máscara causal; softmax por filas.
        pesos = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(DH) + MASCARA_CAUSAL[:T, :T])

        # Suma ponderada de los Values; las 4 cabezas se vuelven a unir
        att = unir(pesos @ Vm)                            # (B, T, D)

        # Proyección de salida y suma a la vía residual
        X1 = Xin + att @ params[f'Wo{k}'] + params[f'bo{k}']

        # --- SUB-CAPA 2: LayerNorm -> FFN con GELU -> suma residual ---
        m, ln2 = layer_norm_fwd(X1, params[f'ln2_g{k}'], params[f'ln2_b{k}'])
        h_pre = m @ params[f'W1{k}'] + params[f'b1{k}']  # (B, T, D_FF)
        h = gelu(h_pre)
        X = X1 + h @ params[f'W2{k}'] + params[f'b2{k}']

        # Se guarda todo lo que el backward necesitará de este bloque
        cache.append(dict(a=a, ln1=ln1, Q=Q, K=K, Vm=Vm, pesos=pesos,
                          att=att, m=m, ln2=ln2, h_pre=h_pre, h=h))

    # --- LayerNorm final y salida ---
    xf, lnf = layer_norm_fwd(X, params['lnf_g'], params['lnf_b'])
    # PESOS COMPARTIDOS: la salida usa la MISMA matriz wte de la
    # entrada, traspuesta. El logit de un carácter es el producto
    # escalar del vector final con el embedding de ese carácter:
    # "¿a qué carácter se parece más lo que viene ahora?".
    logits = xf @ params['wte'].T                         # (B, T, V)
    return logits, (xf, lnf, cache)


# =============================================================================
# FORWARD + BACKWARD: la pérdida y su gradiente exacto
# =============================================================================
# Esta función hace DOS cosas:
#
#   1. FORWARD: llama a forward() y, comparando las predicciones con
#      los caracteres reales (targets), calcula la PÉRDIDA (loss), un
#      único número que mide cuánto se equivoca el modelo.
#
#   2. BACKWARD (backpropagation): recorre el camino del forward AL
#      REVÉS aplicando la regla de la cadena, y obtiene el GRADIENTE
#      de la pérdida respecto a CADA parámetro: "en qué dirección y
#      cuánto debería cambiar ese peso para reducir un poco la
#      pérdida". Es lo que hace `loss.backward()` en PyTorch, escrito
#      aquí a la vista, matriz a matriz.
#
# Dos reglas se repiten una y otra vez:
#   · Capa lineal y = a @ W + b:  dW = aᵀ·dy (tmm),  db = suma de dy,
#     y hacia atrás da = dy @ Wᵀ.
#   · Una SUMA en el forward (la vía residual) COPIA el gradiente a
#     las dos ramas; una variable usada en DOS sitios (como wte, en la
#     entrada y en la salida) SUMA los gradientes de ambos usos.
def forward_backward(tokens, targets):
    """
    tokens, targets : arrays de enteros (B, T); targets es la misma
                      ventana de texto desplazada un carácter.
    Devuelve (grads, loss): un diccionario con el gradiente de cada
    matriz de `params` y la pérdida media del mini-lote.
    """
    B, T = tokens.shape
    logits, (xf, lnf, cache) = forward(tokens)
    probs = softmax(logits)

    # --- PÉRDIDA: entropía cruzada media ---
    # probs[bi, ti, targets] es la probabilidad que el modelo dio al
    # carácter CORRECTO en cada posición de cada secuencia.
    bi, ti = np.arange(B)[:, None], np.arange(T)[None, :]
    loss = -np.log(probs[bi, ti, targets] + 1e-10).mean()

    grads = {k: np.zeros_like(v) for k, v in params.items()}

    # --- Gradiente respecto a los logits ---
    # Para softmax + entropía cruzada la derivada es sencillísima:
    # (probabilidades − 1 en el carácter correcto), dividido entre el
    # número de predicciones promediadas (B·T).
    dlogits = probs.copy()
    dlogits[bi, ti, targets] -= 1.0
    dlogits /= B * T

    # --- Salida: logits = xf @ wteᵀ ---
    grads['wte'] += tmm(dlogits, xf)         # 1er uso de wte (como matriz de salida)
    dxf = dlogits @ params['wte']
    dX, grads['lnf_g'], grads['lnf_b'] = layer_norm_bwd(dxf, params['lnf_g'], lnf)

    # --- Bloques transformer, del último al primero ---
    for k in reversed(range(N_BLQ)):
        c = cache[k]

        # -- Subcapa 2 hacia atrás: X = X1 + h @ W2 + b2 --
        # La suma residual copia dX a las dos ramas: a X1 (atajo) y a
        # la FFN.
        grads[f'W2{k}'] = tmm(c['h'], dX)
        grads[f'b2{k}'] = dX.sum((0, 1))
        # Deshacer la GELU: se multiplica por su derivada
        dh_pre = (dX @ params[f'W2{k}'].T) * gelu_bwd(c['h_pre'])
        grads[f'W1{k}'] = tmm(c['m'], dh_pre)
        grads[f'b1{k}'] = dh_pre.sum((0, 1))
        dm = dh_pre @ params[f'W1{k}'].T
        # Deshacer la LayerNorm ln2 (devuelve también dg y db)
        dx1, grads[f'ln2_g{k}'], grads[f'ln2_b{k}'] = layer_norm_bwd(dm, params[f'ln2_g{k}'], c['ln2'])
        dX1 = dX + dx1                       # atajo residual + rama FFN

        # -- Subcapa 1 hacia atrás: X1 = Xin + att @ Wo + bo --
        grads[f'Wo{k}'] = tmm(c['att'], dX1)
        grads[f'bo{k}'] = dX1.sum((0, 1))
        datt = separar(dX1 @ params[f'Wo{k}'].T)   # de vuelta a 4 cabezas

        # Deshacer la suma ponderada: att = pesos @ Vm
        dpesos = datt @ c['Vm'].transpose(0, 1, 3, 2)
        dVm = c['pesos'].transpose(0, 1, 3, 2) @ datt

        # Deshacer el softmax de la atención, fila a fila:
        #     dscores_i = p_i · (dp_i − Σ_j dp_j·p_j)
        # y, en la misma línea, la división por sqrt(DH).
        s = (dpesos * c['pesos']).sum(-1, keepdims=True)
        dscores = c['pesos'] * (dpesos - s) / np.sqrt(DH)

        # Deshacer Q @ Kᵀ, y volver a unir las cabezas
        dQ = unir(dscores @ c['K'])
        dK = unir(dscores.transpose(0, 1, 3, 2) @ c['Q'])
        dV = unir(dVm)

        # Deshacer las tres proyecciones Q, K, V. Las tres parten de la
        # MISMA entrada a, así que sus contribuciones a da se SUMAN.
        # (Curiosidad: el gradiente de bk sale siempre 0 exacto. Sumar
        # el mismo vector a todas las claves suma la misma constante a
        # toda una fila de puntuaciones, y el softmax no cambia si a
        # toda la fila se le suma lo mismo. GPT-2 tiene bk de todos
        # modos; aquí se mantiene para que el modelo sea idéntico.)
        da = np.zeros_like(c['a'])
        for n, dn in zip('qkv', (dQ, dK, dV)):
            grads[f'W{n}{k}'] = tmm(c['a'], dn)
            grads[f'b{n}{k}'] = dn.sum((0, 1))
            da += dn @ params[f'W{n}{k}'].T
        # Deshacer la LayerNorm ln1
        dxin, grads[f'ln1_g{k}'], grads[f'ln1_b{k}'] = layer_norm_bwd(da, params[f'ln1_g{k}'], c['ln1'])
        dX = dX1 + dxin                      # atajo residual + rama de atención

    # --- Embeddings: X = wte[tokens] + wpe[:T] ---
    # Hacia wpe: cada posición recibe la suma de su gradiente en las
    # B secuencias del lote.
    grads['wpe'][:T] = dX.sum(0)
    # Hacia wte (2º uso, como tabla de entrada): un mismo carácter
    # aparece muchas veces en el lote, y cada aparición aporta su
    # gradiente a la MISMA fila. np.add.at los SUMA todos (una
    # asignación normal se quedaría solo con el último). Y se suman
    # también al gradiente de wte como matriz de salida, calculado
    # arriba: una matriz usada en dos sitios recibe los dos.
    np.add.at(grads['wte'], tokens, dX)
    return grads, loss


# =============================================================================
# LEARNING RATE: WARMUP LINEAL + DECAIMIENTO EN COSENO
# =============================================================================
def lr_warmup_coseno(t, lr_max, t_total, t_warmup=150, lr_min=1e-4):
    """
    Calcula qué tasa de aprendizaje usar en el paso `t` del
    entrenamiento (de un total de `t_total` pasos). En vez de usar
    siempre el mismo LR fijo, se sigue una curva en dos tramos —el
    esquema estándar para entrenar modelos tipo Transformer:

      TRAMO 1 — "warmup" (calentamiento), los primeros `t_warmup`
      pasos: el LR sube LINEALMENTE desde 0 hasta lr_max. Al
      principio del entrenamiento los pesos son aleatorios y los
      gradientes pueden ser grandes y poco fiables; empezar con un
      LR muy bajo evita que esos primeros pasos, inestables, den
      saltos demasiado bruscos que desestabilicen el entrenamiento.

      TRAMO 2 — decaimiento en COSENO, el resto de pasos: el LR baja
      suavemente desde lr_max hasta lr_min, siguiendo la forma de
      media onda de coseno (empieza bajando despacio, acelera en el
      medio, y vuelve a suavizarse al final). Esto permite dar pasos
      grandes mientras el modelo aún tiene mucho que aprender, y
      pasos cada vez más pequeños y precisos a medida que se acerca
      a una buena solución — como "afinar" el resultado al final.

    Parámetros
    ----------
    t         : paso actual del bucle de entrenamiento (0, 1, 2...).
    lr_max    : tasa de aprendizaje máxima, alcanzada al final del warmup.
    t_total   : número total de pasos de entrenamiento (PASOS).
    t_warmup  : cuántos pasos dura la subida lineal inicial.
    lr_min    : tasa de aprendizaje mínima, al final del decaimiento.
    """
    if t < t_warmup:
        # Subida lineal. El +1 hace que el paso 0 ya aprenda (con
        # t / t_warmup, el primer paso tendría lr = 0 y no cambiaría
        # nada) y que en el paso t_warmup − 1 se alcance lr_max.
        return lr_max * (t + 1) / t_warmup
    # A partir de aquí, decaimiento en coseno desde lr_max hasta lr_min.
    t_adj = t - t_warmup           # pasos transcurridos DESDE que acabó el warmup
    t_decay = t_total - t_warmup   # duración total de la fase de decaimiento
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t_adj / t_decay))


# =============================================================================
# BUCLE DE ENTRENAMIENTO: OPTIMIZADOR ADAM CON MINI-LOTES
# =============================================================================
# Adam ("Adaptive Moment Estimation") es el optimizador más usado en
# la práctica para entrenar redes neuronales (y también el que usan,
# por defecto, los LLMs reales). En vez de simplemente restar
# `LR * gradiente` a cada peso (como el descenso de gradiente más
# básico), Adam mantiene, PARA CADA PARÁMETRO POR SEPARADO, dos
# "memorias" que se actualizan en cada paso:
#   m (primer momento)  -> una media móvil del gradiente reciente
#                          (funciona como un "impulso": suaviza la
#                          dirección de la actualización).
#   v (segundo momento) -> una media móvil del CUADRADO del
#                          gradiente reciente (mide cuánto "varía"
#                          el gradiente de ese parámetro).
# Dividir por sqrt(v) hace que los parámetros con gradientes
# históricamente grandes y erráticos reciban pasos más pequeños
# (para no desestabilizarlos), y los que tienen gradientes pequeños
# y estables reciban pasos relativamente mayores — cada parámetro
# "aprende a su propio ritmo".

# m_adam, v_adam: un diccionario con la misma forma que `params`,
# inicializado todo a cero — son las "memorias" de Adam para cada
# matriz de parámetros, que se irán actualizando en cada paso.
m_adam = {k: np.zeros_like(v) for k, v in params.items()}
v_adam = {k: np.zeros_like(v) for k, v in params.items()}

# Hiperparámetros estándar de Adam (los valores por defecto que
# recomienda el artículo original y que casi nunca hace falta tocar):
#   beta1 = 0.9   -> cuánto "recuerda" la media móvil del gradiente (m)
#   beta2 = 0.999 -> cuánto "recuerda" la media móvil del cuadrado (v)
#   eps_adam      -> número minúsculo para evitar dividir entre cero
#   t_adam        -> contador de pasos, usado para la "corrección de
#                    sesgo" (bias correction) de m y v — sin ella,
#                    m y v estarían artificialmente cerca de cero en
#                    los primeros pasos, porque empiezan en cero y
#                    aún no han "visto" suficientes gradientes.
beta1, beta2, eps_adam, t_adam = 0.9, 0.999, 1e-8, 0
LR_MAX = LR   # la tasa de aprendizaje máxima que usará el planificador coseno

print('\nEntrenando Nano-LLM (backpropagation analítico, mini-lotes)...')
print(f'CTX={CTX} | D={D} | D_FF={D_FF} | Bloques={N_BLQ} | Batch={BATCH}\n')

# --- BUCLE PRINCIPAL DE ENTRENAMIENTO ---
# Se repite PASOS veces. Cada iteración es un PASO de optimización:
# procesa solo BATCH secuencias elegidas al azar (no todo el texto)
# y actualiza los pesos una vez.
for paso in range(PASOS):

    # 1) Calcular la tasa de aprendizaje de este paso, según el
    #    esquema de warmup + coseno definido arriba.
    lr_actual = lr_warmup_coseno(paso, LR_MAX, PASOS)

    # 2) Elegir un MINI-LOTE de BATCH ventanas al azar. Para cada una
    #    se toma una posición de inicio dentro de DATOS (el texto ya
    #    convertido a números) y se extrae una ventana de CTX
    #    caracteres como entrada; la misma ventana desplazada un
    #    carácter a la derecha es el objetivo: "dado este trozo de
    #    texto, en cada posición viene después este carácter".
    inicios = np.random.randint(0, len(DATOS) - CTX - 1, size=BATCH)
    xb = np.stack([DATOS[i:i + CTX] for i in inicios])          # (BATCH, CTX)
    yb = np.stack([DATOS[i + 1:i + CTX + 1] for i in inicios])  # desplazado 1

    # 3) y 4) Forward + backward del mini-lote ENTERO de una vez. La
    #    pérdida y el gradiente ya salen PROMEDIADOS sobre las BATCH
    #    ventanas (dlogits se divide entre BATCH·CTX). Promediar da una
    #    estimación más estable de "hacia dónde" mover los pesos que
    #    actualizar tras cada ventana suelta.
    grads_acc, loss = forward_backward(xb, yb)

    # 5) RECORTE DEL GRADIENTE: se calcula la norma global del
    #    gradiente promedio (todas las matrices a la vez) y, si supera
    #    CLIP, se reescala todo el gradiente por el factor CLIP / norma.
    norma = np.sqrt(sum((g ** 2).sum() for g in grads_acc.values()))
    if norma > CLIP:
        for k in grads_acc:
            grads_acc[k] *= CLIP / norma

    # 6) ACTUALIZACIÓN DE ADAM: se aplica, a cada matriz de parámetros
    #    por separado, la fórmula completa de Adam.
    t_adam += 1
    for k in params:
        g = grads_acc[k]
        # Actualizar las medias móviles del gradiente (m) y de su
        # cuadrado (v) para este parámetro:
        m_adam[k] = beta1 * m_adam[k] + (1 - beta1) * g
        v_adam[k] = beta2 * v_adam[k] + (1 - beta2) * g ** 2
        # Corrección de sesgo: en los primeros pasos (t_adam pequeño),
        # m_adam y v_adam están artificialmente cerca de 0 porque
        # empezaron en 0; dividir entre (1 - beta^t) lo compensa.
        m_h = m_adam[k] / (1 - beta1 ** t_adam)
        v_h = v_adam[k] / (1 - beta2 ** t_adam)
        # Actualización final del peso: se mueve en la dirección
        # contraria al gradiente (para REDUCIR la pérdida), con un
        # tamaño de paso que combina la tasa de aprendizaje actual
        # (lr_actual) y la adaptación por parámetro que aporta Adam.
        params[k] -= lr_actual * m_h / (np.sqrt(v_h) + eps_adam)

    # 7) Cada 1000 pasos (y en el último), imprimir el progreso:
    #    la pérdida actual (y
    #    su equivalente en "bits por carácter", una forma alternativa
    #    e intuitiva de leer la misma pérdida: cuántos bits de
    #    información hacen falta, en promedio, para "corregir" al
    #    modelo en cada carácter — cuanto más bajo, mejor predice).
    if paso % 1000 == 0 or paso == PASOS - 1:
        print(f'Paso {paso:5d}: pérdida = {loss:.4f}   lr = {lr_actual:.5f}   '
              f'(bits/char = {loss/np.log(2):.2f})')


# =============================================================================
# ESTRATEGIAS DE MUESTREO: ¿CÓMO ELEGIR EL SIGUIENTE CARÁCTER?
# =============================================================================
# El modelo, para cada posición, produce una distribución de
# probabilidad sobre los 46 caracteres posibles (gracias a softmax).
# Pero de esa distribución hay que ELEGIR uno solo para continuar el
# texto. Existen varias estrategias, cada una con sus ventajas:

def greedy(probs):
    """
    Estrategia "voraz": elegir SIEMPRE el carácter de mayor
    probabilidad, sin ninguna aleatoriedad. `np.argmax` devuelve el
    índice del valor más alto del array.

    Ventaja: el resultado es siempre el mismo (determinista) dado el
    mismo modelo y el mismo texto de partida.
    Inconveniente: el modelo puede quedar "atrapado" repitiendo el
    mismo patrón una y otra vez, porque si un carácter es el más
    probable en un contexto, seguirá siéndolo al volver a aparecer
    ese mismo contexto — produce bucles como "el modelo el modelo
    el modelo...".

    (Esta función se deja como referencia / para experimentar; la
    generación por defecto de este programa usa temperatura + top-p,
    ver `generar()` más abajo.)
    """
    return np.argmax(probs)


def top_k_filtrar(probs, k):
    """
    "Top-k sampling": se queda solo con los `k` caracteres de mayor
    probabilidad, pone a CERO la probabilidad de todos los demás, y
    renormaliza (para que las probabilidades restantes vuelvan a
    sumar 1). Después, se muestrea al azar entre esos k candidatos
    según sus probabilidades relativas.

    Con k=1 equivale a `greedy` (solo puede elegir el más probable).
    Con k grande, se acerca al muestreo "puro" (casi sin filtrar).

    Limitación: k es un número FIJO, sin importar si la distribución
    de probabilidad está muy concentrada (pocos candidatos buenos) o
    muy repartida (muchos candidatos razonables) — de ahí que top-p
    (más abajo) suela funcionar mejor en la práctica.
    """
    indices_ord = np.argsort(probs)[::-1]      # índices ordenados de mayor a menor prob.
    nueva = np.zeros_like(probs)
    nueva[indices_ord[:k]] = probs[indices_ord[:k]]   # conservar solo los k mejores
    return nueva / nueva.sum()                  # renormalizar para que sumen 1


def top_p_filtrar(probs, p):
    """
    "Top-p sampling" o "nucleus sampling": en vez de fijar CUÁNTOS
    caracteres conservar (como top-k), se fija CUÁNTA probabilidad
    acumulada se quiere cubrir. Se ordenan los caracteres de mayor a
    menor probabilidad y se van sumando uno a uno hasta alcanzar
    (como mínimo) `p` de probabilidad acumulada — esos son los
    únicos candidatos que se conservan; el resto se descarta.

    Ventaja sobre top-k: el número de candidatos se ADAPTA a la
    forma de la distribución en cada paso. Si el modelo está muy
    "seguro" (una probabilidad domina claramente), bastarán 1 o 2
    caracteres para cubrir el p%. Si el modelo está más "indeciso"
    (probabilidades repartidas), se conservarán más candidatos.
    Esto es justo lo que se usa en `generar()` para evitar elegir
    caracteres de probabilidad residual ("ruido") que producirían
    texto sin sentido.
    """
    indices_ord = np.argsort(probs)[::-1]        # de mayor a menor probabilidad
    probs_ord   = probs[indices_ord]
    suma_acum   = np.cumsum(probs_ord)           # suma acumulada: 0.4, 0.7, 0.85, ...
    # Buscar cuántos candidatos hacen falta para llegar (o superar) a p:
    n_tokens = int(np.searchsorted(suma_acum, p)) + 1
    nueva = np.zeros_like(probs)
    nueva[indices_ord[:n_tokens]] = probs[indices_ord[:n_tokens]]
    return nueva / nueva.sum()                   # renormalizar


# =============================================================================
# GENERACIÓN DE TEXTO: usar el modelo ya entrenado para "escribir"
# =============================================================================
def generar(semilla, n_chars=120, temperatura=0.8, top_p=0.9):
    """
    Genera texto nuevo, carácter a carácter, de forma AUTORREGRESIVA:
    en cada paso se predice UN carácter, se añade al final del texto
    generado hasta ahora, y ese texto (ya un carácter más largo) se
    usa como entrada para predecir el SIGUIENTE carácter — y así
    sucesivamente hasta completar `n_chars` caracteres nuevos.

    Parámetros
    ----------
    semilla     : texto inicial (string) que "arranca" la generación.
    n_chars     : cuántos caracteres nuevos generar.
    temperatura : controla la aleatoriedad de la elección.
                  - temperatura = 1.0 → probabilidades tal cual las
                    predice el modelo, sin modificar.
                  - temperatura < 1.0 → "afila" la distribución
                    (favorece aún más a los caracteres ya probables):
                    salida más determinista y repetitiva.
                  - temperatura > 1.0 → "aplana" la distribución
                    (da más oportunidad a caracteres poco probables):
                    salida más aleatoria y creativa, con riesgo de
                    volverse incoherente si se sube demasiado.
    top_p       : parámetro del nucleus sampling (ver
                  `top_p_filtrar` arriba). Con top_p=1.0 se desactiva
                  el filtrado y se muestrea de la distribución
                  completa (tras aplicar la temperatura).
    """
    # Convertir la semilla de texto a una lista de índices enteros,
    # usando como máximo los últimos CTX caracteres (el modelo no
    # puede usar un contexto más largo que su ventana CTX).
    # `C2I.get(c, 0)`: si algún carácter de la semilla no estuviera
    # en el vocabulario, se usa el índice 0 como valor por defecto
    # en vez de fallar con un error.
    tokens = [C2I.get(c, 0) for c in semilla[-CTX:]]
    resultado = semilla   # aquí se irá acumulando el texto completo generado

    for _ in range(n_chars):
        # Tomar los últimos CTX tokens como contexto actual.
        ctx = tokens[-CTX:]
        # NO se rellena si el contexto aún es más corto que CTX (al
        # principio de la generación, con semillas cortas): forward()
        # acepta cualquier longitud real gracias a la máscara
        # recortada [:T, :T]. Rellenar con un token artificial
        # desplazaría la posición relativa de los caracteres reales
        # respecto a las ventanas vistas en el entrenamiento.

        # Forward del modelo sobre el contexto actual. forward()
        # trabaja con lotes, así que el contexto se mete en un lote de
        # UNA secuencia: np.array([ctx]), de forma (1, T). Devuelve los
        # logits (1, T, V) y el cache (que aquí no se usa); solo
        # interesa la ÚLTIMA posición, la predicción de "qué viene
        # después del último carácter del contexto": [0][0, -1].
        logits = forward(np.array([ctx]))[0][0, -1]

        # Aplicar la temperatura ANTES del softmax: dividir los
        # logits por `temperatura` cambia cuánto se "exageran" las
        # diferencias entre ellos una vez calculado el softmax.
        probs  = softmax(logits / temperatura)

        # Aplicar nucleus sampling (top-p), salvo que top_p=1.0 (en
        # cuyo caso no se filtra nada y se usa la distribución
        # completa tal cual).
        if top_p < 1.0:
            probs = top_p_filtrar(probs, top_p)

        # Elegir el siguiente carácter MUESTREANDO al azar según las
        # probabilidades finales (no simplemente el más probable):
        # así, ejecuciones distintas pueden generar textos distintos,
        # aunque partan de la misma semilla.
        siguiente = int(np.random.choice(V, p=probs))

        # Añadir el nuevo token a la lista de contexto (para que
        # influya en la predicción del SIGUIENTE carácter) y su
        # versión en texto al resultado final.
        tokens.append(siguiente)
        resultado += I2C[siguiente]

    return resultado


# -----------------------------------------------------------------------------
# DEMOSTRACIÓN FINAL: generar un par de ejemplos de texto con el modelo
# ya entrenado, a partir de dos semillas distintas.
# -----------------------------------------------------------------------------
print('\n=== TEXTO GENERADO POR EL NANO-LLM ===')
print(generar('la inteligencia', n_chars=120))
print()
print(generar('los modelos de', n_chars=120))

# -----------------------------------------------------------------------------
# GUARDAR LOS PESOS ENTRENADOS
# -----------------------------------------------------------------------------
# np.savez guarda todas las matrices de `params` en un único archivo.
# Los Capítulos 11, 14, 15 (Programa RZ-1) y 18 lo cargan con
#   params = dict(np.load('nano_llm_pesos.npz'))
# para usar el modelo ya entrenado sin repetir el entrenamiento.
np.savez('nano_llm_pesos.npz', **params)
print("\nPesos guardados en nano_llm_pesos.npz")
