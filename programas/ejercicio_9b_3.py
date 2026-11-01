# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def softmax(x): e=np.exp(x-x.max()); return e/e.sum()
def top_p_filtrar(p, pp):
    idx=np.argsort(p)[::-1]; ac=np.cumsum(p[idx])
    n=int(np.searchsorted(ac,pp))+1
    m=np.zeros_like(p); m[idx[:n]]=p[idx[:n]]; return m/m.sum()
 
def generar_adaptativo(modelo_forward, c2i, i2c, semilla, n=60,
                        CTX=24, tau_inicial=0.3, tau_resto=0.9,
                        n_tokens_iniciales=10, top_p_resto=0.9):
    '''
    Estrategia adaptativa: temperatura baja al principio (coherencia),
    temperatura mas alta + top-p despues (variedad).
    '''
    V = len(c2i)
    tokens = [c2i.get(c, 0) for c in semilla[-CTX:]]
    resultado = semilla
    
    for i in range(n):
        ctx = tokens[-CTX:]   # SIN relleno: se usa la longitud real
        logits = modelo_forward(ctx)[-1]
        
        # Estrategia adaptativa segun la posicion
        if i < n_tokens_iniciales:
            tau = tau_inicial
            probs = softmax(logits / tau)
        else:
            tau = tau_resto
            probs = softmax(logits / tau)
            probs = top_p_filtrar(probs, top_p_resto)
        
        siguiente = int(np.random.choice(V, p=probs))
        tokens.append(siguiente)
        resultado += i2c[siguiente]
    
    return resultado
 
print('=== ESTRATEGIA DE MUESTREO ADAPTATIVA ===\n')
print('Primeros 10 tokens:  tau=0.3 (determinista, ancla el inicio)')
print('Tokens siguientes:   tau=0.9 + top-p=0.9 (variedad creativa)')
print()
print('Justificacion:')
print('  - El inicio de una respuesta suele tener pocas opciones')
print('    razonables (saludo, estructura gramatical inicial).')
print('  - Una vez establecido el contexto, hay mas margen creativo')
print('    sin perder coherencia.')
print('  - Esta tecnica reduce la varianza de calidad entre generaciones')
print('    manteniendo diversidad donde realmente aporta valor.')
