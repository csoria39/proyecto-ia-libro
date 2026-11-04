# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(7)
 
documentos = [
    'La neurona artificial calcula una suma ponderada y aplica una activacion.',
    'El self-attention conecta directamente cualquier par de tokens.',
    'Adam combina Momentum y RMSprop con correccion de sesgo.',
    'BPE fusiona iterativamente los pares de caracteres mas frecuentes.',
    'El overfitting ocurre cuando el modelo memoriza en vez de generalizar.',
    'Un modelo puede discriminar sin usar la variable protegida directamente.',
]
temas = ['neurona', 'atencion', 'optimizador', 'bpe', 'overfitting', 'etica']
 
# Embeddings simulados: en la practica vendrian de un modelo real
# Para la demo, generamos embeddings correlacionados con el tema
D = 16
embeddings_docs = np.random.randn(len(documentos), D) * 0.5
 
def sim_coseno(u, v):
    return np.dot(u,v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-8)
 
def rag_buscar(pregunta_emb, k=2):
    sims = [sim_coseno(pregunta_emb, d) for d in embeddings_docs]
    idx  = np.argsort(sims)[::-1][:k]
    return [(documentos[i], temas[i], sims[i]) for i in idx]
 
# Simulamos una pregunta 'sobre optimizadores'
pregunta_emb = embeddings_docs[2] + np.random.randn(D)*0.15
 
print('=== SISTEMA RAG MINIMO ===\n')
print('Pregunta simulada: relacionada con optimizadores\n')
resultados = rag_buscar(pregunta_emb, k=2)
for doc, tema, sim in resultados:
    print(f'  [{tema:>12}] sim={sim:.3f}  "{doc}"')
 
print()
print('El LLM recibiria estos 2 fragmentos como contexto adicional')
print('antes de generar su respuesta a la pregunta del usuario.')
