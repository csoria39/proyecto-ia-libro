# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
def costo_consulta_rag(k, tokens_sistema=200, tokens_por_fragmento=150,
                       tokens_pregunta=30, precio_por_1k=0.01):
    total_tokens = tokens_sistema + (k * tokens_por_fragmento) + tokens_pregunta
    costo = total_tokens / 1000 * precio_por_1k
    return total_tokens, costo
 
print('=== COSTE DE TOKENS SEGUN K (fragmentos recuperados) ===\n')
print(f"{'k':>4}  {'Tokens totales':>15}  {'Costo (USD)':>12}")
print('-' * 36)
 
for k in [1, 3, 5, 10, 20]:
    tokens, costo = costo_consulta_rag(k)
    print(f'{k:>4}  {tokens:>15}  ${costo:>11.5f}')
 
print()
print('A 1000 consultas/dia con k=5:')
tokens5, costo5 = costo_consulta_rag(5)
print(f'  Costo diario: ${costo5*1000:.2f}')
print(f'  Costo mensual: ${costo5*1000*30:.2f}')
print()
print('Por eso en produccion se ajusta k cuidadosamente:')
print('mas contexto puede mejorar la respuesta, pero el costo')
print('y la latencia crecen linealmente con k.')
