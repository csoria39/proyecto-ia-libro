# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# EJERCICIO 16.4 — Temperatura frente a perplejidad del texto generado
# Se genera con varias temperaturas y se mide, con el propio GPT-2 a
# temperatura 1, cuánto le «sorprende» cada texto generado.
import numpy as np

fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])

semilla = 'The meaning of life is'
n_sem = len(codificar(semilla))
print(f"{'τ':>4s}  {'PPL':>7s}  texto generado")
for tau in [0.3, 0.7, 1.0, 1.5]:
    np.random.seed(0)
    texto = generar(semilla, n_tokens=30, temperatura=tau, top_p=1.0)
    ids = codificar(texto)
    lp = np.log(softmax(forward(np.array([ids]))[0][0, :-1].astype(np.float64)))
    nll = -lp[np.arange(n_sem - 1, len(ids) - 1), ids[n_sem:]]       # solo lo generado
    print(f'{tau:4.1f}  {np.exp(nll.mean()):7.1f}  {texto[len(semilla):].strip()[:90]!r}')
