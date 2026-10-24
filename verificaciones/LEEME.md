# Verificaciones

Comprobaciones que respaldan las cifras del libro. No se imprimen en él.

- `verificar_gradientes_float64.py`: el backward del Programa 8, aplicado a los pesos de GPT-2, frente al de PyTorch (autograd), en float64. Salida en `salida_verificar_gradientes_float64.txt`.
- `verificar_8bis_contra_transformers.py`: el forward y el tokenizador del Programa 8 bis frente a `GPT2LMHeadModel` y `GPT2TokenizerFast` de Hugging Face. Salida en `salida_verificar_8bis_contra_transformers.txt`.
- `verificar_gradientes_RZ.py`: el backward de los Programas RZ-2, RZ-3 y RZ-4 frente a diferencias finitas en todos sus tensores. Se usa así:

  ```bash
  python verificar_gradientes_RZ.py programa_RZ2_suma.py '# -- Examen' '(X_TR[:6], Y_TR[:6], np.tile(PESO, (6, 1)))'
  EPS=1e-4 SIN_SUELO=1 python verificar_gradientes_RZ.py programa_RZ3_deduce_letras.py '# -- Generación greedy' 'lote(np.random.RandomState(3))'
  EPS=1e-4 python verificar_gradientes_RZ.py programa_RZ4_deduce_frases.py '# -- Generación greedy' 'lote(np.random.RandomState(3))'
  ```

  Errores relativos máximos: 2,4·10⁻⁸ (RZ-2), 4,9·10⁻⁹ (RZ-3) y 1,1·10⁻⁸ (RZ-4).

  `SIN_SUELO=1` quita el `+ 1e-10` de `log(p + 1e-10)`, y solo en la copia que se verifica. Con la inicialización de RZ-3 (`EMB_STD = 0.5`), algunas probabilidades de partida quedan por debajo de 10⁻¹⁰. Ahí la derivada de la pérdida con suelo deja de ser la de `log(p)`, que es la que calcula el backward.

  Se ejecuta desde la carpeta `programas/`.

Los cuatro scripts necesitan `pip install torch transformers`, y los dos primeros también la carpeta `gpt2/` con los pesos (ver el README principal).
