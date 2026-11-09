# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np

# =======================================================
# CLASE NEURONA ARTIFICIAL
# Implementa el modelo: salida = sigmoid(X . W + b)
# =======================================================

class Neurona:
    """
    Una neurona artificial con función de activación sigmoid.

    Parámetros aprendibles:
      - pesos: un valor por cada entrada
      - sesgo: un valor adicional (permite desplazar la activación)
    """

    def __init__(self, n_entradas, semilla=42):
        np.random.seed(semilla)
        # Inicializamos con valores pequeños aleatorios
        # Si fueran todos cero, la red nunca aprendería diferencias
        self.pesos = np.random.randn(n_entradas) * 0.3
        self.sesgo = 0.0
        self.n_entradas = n_entradas

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def calcular(self, entradas):
        """
        Realiza el cálculo forward (hacia adelante).
        Retorna la salida y la pre-activación z.
        """
        entradas = np.array(entradas, dtype=float)
        if entradas.shape != (self.n_entradas,):
            raise ValueError(f'Se esperaban {self.n_entradas} entradas')

        # Paso 1: suma ponderada
        z = np.dot(entradas, self.pesos) + self.sesgo
        # Paso 2: función de activación
        salida = self._sigmoid(z)
        return salida, z

    def __repr__(self):
        return (f'Neurona(n={self.n_entradas}, '
                f'pesos={self.pesos.round(4)}, '
                f'sesgo={self.sesgo:.4f})')


# -- Demostración ---------------------------------------
print('=== NUESTRA PRIMERA NEURONA ARTIFICIAL ===\n')

# Creamos una neurona que acepta 3 entradas
neurona = Neurona(n_entradas=3)
print(f'Neurona creada:\n  {neurona}\n')

# La probamos con distintos conjuntos de entradas
casos = [
    ([0.0, 0.0, 0.0], 'Entradas todas a cero'),
    ([1.0, 1.0, 1.0], 'Entradas todas a uno'),
    ([0.8, 0.2, 0.6], 'Entradas mixtas'),
    ([-1.0, 2.0, -0.5], 'Entradas con negativos'),
]

print(f'{"Descripción":<30} {"z (pre-act.)":>14} {"Salida":>10}')
print('-' * 58)
for entradas, desc in casos:
    salida, z = neurona.calcular(entradas)
    print(f'{desc:<30} {z:14.4f} {salida:10.4f}')

# Interpretación de la salida
print('\nInterpretación:')
print('  Salida ≈ 0.5 -> la neurona no tiene preferencia')
print('  Salida -> 1.0 -> la neurona se activa fuertemente')
print('  Salida -> 0.0 -> la neurona se inhibe fuertemente')
