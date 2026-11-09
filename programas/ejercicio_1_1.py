# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
v1 = np.array([1.0, 2.0, 3.0, 4.0])
v2 = np.array([4.0, 3.0, 2.0, 1.0])
 
print('Suma:                   ', v1 + v2)
print('Diferencia:             ', v1 - v2)
print('Producto elem. a elem.: ', v1 * v2)
print('Producto punto (dot):   ', np.dot(v1, v2))  # 1*4+2*3+3*2+4*1 = 20
 
# Multiplicación de matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])       # (2, 3)
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])        # (3, 2)
 
C = A @ B                      # (2, 3) @ (3, 2) = (2, 2)
print('\nA (2x3):\n', A)
print('B (3x2):\n', B)
print('A @ B (2x2):\n', C)
print('Regla: (mxn) @ (nxp) = (mxp)')
