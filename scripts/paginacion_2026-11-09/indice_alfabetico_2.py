"""Índice alfabético, segunda revisión (9-11-2026, tarde): referencias que faltaban o apuntaban a una página débil."""
import sys
doc = sys.argv[1]; s = open(doc).read()
C = [(' 152, 154, 158, 251-252, 256, 265, 282<', ' 58, 152, 154, 251-252, 256, 265, 282<'),   # Forward pass: 6.3 Forward propagation
     (' 110, 114, 119-120, 354<', ' 110, 114, 117, 119-120, 354<'),                              # Gradiente que desaparece: 10.5 LSTM
     (' 288, 291, 296<', ' 288, 291, 296-297<'),                                                 # Evaluación de modelos: 18.9 Guía práctica
     (' 31, 72, 79, 154, 280-281, 303, 340<', ' 31, 72, 79, 154, 280-281, 303, 342<')]          # Función de pérdida: nn.BCEWithLogitsLoss (D)
for a, b in C:
    n = s.count(a)
    if n != 1: raise SystemExit(f'{a!r}: {n} apariciones')
    s = s.replace(a, b)
open(doc, 'w').write(s); print('índice: 4 entradas')
