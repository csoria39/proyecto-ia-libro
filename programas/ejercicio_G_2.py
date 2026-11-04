# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
def campo_receptivo(capas):
    """
    capas: lista de 'conv3' o 'pool2'
    Retorna el campo receptivo y el historial paso a paso.
    """
    campo, salto = 1, 1   # salto: píxeles de la imagen entre dos posiciones vecinas
    hist  = [('entrada', campo)]
    for capa in capas:
        if capa == 'conv3':
            campo += 2 * salto      # el filtro 3x3 abarca un vecino más a cada lado
            hist.append(('conv 3x3', campo))
        elif capa == 'pool2':
            campo += salto          # la ventana 2x2 abarca una posición más...
            salto *= 2              # ...y a partir de aquí las posiciones van de 2 en 2
            hist.append(('maxpool 2x2', campo))
    return campo, hist

arquitecturas = {
    'Nuestra CNN simple':   ['conv3', 'pool2'],
    'CNN con 2 pools':      ['conv3', 'pool2', 'conv3', 'pool2'],
    'VGG-style (5 conv)':   ['conv3','conv3','pool2','conv3','conv3','pool2','conv3'],
    'Muy profunda (8 conv)':['conv3','conv3','conv3','conv3',
                             'pool2','conv3','conv3','conv3','conv3'],
}

for nombre, arq in arquitecturas.items():
    campo_final, hist = campo_receptivo(arq)
    print(f'  {nombre}:')
    for nombre_capa, c in hist:
        print(f'    {nombre_capa:>15}: campo = {c}x{c}')
    print(f'    -> Una neurona profunda ve {campo_final}x{campo_final} px\n')
