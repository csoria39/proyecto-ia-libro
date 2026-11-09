# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Comprueba el entorno antes de ejecutar los programas del libro.

Muestra la versión de Python y de cada librería frente a la del entorno probado
(requirements-probado.txt), la biblioteca BLAS de NumPy, los hilos, la memoria, y
comprueba con SHA-256 los archivos de GPT-2 de la carpeta programas/gpt2/.
No cambia nada: solo informa.

Uso:  python verificaciones/verificar_entorno.py   [carpeta_gpt2]"""
import hashlib, importlib, os, platform, sys

PROBADO = {'numpy': '2.5.4', 'matplotlib': '3.11.2', 'safetensors': '0.8.0', 'tokenizers': '0.23.3',
           'huggingface_hub': '2.4.0', 'transformers': '5.22.1', 'torch': '2.14.1'}
SHA256_GPT2 = {   # openai-community/gpt2 en huggingface.co
    'model.safetensors': '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707',
    'vocab.json':        '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
    'merges.txt':        '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    'config.json':       '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
}

print(f'Python   {platform.python_version()}  (probado: 3.13.16)   {platform.system()} {platform.machine()}')
for nombre, probada in PROBADO.items():
    try:
        v = importlib.import_module(nombre).__version__
        marca = 'igual' if v.split('+')[0] == probada else f'distinta (probada {probada})'
        print(f'  {nombre:<16} {v:<14} {marca}')
    except ImportError:
        print(f'  {nombre:<16} {"no instalada":<14} (probada {probada}; solo la necesitan algunos capítulos)')

try:
    import numpy as np
    cfg = np.show_config(mode='dicts')
    blas = cfg.get('Build Dependencies', {}).get('blas', {})
    print(f'BLAS de NumPy: {blas.get("name", "?")} {blas.get("version", "")}')
except Exception as e:
    print('BLAS de NumPy: no se pudo leer', e)
hilos = os.environ.get('OPENBLAS_NUM_THREADS')
print(f'OPENBLAS_NUM_THREADS = {hilos}' + ('' if hilos == '1' else '   (las salidas del libro se obtuvieron con 1)'))
try:
    paginas, tam = os.sysconf('SC_PHYS_PAGES'), os.sysconf('SC_PAGE_SIZE')
    print(f'Memoria: {paginas * tam / 1024**3:.1f} GiB')
except (ValueError, AttributeError, OSError):
    print('Memoria: no disponible en este sistema')

carpeta = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'programas', 'gpt2')
print(f'\nArchivos de GPT-2 en {os.path.abspath(carpeta)}:')
for archivo, esperado in SHA256_GPT2.items():
    ruta = os.path.join(carpeta, archivo)
    if not os.path.exists(ruta):
        print(f'  {archivo:<18} no está (solo hace falta desde el Capítulo 16)')
        continue
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for bloque in iter(lambda: f.read(1 << 20), b''):
            h.update(bloque)
    print(f'  {archivo:<18} {"correcto" if h.hexdigest() == esperado else "NO COINCIDE: descárgalo de nuevo"}')
