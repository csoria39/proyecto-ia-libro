#!/bin/bash
# ================================================================
#   INSTALACION DEL ENTORNO DE IA — macOS
#   Libro: De la Neurona al LLM (Carlos Soria Martinez)
#   Repositorio: https://github.com/csoria39/proyecto-ia-libro
# ================================================================
# Compatible con macOS 12 Monterey o posterior, en Mac con chip Apple
# (M1, M2, M3, M4...) o Intel. No necesita Homebrew: si no tienes Python 3.11
# o posterior, instala el de python.org (pide tu contrasena una vez).
#
# Uso:   chmod +x instalar_mac.sh
#        ./instalar_mac.sh            (SIN sudo delante)

set -e

ENTORNO="$HOME/entorno_ia"
ENTORNO_PT="$HOME/entorno_ia_pytorch"      # solo en Mac Intel (ver PASO 5)
# Python que se instala desde python.org si no hay uno valido (instalador universal2).
# En Mac Intel, PyTorch solo llega a la 2.2, que no existe para Python 3.13: alli se usa 3.12.
if [ "$(uname -m)" = "arm64" ]; then PYV="3.13"; PY_PKG="3.13.17"; else PYV="3.12"; PY_PKG="3.12.10"; fi
VERDE='\033[0;32m'; ROJO='\033[0;31m'; AMARILLO='\033[1;33m'; AZUL='\033[0;34m'; NC='\033[0m'

ok()    { echo -e "${VERDE}[OK]${NC} $1"; }
error() { echo -e "${ROJO}[ERROR]${NC} $1"; exit 1; }
aviso() { echo -e "${AMARILLO}[AVISO]${NC} $1"; }
paso()  { echo -e "\n${AZUL}================================================================${NC}"
          echo -e "${AZUL}  $1${NC}"
          echo -e "${AZUL}================================================================${NC}\n"; }

trap 'echo; aviso "Instalacion interrumpida. Puedes volver a ejecutar el script: retoma donde se quedo."' INT

if [ "$(uname -s)" != "Darwin" ]; then
    error "Este script es para macOS. En Linux usa instalar_linux.sh; en Windows, instalar_windows.bat."
fi
if [ "$EUID" -eq 0 ]; then
    error "No ejecutes el script con sudo. Lanzalo como usuario normal (./instalar_mac.sh)."
fi

# Conexion a internet (macOS no trae el comando 'timeout'; se usa curl)
echo "Comprobando conexion a internet..."
if curl -s --head --max-time 8 https://pypi.org > /dev/null; then
    ok "Conexion a internet detectada."
else
    aviso "No se ha podido contactar con pypi.org."
    echo "    El script necesita internet para descargar Python, las librerias y VS Code."
    echo "    Comprueba la conexion (WiFi, cable, VPN). En redes con proxy (instituto,"
    echo "    empresa) puede que necesites ayuda del servicio tecnico."
    read -r -p "Continuar de todas formas? (probablemente fallara) [s/N]: " R
    [ "$R" = "s" ] || [ "$R" = "S" ] || { echo "Instalacion cancelada."; exit 1; }
fi

ARCH=$(uname -m)
clear 2>/dev/null || true
echo "================================================================"
echo "  INSTALACION DEL ENTORNO DE IA — macOS"
echo "  Libro: De la Neurona al LLM"
echo "================================================================"
echo ""
echo "Este script instala, en orden:"
echo "  - Python $PYV de python.org (solo si no tienes Python 3.11 o posterior)"
echo "  - Entorno virtual aislado en ~/entorno_ia"
echo "  - NumPy 2        (calculo numerico — imprescindible, Caps. 1-20)"
echo "  - Matplotlib     (graficas)"
echo "  - PyTorch        (Apendice D)"
echo "  - Transformers y Requests (Hugging Face — Capitulo 20)"
echo "  - Visual Studio Code y su extension de Python"
echo ""
if [ "$ARCH" = "arm64" ]; then
    echo "Chip detectado: Apple Silicon (M1/M2/M3/M4...)"
else
    echo "Chip detectado: Intel x86_64"
fi
echo "Tiempo estimado: 15-25 minutos segun la conexion."
echo ""
read -r -p "Pulsa Enter para comenzar..."

# Devuelve 0 si el interprete $1 existe, es Python 3.11 o posterior y puede crear entornos virtuales
python_valido() {
    [ -x "$1" ] || command -v "$1" &> /dev/null || return 1
    "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null
}

# ================================================================
paso "PASO 1 — Python 3.11 o posterior"
# ================================================================
# El python3 que trae macOS (3.9) es demasiado antiguo para el libro. Se busca uno valido
# (python.org o Homebrew); si no lo hay, se instala el oficial de python.org. No se usa
# Homebrew para instalarlo: en Mac Intel y en macOS 14 o anterior, Homebrew compila Python
# desde el codigo fuente, lo que tarda mucho y puede fallar.
PY=""
if [ "$ARCH" = "arm64" ]; then
    CANDIDATOS="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
                /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
                /opt/homebrew/bin/python3 python3.14 python3.13 python3.12 python3.11 python3"
else
    # Mac Intel: solo 3.11 o 3.12, porque PyTorch 2.2 no tiene version para 3.13
    CANDIDATOS="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
                /usr/local/bin/python3.12 /usr/local/bin/python3.11 python3.12 python3.11"
fi
for c in $CANDIDATOS; do
    if python_valido "$c"; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
    echo "No hay Python 3.11 o posterior. Descargando Python $PY_PKG de python.org (unos 70 MB)..."
    PKG="$(mktemp -d)/python-$PY_PKG-macos11.pkg"
    curl -fL --progress-bar -o "$PKG" "https://www.python.org/ftp/python/$PY_PKG/python-$PY_PKG-macos11.pkg" || \
        error "No se pudo descargar Python. Descargalo desde https://www.python.org/downloads/macos/, instalalo y vuelve a lanzar el script."
    echo "Instalando Python (pedira tu contrasena de usuario del Mac)..."
    sudo installer -pkg "$PKG" -target / || error "No se pudo instalar Python $PY_PKG."
    rm -f "$PKG"
    PY="/Library/Frameworks/Python.framework/Versions/$PYV/bin/python3"
    python_valido "$PY" || error "No se encuentra $PY tras instalar Python."
fi
ok "Python: $("$PY" --version)  ($PY)"

# ================================================================
paso "PASO 2 — Visual Studio Code"
# ================================================================
APP="/Applications/Visual Studio Code.app"
if command -v code &> /dev/null || [ -d "$APP" ]; then
    ok "VS Code ya esta instalado."
else
    echo "Descargando Visual Studio Code (unos 250 MB)..."
    ZIP="$(mktemp -d)/vscode.zip"
    if curl -fL --progress-bar -o "$ZIP" "https://update.code.visualstudio.com/latest/darwin-universal/stable" && \
       { ditto -xk "$ZIP" /Applications 2>/dev/null || sudo ditto -xk "$ZIP" /Applications; }; then
        ok "Visual Studio Code instalado en /Applications."
    else
        aviso "No se pudo instalar VS Code. Descargalo desde https://code.visualstudio.com"
    fi
    rm -f "$ZIP"
fi
# El comando 'code' (para instalar la extension) viene dentro de la aplicacion
CODE="$(command -v code || true)"
[ -n "$CODE" ] || CODE="$APP/Contents/Resources/app/bin/code"

# ================================================================
paso "PASO 3 — Entorno virtual ~/entorno_ia"
# ================================================================
crear_entorno() {   # $1 = carpeta del entorno
    if [ -x "$1/bin/python" ] && "$1/bin/python" -c "" 2>/dev/null; then
        ok "El entorno virtual ya existe en $1"
    else
        if [ -d "$1" ]; then
            aviso "Habia un entorno incompleto en $1 (de una instalacion interrumpida). Se vuelve a crear."
            rm -rf "$1"
        fi
        "$PY" -m venv "$1"
        ok "Entorno virtual creado en $1"
    fi
}
crear_entorno "$ENTORNO"
# shellcheck disable=SC1091
source "$ENTORNO/bin/activate"
python -m pip install --upgrade pip --quiet
ok "Entorno activo y pip actualizado."

# ================================================================
paso "PASO 4 — NumPy 2, Matplotlib, Transformers y Requests"
# ================================================================
pip install "numpy>=2.3" "matplotlib>=3.8" --quiet
ok "NumPy $(python -c 'import numpy; print(numpy.__version__)') y Matplotlib instalados."
if pip install "transformers>=4.40" "requests>=2.31" --quiet; then
    ok "Transformers y Requests instalados (Capitulo 20)."
else
    aviso "No se pudieron instalar Transformers/Requests (solo para el Capitulo 20). Mas tarde: pip install transformers requests"
fi

# ================================================================
paso "PASO 5 — PyTorch (Apendice D)"
# ================================================================
if [ "$ARCH" = "arm64" ]; then
    echo "Instalando PyTorch con aceleracion MPS (chip Apple)..."
    if pip install "torch>=2.1" "torchvision>=0.16" --quiet; then
        ok "PyTorch $(python -c 'import torch; print(torch.__version__)') instalado."
    else
        aviso "No se pudo instalar PyTorch (solo para el Apendice D). Mas tarde: pip install torch torchvision"
    fi
else
    # PyTorch dejo de publicar versiones para Mac Intel en la 2.2, que se compilo con NumPy 1.
    # Para no bajar NumPy en el entorno del libro, el Apendice D usa un segundo entorno.
    echo "Mac Intel: la ultima version de PyTorch para este chip es la 2.2, que necesita NumPy 1."
    echo "Se instala en un entorno aparte, ~/entorno_ia_pytorch, solo para el Apendice D."
    deactivate
    crear_entorno "$ENTORNO_PT"
    if "$ENTORNO_PT/bin/python" -m pip install --upgrade pip --quiet && \
       "$ENTORNO_PT/bin/python" -m pip install "numpy<2" "torch==2.2.*" "torchvision==0.17.*" --quiet; then
        ok "PyTorch $("$ENTORNO_PT/bin/python" -c 'import torch; print(torch.__version__)') instalado en $ENTORNO_PT."
    else
        aviso "No se pudo instalar PyTorch (solo para el Apendice D)."
    fi
    # shellcheck disable=SC1091
    source "$ENTORNO/bin/activate"
fi

# ================================================================
paso "PASO 6 — Extension de Python en VS Code"
# ================================================================
if [ -x "$CODE" ] || command -v code &> /dev/null; then
    "$CODE" --install-extension ms-python.python --force > /dev/null 2>&1 && ok "Extension de Python instalada." \
        || aviso "Instala la extension de Python desde VS Code (icono de extensiones)."
else
    aviso "VS Code no esta instalado: omitido."
fi

# ================================================================
paso "PASO 7 — VERIFICACION FINAL"
# ================================================================
python - <<'FIN'
import sys, importlib
print(f"[OK] Python {sys.version.split()[0]}  ({sys.executable})")
for m in ("numpy", "matplotlib", "torch", "transformers", "requests"):
    try:
        print(f"[OK] {m:<13}{importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"[--] {m:<13}no disponible en este entorno ({type(e).__name__})")
import numpy as np
assert tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 3), "El libro necesita NumPy 2.3 o posterior"
print("[OK] Operacion NumPy:", np.array([1, 2, 3]) * 2)
try:
    import torch
    print("[OK] Aceleracion MPS disponible:", torch.backends.mps.is_available())
except ImportError:
    pass
FIN
if [ "$ARCH" != "arm64" ] && [ -x "$ENTORNO_PT/bin/python" ]; then
    "$ENTORNO_PT/bin/python" -c "import torch; print('[OK] PyTorch (entorno_ia_pytorch)', torch.__version__, torch.tensor([1.0, 2.0]).numpy())" \
        || aviso "PyTorch no disponible en $ENTORNO_PT."
fi

echo ""
echo "================================================================"
echo "  INSTALACION COMPLETADA"
echo "================================================================"
echo ""
echo "Activa el entorno en cada terminal nueva con:"
echo "    source ~/entorno_ia/bin/activate"
if [ "$ARCH" != "arm64" ]; then
    echo ""
    echo "Para los programas PyTorch del Apendice D (Mac Intel):"
    echo "    source ~/entorno_ia_pytorch/bin/activate"
fi
echo ""
echo "Para activarlo automaticamente al abrir la terminal:"
echo "    echo 'source ~/entorno_ia/bin/activate' >> ~/.zshrc"
echo ""
echo "En VS Code: Cmd+Shift+P -> 'Python: Select Interpreter'"
echo "y elige ~/entorno_ia/bin/python"
echo ""
deactivate
ok "Script completado correctamente."
