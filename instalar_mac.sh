#!/bin/bash
# ================================================================
#   INSTALACION DEL ENTORNO DE IA — macOS
#   Libro: De la Neurona al LLM (Carlos Soria Martinez)
#   Repositorio: https://github.com/csoria39/proyecto-ia-libro
# ================================================================
# Compatible con macOS 12 Monterey o posterior, en Mac con chip Apple
# (M1, M2, M3, M4...) o Intel.
#
# Uso:   chmod +x instalar_mac.sh
#        ./instalar_mac.sh            (SIN sudo delante)

set -e

ENTORNO="$HOME/entorno_ia"
ENTORNO_PT="$HOME/entorno_ia_pytorch"      # solo en Mac Intel (ver PASO 6)
PYV="3.12"                                  # version de Python que instala Homebrew
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
    echo "    El script necesita internet para descargar Homebrew, Python y las librerias."
    echo "    Comprueba la conexion (WiFi, cable, VPN). En redes con proxy (instituto,"
    echo "    empresa) puede que necesites ayuda del servicio tecnico."
    read -r -p "Continuar de todas formas? (probablemente fallara) [s/N]: " R
    [ "$R" = "s" ] || [ "$R" = "S" ] || { echo "Instalacion cancelada."; exit 1; }
fi

ARCH=$(uname -m)
clear
echo "================================================================"
echo "  INSTALACION DEL ENTORNO DE IA — macOS"
echo "  Libro: De la Neurona al LLM"
echo "================================================================"
echo ""
echo "Este script instala, en orden:"
echo "  - Homebrew       (gestor de paquetes para Mac)"
echo "  - Python $PYV"
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

# ================================================================
paso "PASO 1 — Homebrew"
# ================================================================
if [ -x /opt/homebrew/bin/brew ]; then eval "$(/opt/homebrew/bin/brew shellenv)"; fi
if [ -x /usr/local/bin/brew ] && ! command -v brew &> /dev/null; then eval "$(/usr/local/bin/brew shellenv)"; fi
if command -v brew &> /dev/null; then
    ok "Homebrew ya esta instalado: $(brew --version | head -1)"
    brew update --quiet || aviso "No se pudo actualizar Homebrew (se continua)."
else
    echo "Instalando Homebrew (pedira tu contrasena de usuario del Mac)..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ "$ARCH" = "arm64" ]; then BREW=/opt/homebrew/bin/brew; else BREW=/usr/local/bin/brew; fi
    # Se anade al PATH una sola vez, aunque se ejecute el script varias veces
    grep -q "brew shellenv" ~/.zprofile 2>/dev/null || echo "eval \"\$($BREW shellenv)\"" >> ~/.zprofile
    eval "$($BREW shellenv)"
    ok "Homebrew instalado y configurado."
fi

# ================================================================
paso "PASO 2 — Python $PYV"
# ================================================================
# El python3 que trae macOS (3.9) es demasiado antiguo para el libro.
if brew list "python@$PYV" &> /dev/null; then
    ok "Python $PYV ya estaba instalado."
else
    brew install "python@$PYV"
fi
PY="$(brew --prefix "python@$PYV")/bin/python$PYV"
[ -x "$PY" ] || error "No se encuentra $PY tras instalar Python con Homebrew."
ok "Python: $("$PY" --version)  ($PY)"

# ================================================================
paso "PASO 3 — Visual Studio Code"
# ================================================================
if command -v code &> /dev/null || [ -d "/Applications/Visual Studio Code.app" ]; then
    ok "VS Code ya esta instalado."
elif brew install --cask visual-studio-code; then
    ok "Visual Studio Code instalado."
else
    aviso "No se pudo instalar VS Code. Descargalo desde https://code.visualstudio.com"
fi
# El comando 'code' (para instalar la extension) viene dentro de la aplicacion
CODE="$(command -v code || true)"
[ -n "$CODE" ] || CODE="/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"

# ================================================================
paso "PASO 4 — Entorno virtual ~/entorno_ia"
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
paso "PASO 5 — NumPy 2, Matplotlib, Transformers y Requests"
# ================================================================
pip install "numpy>=2.3" "matplotlib>=3.8" --quiet
ok "NumPy $(python -c 'import numpy; print(numpy.__version__)') y Matplotlib instalados."
if pip install "transformers>=4.40" "requests>=2.31" --quiet; then
    ok "Transformers y Requests instalados (Capitulo 20)."
else
    aviso "No se pudieron instalar Transformers/Requests (solo para el Capitulo 20). Mas tarde: pip install transformers requests"
fi

# ================================================================
paso "PASO 6 — PyTorch (Apendice D)"
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
paso "PASO 7 — Extension de Python en VS Code"
# ================================================================
if [ -x "$CODE" ] || command -v code &> /dev/null; then
    "$CODE" --install-extension ms-python.python --force > /dev/null 2>&1 && ok "Extension de Python instalada." \
        || aviso "Instala la extension de Python desde VS Code (icono de extensiones)."
else
    aviso "VS Code no esta instalado: omitido."
fi

# ================================================================
paso "PASO 8 — VERIFICACION FINAL"
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
