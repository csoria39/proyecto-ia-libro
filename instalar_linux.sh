#!/bin/bash
# ================================================================
#   INSTALACION DEL ENTORNO DE IA — LINUX
#   Libro: De la Neurona al LLM (Carlos Soria Martinez)
#   Repositorio: https://github.com/csoria39/proyecto-ia-libro
# ================================================================
# Probado en: Ubuntu 22.04 / 24.04, Debian 12 / 13, Linux Mint 21 / 22.
# En Fedora, Arch u openSUSE instala antes Python 3.11+ con su gestor
# de paquetes (ver notas al final); el resto del script funciona igual.
#
# Uso:   chmod +x instalar_linux.sh
#        ./instalar_linux.sh          (SIN sudo delante)

set -e

ENTORNO="$HOME/entorno_ia"
VERDE='\033[0;32m'; ROJO='\033[0;31m'; AMARILLO='\033[1;33m'; AZUL='\033[0;34m'; NC='\033[0m'

ok()    { echo -e "${VERDE}[OK]${NC} $1"; }
error() { echo -e "${ROJO}[ERROR]${NC} $1"; exit 1; }
aviso() { echo -e "${AMARILLO}[AVISO]${NC} $1"; }
paso()  { echo -e "\n${AZUL}================================================================${NC}"
          echo -e "${AZUL}  $1${NC}"
          echo -e "${AZUL}================================================================${NC}\n"; }

trap 'echo; aviso "Instalacion interrumpida. Puedes volver a ejecutar el script: retoma donde se quedo."' INT

if [ "$(uname -s)" != "Linux" ]; then
    error "Este script es para Linux. En macOS usa instalar_mac.sh; en Windows, instalar_windows.bat."
fi
if [ "$EUID" -eq 0 ]; then
    error "No ejecutes el script con sudo: el entorno quedaria en la carpeta de root.
        Lanzalo como usuario normal (./instalar_linux.sh); pedira la contrasena cuando la necesite."
fi

clear 2>/dev/null || true
echo "================================================================"
echo "  INSTALACION DEL ENTORNO DE IA — LINUX"
echo "  Libro: De la Neurona al LLM"
echo "================================================================"
echo ""
echo "Este script instala, en orden:"
echo "  - Python 3.11 o posterior (usa el del sistema si ya lo tiene)"
echo "  - Entorno virtual aislado en ~/entorno_ia"
echo "  - NumPy 2        (calculo numerico — imprescindible, Caps. 1-20)"
echo "  - Matplotlib     (graficas)"
echo "  - PyTorch CPU    (Apendice D)"
echo "  - Transformers y Requests (Hugging Face — Capitulo 20)"
echo "  - Visual Studio Code y su extension de Python"
echo ""
echo "Todo lo de Python queda dentro de ~/entorno_ia: puedes borrar esa"
echo "carpeta en cualquier momento sin afectar al sistema."
echo ""
echo "Tiempo estimado: 10-20 minutos segun la conexion."
echo ""
read -r -p "Pulsa Enter para comenzar..."

# Devuelve 0 si el interprete $1 existe y es Python 3.11 o posterior
python_valido() {
    command -v "$1" &> /dev/null && "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null
}

# ================================================================
paso "PASO 1 — Python 3.11 o posterior"
# ================================================================
PY=""
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq || error "No se pudo actualizar la lista de paquetes (apt). Comprueba la conexion a internet."
    # python3-venv crea entornos virtuales; en Debian/Ubuntu va aparte.
    sudo apt-get install -y -qq python3 python3-venv python3-pip git curl wget gpg
    if python_valido python3; then
        PY=python3
    else
        # Distribuciones antiguas (Python 3.10 o anterior, como Ubuntu 22.04): se intenta instalar el 3.11 junto al del sistema
        echo "El Python del sistema es anterior a 3.11. Instalando Python 3.11..."
        sudo apt-get install -y -qq python3.11 python3.11-venv || \
            error "No se pudo instalar Python 3.11. Actualiza tu distribucion o instala Python 3.11+ a mano."
        PY=python3.11
    fi
else
    aviso "Esta distribucion no usa apt. Se usara el Python que ya tengas instalado."
    for c in python3.14 python3.13 python3.12 python3.11 python3; do
        if python_valido "$c"; then PY="$c"; break; fi
    done
    [ -n "$PY" ] || error "No se encontro Python 3.11 o posterior. Instalalo (ver notas al final del script) y repite."
fi
"$PY" -c 'import venv, ensurepip' 2>/dev/null || \
    error "A este Python le falta el modulo venv. En Debian/Ubuntu: sudo apt install python3-venv"
ok "Python: $("$PY" --version) ($(command -v "$PY"))"

instalar_vscode() {
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /tmp/microsoft.gpg || return 1
    sudo install -D -o root -g root -m 644 /tmp/microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg || return 1
    rm -f /tmp/microsoft.gpg
    echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | \
        sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null || return 1
    sudo apt-get update -qq || return 1
    # Sin esto, el paquete 'code' pregunta en una ventana de texto si anade su propio repositorio
    echo "code code/add-microsoft-repo boolean false" | sudo debconf-set-selections
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq code
}

# ================================================================
paso "PASO 2 — Visual Studio Code"
# ================================================================
if command -v code &> /dev/null; then
    ok "VS Code ya esta instalado: $(code --version | head -1)"
elif command -v apt-get &> /dev/null; then
    echo "Instalando Visual Studio Code desde el repositorio oficial de Microsoft..."
    # Si falla (sin conexion con Microsoft, por ejemplo), se avisa y se sigue: VS Code no es imprescindible
    if instalar_vscode; then
        ok "Visual Studio Code instalado."
    else
        sudo rm -f /etc/apt/sources.list.d/vscode.list
        aviso "No se pudo instalar VS Code. Descargalo desde https://code.visualstudio.com"
    fi
else
    aviso "Instala VS Code desde https://code.visualstudio.com (no es imprescindible)."
fi

# ================================================================
paso "PASO 3 — Entorno virtual ~/entorno_ia"
# ================================================================
if [ -x "$ENTORNO/bin/python" ] && "$ENTORNO/bin/python" -c "" 2>/dev/null; then
    ok "El entorno virtual ya existe en $ENTORNO"
else
    if [ -d "$ENTORNO" ]; then
        aviso "Habia un entorno incompleto en $ENTORNO (de una instalacion interrumpida). Se vuelve a crear."
        rm -rf "$ENTORNO"
    fi
    "$PY" -m venv "$ENTORNO"
    ok "Entorno virtual creado en $ENTORNO"
fi
# shellcheck disable=SC1091
source "$ENTORNO/bin/activate"
python -m pip install --upgrade pip --quiet
ok "Entorno activo y pip actualizado (dentro del entorno, no en el sistema)."

# ================================================================
paso "PASO 4 — NumPy 2 y Matplotlib"
# ================================================================
pip install "numpy>=2.3" "matplotlib>=3.8" --quiet
ok "NumPy $(python -c 'import numpy; print(numpy.__version__)') y Matplotlib $(python -c 'import matplotlib; print(matplotlib.__version__)') instalados."

# ================================================================
paso "PASO 5 — PyTorch CPU (Apendice D)"
# ================================================================
echo "Descargando PyTorch para CPU (unos 200 MB; puede tardar varios minutos)..."
# El indice /whl/cpu evita la version con CUDA, que pesa varios GB.
if pip install "torch>=2.1" "torchvision>=0.16" --index-url https://download.pytorch.org/whl/cpu --quiet; then
    ok "PyTorch $(python -c 'import torch; print(torch.__version__)') instalado."
else
    aviso "No se pudo instalar PyTorch (solo hace falta para el Apendice D). Mas tarde:"
    aviso "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
fi

# ================================================================
paso "PASO 6 — Transformers y Requests (Capitulo 20)"
# ================================================================
if pip install "transformers>=4.40" "requests>=2.31" --quiet; then
    ok "Transformers $(python -c 'import transformers; print(transformers.__version__)') y Requests instalados."
else
    aviso "No se pudieron instalar (solo hacen falta en el Capitulo 20). Mas tarde: pip install transformers requests"
fi

# ================================================================
paso "PASO 7 — Extension de Python en VS Code"
# ================================================================
if command -v code &> /dev/null; then
    code --install-extension ms-python.python --force > /dev/null 2>&1 && ok "Extension de Python instalada." \
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
        print(f"[--] {m:<13}no disponible ({type(e).__name__})")
import numpy as np
assert tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 3), "El libro necesita NumPy 2.3 o posterior"
print("[OK] Operacion NumPy:", np.array([1, 2, 3]) * 2)
FIN

echo ""
echo "================================================================"
echo "  INSTALACION COMPLETADA"
echo "================================================================"
echo ""
echo "Activa el entorno en cada terminal nueva con:"
echo "    source ~/entorno_ia/bin/activate"
echo ""
echo "Para activarlo automaticamente al abrir la terminal:"
echo "    echo 'source ~/entorno_ia/bin/activate' >> ~/.bashrc"
echo ""
echo "En VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter'"
echo "y elige ~/entorno_ia/bin/python"
echo ""
echo "OTRAS DISTRIBUCIONES (instala Python antes y vuelve a lanzar el script):"
echo "  Fedora/RHEL:  sudo dnf install python3 python3-pip"
echo "  Arch Linux:   sudo pacman -S python python-pip"
echo "  openSUSE:     sudo zypper install python311 python311-pip"
echo ""
deactivate
ok "Script completado correctamente."
