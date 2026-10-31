@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Instalacion del entorno IA - De la Neurona al LLM

rem ================================================================
rem   INSTALACION DEL ENTORNO DE IA - WINDOWS 10 / 11
rem   Libro: De la Neurona al LLM (Carlos Soria Martinez)
rem   Repositorio: https://github.com/csoria39/proyecto-ia-libro
rem ================================================================

set "ENTORNO=%USERPROFILE%\entorno_ia"
set "VPY=%ENTORNO%\Scripts\python.exe"

cls
echo.
echo  ================================================================
echo    INSTALACION DEL ENTORNO - De la Neurona al LLM
echo  ================================================================
echo.
echo  Instala: Python 3.13 (si no tienes 3.11 o posterior), NumPy 2,
echo           Matplotlib, PyTorch CPU, Transformers, Requests
echo           y Visual Studio Code.
echo.
echo  Todo lo de Python queda en %ENTORNO%
echo  Tiempo estimado: 15-25 minutos. Necesitas internet.
echo.
pause

rem ================================================================
rem  PASO 1 - winget
rem ================================================================
echo.
echo  [1/7] Verificando winget...
winget --version >nul 2>&1
if !errorlevel! neq 0 (
    echo  [ERROR] winget no disponible.
    echo          Instala "Instalador de aplicacion" desde Microsoft Store
    echo          o actualiza Windows desde Windows Update y reintenta.
    pause & exit /b 1
)
echo  [OK] winget disponible.

rem ================================================================
rem  PASO 2 - Python 3.11 o posterior
rem ================================================================
echo.
echo  [2/7] Verificando Python...
call :find_python
if "!PYTHON_EXE!"=="" (
    echo  No hay Python 3.11 o posterior. Instalando Python 3.13...
    winget install --id Python.Python.3.13 --exact --silent --accept-package-agreements --accept-source-agreements
    call :refresh_path
    call :find_python
)
if "!PYTHON_EXE!"=="" (
    echo  [ERROR] No se ha podido instalar o localizar Python 3.11 o posterior.
    echo          Cierra esta ventana, reinicia el PC y vuelve a ejecutar el script,
    echo          o instala Python desde https://python.org marcando "Add python.exe to PATH".
    pause & exit /b 1
)
echo  [OK] Python: !PYTHON_EXE!
"!PYTHON_EXE!" --version

rem ================================================================
rem  PASO 3 - Visual Studio Code
rem ================================================================
echo.
echo  [3/7] Verificando Visual Studio Code...
rem Sin --accept-source-agreements, la primera vez winget pregunta por los terminos
rem y, con la salida oculta, el script pareceria colgado.
winget list --id Microsoft.VisualStudioCode --exact --accept-source-agreements >nul 2>&1
if !errorlevel! neq 0 (
    echo  Instalando Visual Studio Code...
    winget install --id Microsoft.VisualStudioCode --exact --silent --accept-package-agreements --accept-source-agreements
    if !errorlevel! equ 0 (
        echo  [OK] VS Code instalado.
    ) else (
        echo  [AVISO] No se pudo instalar VS Code automaticamente.
        echo          Descargalo desde https://code.visualstudio.com
    )
) else (
    echo  [OK] VS Code ya estaba instalado.
)

rem ================================================================
rem  PASO 4 - Entorno virtual
rem ================================================================
echo.
echo  [4/7] Creando el entorno virtual...
if exist "!VPY!" (
    "!VPY!" -c "" >nul 2>&1
    if !errorlevel! equ 0 (
        echo  [OK] El entorno virtual ya existe en !ENTORNO!
    ) else (
        echo  [AVISO] Entorno incompleto o roto en !ENTORNO!: se vuelve a crear.
        rmdir /s /q "!ENTORNO!"
    )
)
if not exist "!VPY!" (
    "!PYTHON_EXE!" -m venv "!ENTORNO!"
    if not exist "!VPY!" (
        echo  [ERROR] No se pudo crear el entorno virtual.
        pause & exit /b 1
    )
    echo  [OK] Entorno virtual creado en !ENTORNO!
)
rem A partir de aqui se usa siempre el Python del entorno, por su ruta completa:
rem asi no depende de que el entorno siga "activado" ni del PATH.
"!VPY!" -m pip install --upgrade pip --quiet
echo  [OK] pip actualizado dentro del entorno.

rem ================================================================
rem  PASO 5 - NumPy 2 y Matplotlib
rem ================================================================
echo.
echo  [5/7] Instalando NumPy y Matplotlib...
"!VPY!" -m pip install "numpy>=2.3" "matplotlib>=3.8" --quiet
if !errorlevel! neq 0 (
    echo  [ERROR] No se pudieron instalar NumPy o Matplotlib.
    pause & exit /b 1
)
echo  [OK] NumPy y Matplotlib instalados.

rem ================================================================
rem  PASO 6 - PyTorch CPU
rem ================================================================
echo.
echo  [6/7] Instalando PyTorch CPU (unos 200 MB, puede tardar)...
set "TORCH_OK=0"
for /l %%I in (1,1,2) do (
    if "!TORCH_OK!"=="0" (
        "!VPY!" -m pip install "torch>=2.1" "torchvision>=0.16" --index-url https://download.pytorch.org/whl/cpu --quiet
        if !errorlevel! equ 0 set "TORCH_OK=1"
    )
)
if "!TORCH_OK!"=="1" (
    echo  [OK] PyTorch CPU instalado.
) else (
    echo  [AVISO] PyTorch no instalado ^(solo hace falta para el Apendice D^). Mas tarde:
    echo          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

rem ================================================================
rem  PASO 7 - Transformers y Requests
rem ================================================================
echo.
echo  [7/7] Instalando Transformers y Requests (Capitulo 20)...
"!VPY!" -m pip install "transformers>=4.40" "requests>=2.31" --quiet
if !errorlevel! neq 0 (
    echo  [AVISO] No se pudieron instalar. Mas tarde: pip install transformers requests
) else (
    echo  [OK] Transformers y Requests instalados.
)

rem ================================================================
rem  EXTENSION DE PYTHON EN VS CODE
rem ================================================================
echo.
echo  Instalando la extension de Python en VS Code...
set "CODE_CMD="
for %%C in (
    "%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd"
    "%ProgramFiles%\Microsoft VS Code\bin\code.cmd"
) do if exist "%%~C" if "!CODE_CMD!"=="" set "CODE_CMD=%%~C"
if "!CODE_CMD!"=="" (
    echo  [AVISO] Instala la extension de Python desde VS Code ^(icono de extensiones^).
) else (
    call "!CODE_CMD!" --install-extension ms-python.python --force >nul 2>&1
    if !errorlevel! equ 0 (
        echo  [OK] Extension de Python instalada.
    ) else (
        echo  [AVISO] Instala la extension de Python desde VS Code ^(icono de extensiones^).
    )
)

rem ================================================================
rem  VERIFICACION FINAL (con el Python del entorno)
rem ================================================================
echo.
echo  ================================================================
echo    VERIFICACION FINAL
echo  ================================================================
echo.
"!VPY!" -c "import sys; print('  [OK] Python      ', sys.version.split()[0])"
"!VPY!" -c "import numpy as np; assert tuple(int(x) for x in np.__version__.split('.')[:2]) >= (2, 3); print('  [OK] NumPy       ', np.__version__)"
"!VPY!" -c "import matplotlib as m; print('  [OK] Matplotlib  ', m.__version__)"
"!VPY!" -c "import torch; print('  [OK] PyTorch     ', torch.__version__)"
"!VPY!" -c "import transformers; print('  [OK] Transformers', transformers.__version__)"
"!VPY!" -c "import requests; print('  [OK] Requests    ', requests.__version__)"

echo.
echo  ================================================================
echo    INSTALACION COMPLETADA
echo  ================================================================
echo.
echo  Para trabajar con el libro, activa el entorno en cada ventana
echo  de CMD (Simbolo del sistema):
echo.
echo    %%USERPROFILE%%\entorno_ia\Scripts\activate.bat
echo.
echo  En PowerShell el comando es otro:
echo    ^& "$env:USERPROFILE\entorno_ia\Scripts\Activate.ps1"
echo.
echo  En VS Code: Ctrl+Shift+P - "Python: Select Interpreter" y elige
echo    !VPY!
echo.
pause
exit /b 0

rem ================================================================
rem  SUBRUTINAS
rem ================================================================

:find_python
rem Busca un Python 3.11 o posterior: primero el lanzador "py", luego "python"
rem del PATH (sin el acceso directo de Microsoft Store), luego rutas habituales.
rem (la entrada vacia evita que el lanzador se quede esperando si pregunta si instalar una version)
set "PYTHON_EXE="
for %%T in (3.14 3.13 3.12 3.11) do (
    if "!PYTHON_EXE!"=="" (
        for /f "delims=" %%P in ('py -%%T -c "import sys; print(sys.executable)" ^<nul 2^>nul') do set "PYTHON_EXE=%%P"
    )
)
if "!PYTHON_EXE!"=="" (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if "!PYTHON_EXE!"=="" (
            echo %%P | findstr /i "WindowsApps" >nul
            if !errorlevel! neq 0 call :probar "%%P"
        )
    )
)
if "!PYTHON_EXE!"=="" (
    for %%D in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
        "%ProgramFiles%\Python313\python.exe"
        "%ProgramFiles%\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
        "%ProgramFiles%\Python314\python.exe"
    ) do if "!PYTHON_EXE!"=="" if exist "%%~D" call :probar "%%~D"
)
goto :eof

:probar
rem Acepta el interprete %1 solo si es Python 3.11 o posterior
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if !errorlevel! equ 0 set "PYTHON_EXE=%~1"
goto :eof

:refresh_path
rem Relee el PATH del registro tras instalar Python. Los valores pueden contener
rem variables sin expandir (%%SystemRoot%%, %%LOCALAPPDATA%%): "call set" las expande.
set "SYS_PATH=" & set "USR_PATH="
for /f "skip=2 tokens=2,*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
for /f "skip=2 tokens=2,*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USR_PATH=%%B"
if defined SYS_PATH call set "PATH=%SYS_PATH%;%USR_PATH%"
goto :eof
