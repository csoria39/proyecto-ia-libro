@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Desinstalador seguro - De la Neurona al LLM
rem Uso: desinstalar_windows.bat           (simulacion: solo muestra el plan)
rem      desinstalar_windows.bat /EXECUTE  (desinstala, pidiendo confirmacion)

set "ENTORNO=%USERPROFILE%\entorno_ia"
set "LOG=%USERPROFILE%\desinstalar_entorno_ia.log"
set "MODE=SIMULACION"

rem Python que instalo el script del libro: 3.12 (versiones nuevas) o 3.11 (versiones anteriores)
set "PYV=3.12"
winget list --id Python.Python.3.12 --exact >nul 2>&1
if errorlevel 1 set "PYV=3.11"

if /I "%~1"=="/EXECUTE" set "MODE=EJECUCION"

cls
echo ================================================================
echo   DESINSTALADOR - De la Neurona al LLM
echo ================================================================
echo.
echo Modo: %MODE%
echo.
echo Este script usa comandos compatibles con CMD de Windows.
echo No elimina archivos personales ni limpia carpetas globales.
echo.

echo [INVENTARIO]
if exist "%ENTORNO%\" (
    echo [DETECTADO] Entorno virtual: "%ENTORNO%"
) else (
    echo [NO ENCONTRADO] Entorno virtual: "%ENTORNO%"
)

where python >nul 2>&1
if !errorlevel! equ 0 (
    echo [AVISO] Se detecta Python en PATH:
    where python
) else (
    echo [INFO] No se detecta Python mediante PATH.
)

where code >nul 2>&1
if !errorlevel! equ 0 (
    echo [AVISO] Se detecta VS Code en PATH:
    where code
) else (
    echo [INFO] No se detecta VS Code mediante PATH.
)

echo.
echo [PLAN]
echo 1. Eliminar el entorno virtual "%ENTORNO%" si se confirma.
echo 2. Preguntar por separado si se desea desinstalar Python %PYV%.
echo 3. Preguntar por separado si se desea desinstalar VS Code.
echo 4. No borrar configuraciones ni extensiones de VS Code automaticamente.
echo.
echo IMPORTANTE: el instalador no guardo un inventario del estado previo.
echo Python o VS Code podrian haberse instalado antes del proyecto.
echo.

if /I not "%MODE%"=="EJECUCION" (
    echo SIMULACION: no se ha eliminado ni desinstalado nada.
    echo Para ejecutar tras revisar este plan, utiliza:
    echo   %~nx0 /EXECUTE
    echo.
    pause
    exit /b 0
)

echo MODO EJECUCION: se solicitaran confirmaciones antes de cada cambio.
set /p "CONFIRMAR=Escribe ELIMINAR para continuar: "
if /I not "!CONFIRMAR!"=="ELIMINAR" (
    echo Operacion cancelada. No se han realizado cambios.
    pause
    exit /b 0
)

> "%LOG%" echo Registro de desinstalacion - %DATE% %TIME%
>>"%LOG%" echo Usuario: %USERNAME%
>>"%LOG%" echo Entorno objetivo: %ENTORNO%
>>"%LOG%" echo.

echo.
echo [PASO 1] Entorno virtual
if exist "%ENTORNO%\" (
    echo Se propone eliminar exclusivamente:
    echo "%ENTORNO%"
    set /p "BORRAR_ENTORNO=Escribe SI para eliminar esa carpeta: "
    if /I "!BORRAR_ENTORNO!"=="SI" (
        rmdir /s /q "%ENTORNO%"
        if exist "%ENTORNO%\" (
            echo [ERROR] No se pudo eliminar completamente el entorno.
            >>"%LOG%" echo ERROR: no se pudo eliminar el entorno virtual.
        ) else (
            echo [OK] Entorno virtual eliminado.
            >>"%LOG%" echo OK: entorno virtual eliminado.
        )
    ) else (
        echo [CONSERVADO] Entorno virtual no eliminado.
        >>"%LOG%" echo Conservado: entorno virtual.
    )
) else (
    echo [INFO] El entorno virtual no existe.
    >>"%LOG%" echo INFO: entorno virtual no encontrado.
)

echo.
echo [PASO 2] Python !PYV!
echo Solo continua si estas seguro de que deseas quitar Python !PYV!.
set /p "QUITAR_PYTHON=Intentar desinstalar Python.Python.!PYV! con winget? (SI/NO): "
if /I "!QUITAR_PYTHON!"=="SI" (
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        winget uninstall --id Python.Python.!PYV! --exact
        if !errorlevel! equ 0 (
            echo [OK] Winget finalizo la solicitud de desinstalacion.
            >>"%LOG%" echo Solicitud de desinstalacion de Python !PYV! completada por winget.
        ) else (
            echo [AVISO] Winget no pudo desinstalar Python !PYV! o no lo encontro.
            >>"%LOG%" echo AVISO: winget no desinstalo Python !PYV!.
        )
    ) else (
        echo [AVISO] winget no esta disponible.
        >>"%LOG%" echo AVISO: winget no disponible para Python.
    )
) else (
    echo [CONSERVADO] No se ha solicitado desinstalar Python.
    >>"%LOG%" echo Conservado: Python !PYV! por decision del usuario.
)

echo.
echo [PASO 3] Visual Studio Code
echo Comprueba que no lo necesitas para otros proyectos.
set /p "QUITAR_VSCODE=Intentar desinstalar Microsoft.VisualStudioCode con winget? (SI/NO): "
if /I "!QUITAR_VSCODE!"=="SI" (
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        winget uninstall --id Microsoft.VisualStudioCode --exact
        if !errorlevel! equ 0 (
            echo [OK] Winget finalizo la solicitud de desinstalacion.
            >>"%LOG%" echo Solicitud de desinstalacion de VS Code completada por winget.
        ) else (
            echo [AVISO] Winget no pudo desinstalar VS Code o no lo encontro.
            >>"%LOG%" echo AVISO: winget no desinstalo VS Code.
        )
    ) else (
        echo [AVISO] winget no esta disponible.
        >>"%LOG%" echo AVISO: winget no disponible para VS Code.
    )
) else (
    echo [CONSERVADO] No se ha solicitado desinstalar VS Code.
    >>"%LOG%" echo Conservado: VS Code por decision del usuario.
)

echo.
echo [PASO 4] Extensiones y configuracion de VS Code
echo No se eliminan automaticamente para no borrar configuraciones previas.
>>"%LOG%" echo No se eliminaron automaticamente extensiones ni datos de VS Code.

echo.
echo ================================================================
echo PROCESO FINALIZADO
echo ================================================================
echo Registro: "%LOG%"
echo Revisa los resultados y verifica manualmente las aplicaciones.
echo El instalador no guardo el estado anterior; no se puede garantizar
echo una restauracion exacta del sistema.
echo.
pause
exit /b 0
