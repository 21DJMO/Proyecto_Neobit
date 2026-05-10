@echo off
echo ==============================================
echo   Iniciando Proyecto Neobit - Asistente
echo ==============================================

:: Verificar si existe la carpeta del entorno virtual
if not exist "venv\" (
    echo Creando entorno virtual local...
    python -m venv venv
    
    echo Instalando dependencias (esto puede tardar unos minutos la primera vez)...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
)

echo.
echo ==============================================
echo   Asegurate de tener OLLAMA corriendo!
echo   (Abre la aplicacion de Ollama en tu PC)
echo ==============================================
echo.

echo Ejecutando aplicacion...
python src\app_voice_chat.py

echo.
pause
