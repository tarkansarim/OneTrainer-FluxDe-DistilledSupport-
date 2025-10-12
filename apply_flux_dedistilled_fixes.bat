@echo off
echo.
echo ================================================================================
echo     Flux Dedistilled Fixes for OneTrainer
echo ================================================================================
echo.

REM Check if we're in the right directory
if not exist "modules" (
    echo ERROR: 'modules' directory not found!
    echo Please run this script from the OneTrainer root directory.
    echo.
    pause
    exit /b 1
)

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
)

REM Run the Python script
python apply_flux_dedistilled_fixes.py

REM Check if Python script succeeded
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo Script completed successfully!
    echo ================================================================================
) else (
    echo.
    echo ================================================================================
    echo Script encountered errors. Please check the output above.
    echo ================================================================================
)

echo.
pause

