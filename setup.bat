@echo off
echo Installing required packages for Amazon Review Classification App...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python is installed.
echo.

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Error installing packages. Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Installation complete!
echo.
echo To run the app, execute: streamlit run app.py
echo.
pause
