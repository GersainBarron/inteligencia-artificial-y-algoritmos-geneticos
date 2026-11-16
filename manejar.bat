@echo off
rem Manejo manual con teclado: flechas para manejar, R reinicia, Q sale.
cd /d "%~dp0"
call conda run --no-capture-output -n evocarros python scripts\manejar_teclado.py %*
if errorlevel 1 pause
