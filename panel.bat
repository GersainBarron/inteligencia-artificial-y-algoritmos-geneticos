@echo off
rem Abre el panel local de evocarros en http://localhost:8080 (solo esta maquina).
cd /d "%~dp0"
start "" http://localhost:8080/
call conda run --no-capture-output -n evocarros python -m evocarros.servidor.servidor %*
