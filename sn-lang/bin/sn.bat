@echo off
rem Run S#N without installing:  bin\sn.bat examples\01_hello.sn
set "DIR=%~dp0.."
set "PYTHONPATH=%DIR%;%PYTHONPATH%"
python -m sn %*
