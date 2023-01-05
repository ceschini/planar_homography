@echo off
title Install Planar Homog
virtualenv .venv
CALL .venv\Scripts\activate.bat
pip install -r requirements.txt
echo Finished installing.
pause
