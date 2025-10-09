@echo off
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
cd /d %~dp0
python setup.py build_ext --inplace
pause