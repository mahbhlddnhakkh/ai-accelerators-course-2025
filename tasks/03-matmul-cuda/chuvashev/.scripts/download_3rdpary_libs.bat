set CUTLASS_URL=https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.2.1.zip
set CUTLASS_ZIP=cutlass-4.2.1.zip
set CUTLASS_DIR=cutlass-4.2.1

echo Download library from %CUTLASS_URL%...
powershell -Command "(New-Object Net.WebClient).DownloadFile('%CUTLASS_URL%', '%CUTLASS_ZIP%')"

echo Extracting archive %CUTLASS_ZIP% in %CUTLASS_DIR%...
powershell -Command "Expand-Archive -Path '%CUTLASS_ZIP%' -DestinationPath '../3rdparty/%CUTLASS_DIR%'"

pause
