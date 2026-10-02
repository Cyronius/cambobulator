# Build dist\cambobulator.exe: a single file that runs on a Windows PC without Python.
# Usage, from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\build-exe.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv\Scripts\python.exe)) { py -3 -m venv .venv }
.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install --quiet -e ".[build]"

# --collect-data bundles web/, models/ and virtualcam/ (the driver DLLs) from the package.
.venv\Scripts\pyinstaller --noconfirm --clean --onefile --name cambobulator `
    --paths src --collect-data cambobulator --specpath build `
    src\cambobulator\__main__.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Get-Item dist\cambobulator.exe | Select-Object FullName, @{n = "MB"; e = { [math]::Round($_.Length / 1MB) } }
