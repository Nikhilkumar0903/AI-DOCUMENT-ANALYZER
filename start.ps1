$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv")) { py -3.11 -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    $nodeInstallDir = Join-Path $env:ProgramFiles "nodejs"
    if (Test-Path (Join-Path $nodeInstallDir "npm.cmd")) { $env:Path = "$nodeInstallDir;$env:Path" }
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "Node.js 20+ and npm are required. Install Node.js, then run this script again." }
if (-not (Test-Path "frontend\node_modules")) { Push-Location frontend; npm install; Pop-Location }
if (-not (Test-Path "frontend\dist\index.html")) { Push-Location frontend; npm run build; Pop-Location }
python main.py
