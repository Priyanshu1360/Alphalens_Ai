$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python not found in venv at: $python"
}

# Safe defaults for UI stability
$env:QDRANT_FALLBACK_TO_LOCAL_ON_ERROR = "true"
$env:EMBEDDING_FALLBACK_TO_LOCAL_ON_ERROR = "true"

& $python -m streamlit run streamlit_app.py --browser.gatherUsageStats false
