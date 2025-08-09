# run.ps1 - EdgeMind Platform Windows Scripts

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ProjectRoot = $PSScriptRoot
$VenvPath = "$ProjectRoot\ai-env\Scripts"

function Show-Help {
    Write-Host "EdgeMind Platform - Windows Commands" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\run.ps1 [command]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Green
    Write-Host "  install     - Install Python dependencies"
    Write-Host "  api         - Run FastAPI backend"
    Write-Host "  ui          - Run Streamlit UI"
    Write-Host "  both        - Run both API and UI (opens 2 terminals)"
    Write-Host "  docker      - Build and run with Docker"
    Write-Host "  clean       - Clean up Docker containers"
    Write-Host "  test        - Run tests"
    Write-Host ""
}

function Install-Dependencies {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & "$VenvPath\activate.ps1"
    pip install -r requirements_core.txt
    pip install fastapi uvicorn redis streamlit
    Write-Host "✅ Dependencies installed!" -ForegroundColor Green
}

function Start-API {
    Write-Host "Starting FastAPI..." -ForegroundColor Yellow
    & "$VenvPath\activate.ps1"
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
}

function Start-UI {
    Write-Host "Starting Streamlit UI..." -ForegroundColor Yellow
    & "$VenvPath\activate.ps1"
    streamlit run web/streamlit_app.py
}

function Start-Both {
    Write-Host "Starting both services..." -ForegroundColor Yellow
    
    # Start API in new window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; .\ai-env\Scripts\activate; uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"
    
    # Start UI in new window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; .\ai-env\Scripts\activate; streamlit run web/streamlit_app.py"
    
    Write-Host "✅ Services starting in new windows!" -ForegroundColor Green
    Write-Host "📺 Streamlit UI: http://localhost:8501" -ForegroundColor Cyan
    Write-Host "🚀 FastAPI: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "📊 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
}

function Start-Docker {
    Write-Host "Building and starting Docker containers..." -ForegroundColor Yellow
    docker-compose build
    docker-compose up -d
    Write-Host "✅ Docker containers started!" -ForegroundColor Green
    Write-Host "📺 Streamlit UI: http://localhost:8501" -ForegroundColor Cyan
    Write-Host "🚀 FastAPI: http://localhost:8000" -ForegroundColor Cyan
}

function Clean-Docker {
    Write-Host "Cleaning Docker containers..." -ForegroundColor Yellow
    docker-compose down -v
    docker system prune -f
    Write-Host "✅ Cleaned!" -ForegroundColor Green
}

function Run-Tests {
    Write-Host "Running tests..." -ForegroundColor Yellow
    & "$VenvPath\activate.ps1"
    pytest tests/ -v
}

# Execute command
switch ($Command) {
    "help"    { Show-Help }
    "install" { Install-Dependencies }
    "api"     { Start-API }
    "ui"      { Start-UI }
    "both"    { Start-Both }
    "docker"  { Start-Docker }
    "clean"   { Clean-Docker }
    "test"    { Run-Tests }
    default   { 
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Help 
    }
}