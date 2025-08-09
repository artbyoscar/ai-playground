# setup.ps1 - EdgeMind Platform Windows Setup

Write-Host "🚀 EdgeMind Platform - Windows Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (Test-Path ".\ai-env") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv ai-env
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\ai-env\Scripts\activate.ps1"

# Create directory structure
Write-Host "Creating directory structure..." -ForegroundColor Yellow
$dirs = @("src", "src\api", "src\agents", "src\core", "web", "models", "data", "tests")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✅ Created $dir" -ForegroundColor Green
    }
}

# Create __init__.py files
Write-Host "Creating __init__.py files..." -ForegroundColor Yellow
$initFiles = @("src\__init__.py", "src\api\__init__.py", "src\agents\__init__.py", "src\core\__init__.py")
foreach ($file in $initFiles) {
    if (!(Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "  ✅ Created $file" -ForegroundColor Green
    }
}

# Install core dependencies
Write-Host "Installing core dependencies..." -ForegroundColor Yellow
pip install --upgrade pip

# Create minimal requirements file if it doesn't exist
if (!(Test-Path "requirements_core.txt")) {
    @"
together>=0.2.11
requests>=2.31.0
python-dotenv>=1.0.0
streamlit>=1.28.0
pandas>=2.1.0
plotly>=5.17.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
redis>=5.0.0
beautifulsoup4>=4.12.0
httpx>=0.25.0
aiohttp>=3.9.0
loguru>=0.7.0
"@ | Out-File -FilePath "requirements_core.txt" -Encoding UTF8
}

pip install -r requirements_core.txt

# Install Playwright
Write-Host "Installing Playwright..." -ForegroundColor Yellow
pip install playwright
playwright install chromium

# Create .env file if it doesn't exist
if (!(Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    @"
# EdgeMind Platform Environment Variables
TOGETHER_API_KEY=your_api_key_here
ENVIRONMENT=development
REDIS_HOST=localhost
REDIS_PORT=6379
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "  ⚠️  Please add your TOGETHER_API_KEY to .env file" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Add your TOGETHER_API_KEY to .env file" -ForegroundColor White
Write-Host "2. Run: .\run.ps1 both" -ForegroundColor White
Write-Host "   This will start both API and UI" -ForegroundColor Gray
Write-Host ""
Write-Host "Available Commands:" -ForegroundColor Cyan
Write-Host "  .\run.ps1 install  - Install dependencies" -ForegroundColor White
Write-Host "  .\run.ps1 api      - Run API only" -ForegroundColor White
Write-Host "  .\run.ps1 ui       - Run UI only" -ForegroundColor White
Write-Host "  .\run.ps1 both     - Run both API and UI" -ForegroundColor White
Write-Host "  .\run.ps1 docker   - Run with Docker" -ForegroundColor White
Write-Host ""
Write-Host "URLs:" -ForegroundColor Cyan
Write-Host "  📺 Streamlit UI: http://localhost:8501" -ForegroundColor White
Write-Host "  🚀 FastAPI: http://localhost:8000" -ForegroundColor White
Write-Host "  📊 API Docs: http://localhost:8000/docs" -ForegroundColor White