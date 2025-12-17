# Skin Cancer Model Training Pipeline - Run Script
# This script activates the virtual environment and runs the workflow

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "🚀 Skin Cancer Model Training Pipeline" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Set PYTHONPATH
$env:PYTHONPATH = "C:\Users\ibrah\Desktop\New Project"

# Run the workflow
Write-Host "🏃 Running workflow...`n" -ForegroundColor Green
python app/workflow.py $args

Write-Host "`n✅ Pipeline complete!" -ForegroundColor Green
Write-Host "📊 Check reports/validation_report.html for detailed analysis`n" -ForegroundColor Cyan
