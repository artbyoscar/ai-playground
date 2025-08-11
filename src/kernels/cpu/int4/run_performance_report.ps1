# Performance Report
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$results = @()

# Test configurations
$tests = @(
    @{Name="INT4 Single"; Exe=".\build-final\test_qgemm_perf.exe"; Args="--M 256 --N 256 --K 2048 --it 5"},
    @{Name="INT4 MT"; Exe=".\build-final\test_qgemm_perf_mt.exe"; Args="--M 256 --N 256 --K 2048 --it 5"},
    @{Name="INT4 Tiled"; Exe=".\build-final\test_qgemm_perf_tiled_mt.exe"; Args="--M 256 --N 256 --K 2048 --it 5"},
    @{Name="Q8 MT"; Exe=".\build-final\test_qgemm_perf_q8_mt.exe"; Args="--M 256 --N 256 --K 2048 --it 5 --threads 8"},
    @{Name="Baseline"; Exe=".\build-final\test_qgemm_perf_vs_baseline.exe"; Args="--M 256 --N 256 --K 2048 --it 5"}
)

Write-Host "`n📊 EdgeMind Performance Report - $timestamp" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

foreach ($test in $tests) {
    if (Test-Path $test.Exe) {
        $output = & cmd /c "$($test.Exe) $($test.Args) 2>&1"
        $lines = $output | Select-String -Pattern "(\d+\.\d+)\s+GFLOP/s"
        
        foreach ($line in $lines) {
            if ($line -match "(\w+)\s*:\s*[\d.]+\s*ms\s*\((\d+\.\d+)\s*GFLOP/s\)") {
                $variant = $matches[1]
                $gflops = [double]$matches[2]
                $results += [PSCustomObject]@{
                    Test = "$($test.Name)-$variant"
                    GFLOPS = $gflops
                }
            }
        }
    }
}

# Display results
Write-Host "`n📈 Performance Summary:" -ForegroundColor Yellow
$results | Sort-Object GFLOPS -Descending | Format-Table -AutoSize

# Calculate speedups
$baseline = $results | Where-Object {$_.Test -like "*FP32*"} | Select-Object -First 1
if ($baseline) {
    Write-Host "`n⚡ Speedup vs FP32 Baseline ($($baseline.GFLOPS) GFLOP/s):" -ForegroundColor Yellow
    $results | ForEach-Object {
        $speedup = $_.GFLOPS / $baseline.GFLOPS
        Write-Host ("  {0,-20} {1,8:F2} GFLOP/s = {2,6:F1}×" -f $_.Test, $_.GFLOPS, $speedup)
    }
}

# Find best result
$best = $results | Sort-Object GFLOPS -Descending | Select-Object -First 1
Write-Host "`n🏆 Best Performance: $($best.Test)" -ForegroundColor Green
Write-Host "   $($best.GFLOPS) GFLOP/s" -ForegroundColor Green

# Export to CSV
$csvFile = "performance_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
$results | Export-Csv -Path $csvFile -NoTypeInformation
Write-Host "`n📁 Results saved to: $csvFile" -ForegroundColor Cyan
