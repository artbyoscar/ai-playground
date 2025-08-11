# generate_report.ps1
Write-Host "`n📈 EdgeMind INT4/Q8 Performance Report" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Get latest results
$latest = Get-ChildItem .\benchmark_results\*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($latest) {
    $data = Import-Csv $latest.FullName
    
    # Best performance by kernel type
    Write-Host "`n🏆 Best Performance:" -ForegroundColor Yellow
    $data | Group-Object Kernel | ForEach-Object {
        $best = $_.Group | Sort-Object GFLOPS -Descending | Select-Object -First 1
        Write-Host "  $($_.Name): $($best.GFLOPS) GFLOP/s (M=$($best.M) N=$($best.N) K=$($best.K) T=$($best.Threads))" -ForegroundColor Green
    }
    
    # Speedup analysis
    Write-Host "`n📊 Speedup Analysis:" -ForegroundColor Yellow
    $baseline = ($data | Where-Object {$_.Kernel -eq "INT4_Tiled"} | Measure-Object GFLOPS -Average).Average
    if ($baseline) {
        $data | Group-Object Kernel | ForEach-Object {
            $avg = ($_.Group | Measure-Object GFLOPS -Average).Average
            $speedup = $avg / $baseline
            Write-Host ("  {0,-15} {1:F2}× speedup" -f $_.Name, $speedup)
        }
    }
}

Write-Host "`n✅ Report complete!" -ForegroundColor Green