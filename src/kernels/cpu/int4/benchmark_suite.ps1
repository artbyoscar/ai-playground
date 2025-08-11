# benchmark_suite.ps1
param(
    [string]$OutputDir = ".\benchmark_results",
    [switch]$Quick = $false
)

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultFile = Join-Path $OutputDir "results_$timestamp.csv"

Write-Host "`n🚀 EdgeMind INT4/Q8 Benchmark Suite" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Test configurations
if ($Quick) {
    $configs = @(
        @{M=256; N=256; K=2048}
    )
    $iterations = 3
} else {
    $configs = @(
        @{M=128; N=128; K=1024},
        @{M=256; N=256; K=2048},
        @{M=512; N=512; K=4096},
        @{M=1024; N=1024; K=4096}
    )
    $iterations = 5
}

$threadCounts = @(4, 8, 16)
$results = @()

# Function to run benchmark
function Run-Benchmark {
    param($Exe, $M, $N, $K, $Threads, $Iterations, $ExtraArgs = "")
    
    $cmd = "& `"$Exe`" --M $M --N $N --K $K --it $Iterations --threads $Threads $ExtraArgs 2>&1"
    $output = Invoke-Expression $cmd
    
    # Extract GFLOP/s from output
    $gflops = $output | Select-String -Pattern "(\d+\.\d+)\s+GFLOP/s" | 
              ForEach-Object { $_.Matches[0].Groups[1].Value } | 
              Select-Object -Last 1
    
    return [double]$gflops
}

Write-Host "`nBuilding all configurations..." -ForegroundColor Yellow

# Build configurations
$builds = @(
    @{Name="Standard"; Dir="build"; Flags=""},
    @{Name="Fused"; Dir="build-fused"; Flags="-DINT4_FUSE_BIAS=ON"}
)

foreach ($build in $builds) {
    Write-Host "  Building $($build.Name)..." -NoNewline
    $cmd = "cmake -S . -B $($build.Dir) -G Ninja -DCMAKE_BUILD_TYPE=Release " +
           "-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ $($build.Flags) 2>&1 | Out-Null"
    Invoke-Expression $cmd
    cmake --build $($build.Dir) 2>&1 | Out-Null
    Write-Host " Done" -ForegroundColor Green
}

Write-Host "`nRunning benchmarks..." -ForegroundColor Yellow

# CSV header
"Timestamp,Build,Kernel,M,N,K,Threads,GFLOPS,Time_ms" | Out-File $resultFile

foreach ($config in $configs) {
    Write-Host "`n  Testing M=$($config.M) N=$($config.N) K=$($config.K)" -ForegroundColor Cyan
    
    foreach ($threads in $threadCounts) {
        Write-Host "    Threads=$threads" -ForegroundColor Yellow
        
        # Test INT4 tiled
        if (Test-Path ".\build\test_qgemm_perf_tiled_mt.exe") {
            $gflops = Run-Benchmark ".\build\test_qgemm_perf_tiled_mt.exe" `
                                   $config.M $config.N $config.K $threads $iterations
            
            $time_ms = (2.0 * $config.M * $config.N * $config.K) / (1e9 * $gflops / 1000)
            Write-Host "      INT4 Tiled: $gflops GFLOP/s" -ForegroundColor Green
            
            "$timestamp,Standard,INT4_Tiled,$($config.M),$($config.N),$($config.K),$threads,$gflops,$time_ms" | 
                Add-Content $resultFile
        }
        
        # Test Q8
        if (Test-Path ".\build\test_qgemm_perf_q8_mt.exe") {
            $gflops = Run-Benchmark ".\build\test_qgemm_perf_q8_mt.exe" `
                                   $config.M $config.N $config.K $threads $iterations
            
            $time_ms = (2.0 * $config.M * $config.N * $config.K) / (1e9 * $gflops / 1000)
            Write-Host "      Q8: $gflops GFLOP/s" -ForegroundColor Green
            
            "$timestamp,Standard,Q8,$($config.M),$($config.N),$($config.K),$threads,$gflops,$time_ms" | 
                Add-Content $resultFile
        }
        
        # Test Fused
        if (Test-Path ".\build-fused\test_perf_bias.exe") {
            $output = & ".\build-fused\test_perf_bias.exe" `
                        --M $config.M --N $config.N --K $config.K `
                        --it $iterations --use_bias 1 --relu 1 --fused 1 2>&1
            
            $fusedLine = $output | Select-String "Fused.*GFLOP/s"
            if ($fusedLine) {
                $gflops = $fusedLine -replace '.*\((\d+\.\d+) GFLOP/s\).*', '$1'
                $time_ms = (2.0 * $config.M * $config.N * $config.K) / (1e9 * [double]$gflops / 1000)
                Write-Host "      Fused: $gflops GFLOP/s" -ForegroundColor Green
                
                "$timestamp,Fused,INT4_Fused,$($config.M),$($config.N),$($config.K),$threads,$gflops,$time_ms" | 
                    Add-Content $resultFile
            }
        }
    }
}

Write-Host "`n✅ Benchmark complete! Results saved to: $resultFile" -ForegroundColor Green

# Generate summary
Write-Host "`n📊 Performance Summary:" -ForegroundColor Cyan
$data = Import-Csv $resultFile
$summary = $data | Group-Object Kernel | ForEach-Object {
    $avg = ($_.Group | Measure-Object -Property GFLOPS -Average).Average
    $max = ($_.Group | Measure-Object -Property GFLOPS -Maximum).Maximum
    [PSCustomObject]@{
        Kernel = $_.Name
        "Avg GFLOP/s" = [math]::Round($avg, 2)
        "Max GFLOP/s" = [math]::Round($max, 2)
    }
}
$summary | Format-Table -AutoSize