param(
  [string]$DmlVersion   = "1.13.1",
  [string]$D3D12Version = "1.614.1"
)

$ErrorActionPreference = "Stop"

# Make TLS happy on older boxes
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = (Resolve-Path "$PSScriptRoot\..").Path
$tp   = Join-Path $root "third_party"
New-Item -ItemType Directory -Force -Path $tp | Out-Null

$pkgs = @(
  @{ Id="Microsoft.AI.DirectML";    Ver=$DmlVersion;   Dst="Microsoft.AI.DirectML.$DmlVersion" },
  @{ Id="Microsoft.Direct3D.D3D12"; Ver=$D3D12Version; Dst="Microsoft.Direct3D.D3D12.$D3D12Version" }
)

function Get-NuGetPackage {
  param(
    [Parameter(Mandatory=$true)][string]$Id,
    [Parameter(Mandatory=$true)][string]$Ver,
    [Parameter(Mandatory=$true)][string]$DstDir
  )
  if (Test-Path $DstDir) {
    Write-Host "Already present: $Id $Ver at $DstDir"
    return
  }

  New-Item -ItemType Directory -Force -Path $DstDir | Out-Null

  $url   = "https://www.nuget.org/api/v2/package/$Id/$Ver"
  $nupkg = Join-Path $DstDir "$Id.$Ver.nupkg"

  Write-Host "Downloading $Id $Ver ..."
  Invoke-WebRequest -Uri $url -OutFile $nupkg

  Write-Host "Extracting $nupkg ..."
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [IO.Compression.ZipFile]::ExtractToDirectory($nupkg, $DstDir)

  Remove-Item $nupkg -Force
  Write-Host "Installed: $Id $Ver -> $DstDir"
}

foreach ($p in $pkgs) {
  $dst = Join-Path $tp $p.Dst
  Get-NuGetPackage -Id $p.Id -Ver $p.Ver -DstDir $dst
}

Write-Host "Done."
