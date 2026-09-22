$ErrorActionPreference = "Stop"
Set-Location "D:\Ansys\SymPhONIC\OtherPDEs"

$baselinePid = 34148
$root = "LegacyHF_PDE_Benchmark_Hybrid_20260705"
$watchLog = ".\benchmark_logs\legacy_hf_hybrid_watcher_20260706.log"
$out = ".\benchmark_logs\legacy_hf_hybrid_all_high_nonlinear_20260706.out.log"
$err = ".\benchmark_logs\legacy_hf_hybrid_all_high_nonlinear_20260706.err.log"

function Write-WatchLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $watchLog -Value $line
}

Write-WatchLog "Waiting for baseline PID $baselinePid"
try {
    Wait-Process -Id $baselinePid
} catch {
    Write-WatchLog "Baseline PID $baselinePid is no longer present: $($_.Exception.Message)"
}

$csv = Join-Path $root "global_comparison.csv"
if (-not (Test-Path -LiteralPath $csv)) {
    Write-WatchLog "Not starting high_nonlinear: missing $csv"
    exit 1
}

$rows = (Import-Csv -LiteralPath $csv | Measure-Object).Count
Write-WatchLog "Baseline row count after wait: $rows"
if ($rows -lt 60) {
    Write-WatchLog "Not starting high_nonlinear: expected at least 60 baseline rows"
    exit 1
}

$args = @(
    "-u", ".\run_legacy_hf_pde_suite.py",
    "--training-mode", "hybrid",
    "--physics-weight", "0.05",
    "--profile", "high_nonlinear",
    "--append-existing-global",
    "--output-root", $root,
    "--epochs", "1200",
    "--grid", "96",
    "--modes", "12",
    "--high-modes", "24",
    "--target-params", "500000",
    "--train-samples", "32",
    "--val-samples", "8",
    "--test-samples", "4",
    "--batch-size", "16"
)

$proc = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory (Get-Location) -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -WindowStyle Hidden
Write-WatchLog "Started high_nonlinear PID $($proc.Id)"
