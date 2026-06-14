param(
    [string]$Device = "cuda",
    [int]$MaxIter = 5000,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$experimentRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $experimentRoot "..")).Path
$solver = Join-Path $repoRoot "CFNO_Coupled_uvp.py"
$logDir = Join-Path $experimentRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$baseArgs = @(
    "-u", $solver,
    "--operator", "hf_cfno",
    "--output-mode", "streamfunction",
    "--n", "129",
    "--max-iter", "$MaxIter",
    "--tol", "1e-8",
    "--device", $Device,
    "--modes", "16",
    "--high-modes", "32",
    "--width", "24",
    "--depth", "5",
    "--lr", "1e-3",
    "--interior-margin", "1",
    "--seed", "10492"
)

$cases = @(
    [ordered]@{
        Name = "old_hf_cfno"
        Description = "Original HF-CFNO reproduction with legacy mixed rotor padding."
        Args = @(
            "--gate-mode", "legacy",
            "--boundary-mode", "legacy_mixed",
            "--high-gate-init", "-1.0",
            "--gate-alignment-weight", "0.0"
        )
    },
    [ordered]@{
        Name = "new_gate_weak"
        Description = "Current subgrid vorticity gate."
        Args = @(
            "--gate-mode", "subgrid",
            "--boundary-mode", "replicate",
            "--high-gate-init", "-2.0",
            "--gate-threshold", "1.0",
            "--gate-slope", "2.0",
            "--gate-subgrid-weight", "1.0",
            "--gate-alignment-weight", "1e-2"
        )
    },
    [ordered]@{
        Name = "new_gate_stronger"
        Description = "Stronger global cap and stronger spatial-gate alignment."
        Args = @(
            "--gate-mode", "subgrid",
            "--boundary-mode", "replicate",
            "--high-gate-init", "-1.0",
            "--gate-threshold", "1.0",
            "--gate-slope", "2.0",
            "--gate-subgrid-weight", "1.0",
            "--gate-alignment-weight", "0.05"
        )
    },
    [ordered]@{
        Name = "new_gate_no_align"
        Description = "Subgrid vorticity gate without explicit physical alignment loss."
        Args = @(
            "--gate-mode", "subgrid",
            "--boundary-mode", "replicate",
            "--high-gate-init", "-2.0",
            "--gate-threshold", "1.0",
            "--gate-slope", "2.0",
            "--gate-subgrid-weight", "1.0",
            "--gate-alignment-weight", "0.0"
        )
    },
    [ordered]@{
        Name = "new_gate_with_fuse_but_gated"
        Description = "Restores legacy fuse capacity, but gates all high/fuse residual by lambda*g."
        Args = @(
            "--gate-mode", "subgrid_gated_fuse",
            "--boundary-mode", "replicate",
            "--high-gate-init", "-2.0",
            "--gate-threshold", "1.0",
            "--gate-slope", "2.0",
            "--gate-subgrid-weight", "1.0",
            "--gate-alignment-weight", "1e-2"
        )
    }
)

$config = foreach ($case in $cases) {
    [ordered]@{
        name = $case.Name
        description = $case.Description
        args = $case.Args
    }
}
$config | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $experimentRoot "case_config.json") -Encoding UTF8

$statusFile = Join-Path $experimentRoot "ablation_status.csv"
"case,status,start_time,end_time,exit_code,output_dir,stdout_log,stderr_log" | Set-Content -Path $statusFile -Encoding UTF8

foreach ($case in $cases) {
    $name = $case.Name
    $outDir = Join-Path $experimentRoot $name
    $stdoutLog = Join-Path $logDir "$name.log"
    $stderrLog = Join-Path $logDir "$name.err.log"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $statusFile -Value "$name,running,$start,,,$outDir,$stdoutLog,$stderrLog"
    Write-Host "[$start] Running $name"

    $allArgs = $baseArgs + @("--output-dir", $outDir) + $case.Args
    & $Python @allArgs > $stdoutLog 2> $stderrLog
    $exitCode = $LASTEXITCODE

    $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($exitCode -eq 0) {
        Add-Content -Path $statusFile -Value "$name,finished,$start,$end,$exitCode,$outDir,$stdoutLog,$stderrLog"
        Write-Host "[$end] Finished $name"
    } else {
        Add-Content -Path $statusFile -Value "$name,failed,$start,$end,$exitCode,$outDir,$stdoutLog,$stderrLog"
        throw "$name failed with exit code $exitCode. See $stderrLog"
    }
}

& $Python (Join-Path $experimentRoot "summarize_gate_ablation.py")
