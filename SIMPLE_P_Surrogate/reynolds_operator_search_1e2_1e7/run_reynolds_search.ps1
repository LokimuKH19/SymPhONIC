param(
    [string]$Device = "cuda",
    [int]$MaxIter = 5000,
    [int]$N = 129,
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
    "--output-mode", "streamfunction",
    "--n", "$N",
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

$reynolds = @(
    [ordered]@{ Label = "Re1e02"; Value = "1e2" },
    [ordered]@{ Label = "Re1e03"; Value = "1e3" },
    [ordered]@{ Label = "Re1e04"; Value = "1e4" },
    [ordered]@{ Label = "Re1e05"; Value = "1e5" },
    [ordered]@{ Label = "Re1e06"; Value = "1e6" },
    [ordered]@{ Label = "Re1e07"; Value = "1e7" }
)

$methods = @(
    [ordered]@{
        Name = "fno"
        Description = "Plain Fourier neural operator with the same Fourier coordinate features."
        Args = @("--operator", "fno", "--boundary-mode", "replicate", "--gate-alignment-weight", "0.0")
    },
    [ordered]@{
        Name = "cfno"
        Description = "Plain coupled Chebyshev-Fourier neural operator with the same Fourier coordinate features."
        Args = @("--operator", "cfno", "--boundary-mode", "replicate", "--gate-alignment-weight", "0.0")
    },
    [ordered]@{
        Name = "hf_cfno"
        Description = "Legacy HF-CFNO high-pass mechanism under non-periodic cavity padding."
        Args = @(
            "--operator", "hf_cfno",
            "--gate-mode", "legacy",
            "--boundary-mode", "replicate",
            "--high-gate-init", "-1.0",
            "--gate-alignment-weight", "0.0"
        )
    },
    [ordered]@{
        Name = "subgrid_hf_cfno"
        Description = "Subgrid vorticity-gated HF-CFNO without explicit L_gate alignment."
        Args = @(
            "--operator", "hf_cfno",
            "--gate-mode", "subgrid",
            "--boundary-mode", "replicate",
            "--high-gate-init", "-2.0",
            "--gate-threshold", "1.0",
            "--gate-slope", "2.0",
            "--gate-subgrid-weight", "1.0",
            "--gate-alignment-weight", "0.0"
        )
    }
)

$config = [ordered]@{
    n = $N
    max_iter = $MaxIter
    reynolds = $reynolds
    methods = $methods
}
$config | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $experimentRoot "case_config.json") -Encoding UTF8

$statusFile = Join-Path $experimentRoot "reynolds_search_status.csv"
"re,method,status,start_time,end_time,exit_code,output_dir,stdout_log,stderr_log" | Set-Content -Path $statusFile -Encoding UTF8

foreach ($re in $reynolds) {
    foreach ($method in $methods) {
        $caseName = "$($re.Label)_$($method.Name)"
        $outDir = Join-Path (Join-Path $experimentRoot $re.Label) $method.Name
        $stdoutLog = Join-Path $logDir "$caseName.log"
        $stderrLog = Join-Path $logDir "$caseName.err.log"
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null

        $start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path $statusFile -Value "$($re.Value),$($method.Name),running,$start,,,$outDir,$stdoutLog,$stderrLog"
        Write-Host "[$start] Running Re=$($re.Value), method=$($method.Name)"

        $allArgs = $baseArgs + @("--re", "$($re.Value)", "--output-dir", $outDir) + $method.Args
        & $Python @allArgs > $stdoutLog 2> $stderrLog
        $exitCode = $LASTEXITCODE
        $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        if ($exitCode -eq 0) {
            Add-Content -Path $statusFile -Value "$($re.Value),$($method.Name),finished,$start,$end,$exitCode,$outDir,$stdoutLog,$stderrLog"
            Write-Host "[$end] Finished Re=$($re.Value), method=$($method.Name)"
        } else {
            Add-Content -Path $statusFile -Value "$($re.Value),$($method.Name),failed,$start,$end,$exitCode,$outDir,$stdoutLog,$stderrLog"
            Write-Host "[$end] Failed Re=$($re.Value), method=$($method.Name), exit=$exitCode"
        }
    }
}

& $Python (Join-Path $experimentRoot "summarize_reynolds_search.py")
