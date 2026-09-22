param(
    [int]$Epochs = 1200,
    [int]$MicroBatchSize = 8,
    [string]$OutputRoot = "LegacyHF_PDE_Benchmark_UniqueNativeSteady1D_SqrtParams_Final_20260820",
    [int]$MaximumAttempts = 4
)

$ErrorActionPreference = "Stop"
$Python = "python"
$Program = Join-Path $PSScriptRoot "run_legacy_hf_pde_steady1d_suite.py"
$Modes = @("pde", "hybrid", "data")
$Cases = @(
    "Poisson_Steady1D",
    "KdV_Steady1D",
    "AllenCahn_Steady1D",
    "Burgers_Steady1D",
    "ReactionDiffusion_Steady1D",
    "KdV_HighNonlinear_Steady1D",
    "AllenCahn_HighNonlinear_Steady1D",
    "Burgers_HighNonlinear_Steady1D",
    "ReactionDiffusion_HighNonlinear_Steady1D"
)
$Variants = @("FNO", "CFNO", "HF_FNO", "HF_CFNO")
$ModeDirectories = @{pde = "PhysicsOnly"; hybrid = "Hybrid"; data = "DataOnly"}

foreach ($Mode in $Modes) {
    foreach ($Case in $Cases) {
        $Separator = $Case.IndexOf("_")
        $Pde = $Case.Substring(0, $Separator)
        $CaseName = $Case.Substring($Separator + 1)
        foreach ($Variant in $Variants) {
            $Metrics = Join-Path $PSScriptRoot "$OutputRoot\$($ModeDirectories[$Mode])\$Pde\$CaseName\$Variant\metrics.json"
            if (Test-Path -LiteralPath $Metrics) {
                Write-Host "Skipping completed $Mode/$Case/$Variant"
                continue
            }
            $Succeeded = $false
            for ($Attempt = 1; $Attempt -le $MaximumAttempts; $Attempt++) {
                $AttemptMicroBatch = [Math]::Max(1, [Math]::Floor($MicroBatchSize / [Math]::Pow(2, $Attempt - 1)))
                Write-Host "Running $Mode/$Case/$Variant (attempt $Attempt/$MaximumAttempts, micro-batch $AttemptMicroBatch)"
                & $Python $Program `
                    --output-root $OutputRoot `
                    --epochs $Epochs `
                    --micro-batch-size $AttemptMicroBatch `
                    --training-modes $Mode `
                    --cases $Case `
                    --variants $Variant `
                    --skip-plots `
                    --resume
                if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $Metrics)) {
                    $Succeeded = $true
                    break
                }
                Start-Sleep -Seconds 5
            }
            if (-not $Succeeded) {
                throw "Failed after $MaximumAttempts attempts: $Mode/$Case/$Variant"
            }
        }
    }
}

& $Python $Program `
    --output-root $OutputRoot `
    --epochs $Epochs `
    --micro-batch-size $MicroBatchSize `
    --skip-plots `
    --resume
if ($LASTEXITCODE -ne 0) {
    throw "Final benchmark aggregation failed."
}

& $Python $Program `
    --output-root $OutputRoot `
    --epochs $Epochs `
    --plots-only
if ($LASTEXITCODE -ne 0) {
    throw "Final benchmark plotting failed."
}
