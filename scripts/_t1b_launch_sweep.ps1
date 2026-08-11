# TRACK 1b' / Tier 1 sweep launcher.
# Sequentially:
#   for K in 256, 64, 16:
#     1. Train bottleneck head (20 epochs max, patience 5)  [skip if metrics.json already exists]
#     2. Export K-d latents for all 6 splits                [skip individual files that already exist]
#
# K=1024 is skipped: exp7_baseline serves as the K=1024 reference.
#
# Error handling: $ErrorActionPreference is "Continue" because conda writes to stderr
# on success in some envs, which PowerShell strict mode would interpret as a fatal
# error. We rely on $LASTEXITCODE from cmd /c, not on stderr presence.

$ErrorActionPreference = "Continue"
$REPO = Split-Path -Parent $PSScriptRoot
Set-Location $REPO
$LOG = Join-Path $REPO "outputs\_t1b_sweep_log.txt"
"" | Out-File -FilePath $LOG -Encoding UTF8

function Invoke-Stage {
    param([string]$Msg, [string]$Cmd)
    $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    "===  $stamp  ===  $Msg"  | Tee-Object -FilePath $LOG -Append | Write-Host
    "    -> $Cmd"             | Tee-Object -FilePath $LOG -Append | Write-Host
    # cmd /c absorbs the conda-on-stderr noise and reports a single exit code.
    cmd /c "$Cmd >> `"$LOG`" 2>&1"
    $exit = $LASTEXITCODE
    "    exit=$exit"           | Tee-Object -FilePath $LOG -Append | Write-Host
    if ($exit -ne 0) {
        "!!! exit code $exit for: $Cmd" | Tee-Object -FilePath $LOG -Append | Write-Host
        throw "Stage failed: $Msg (exit=$exit)"
    }
}

$KS = @(256, 64, 16)
$CKPT = "outputs\exp7_baseline\checkpoints\linear_best.pt"

foreach ($K in $KS) {
    $RUN_ID = "exp7_bottleneck_K$K"
    $BOTTLE_CKPT = "outputs\$RUN_ID\checkpoints\linear_best.pt"

    # Skip training if metrics.json + best ckpt already exist.
    $metrics = "outputs\$RUN_ID\metrics.json"
    if ((Test-Path $metrics) -and (Test-Path $BOTTLE_CKPT)) {
        "===  SKIP TRAIN K=$K  (already done: $metrics + $BOTTLE_CKPT)" |
            Tee-Object -FilePath $LOG -Append | Write-Host
    } else {
        $trainCmd = "conda run -n ECGFounder python scripts\finetune_bottleneck.py --checkpoint $CKPT --bottleneck-dim $K --run-id $RUN_ID --epochs 20 --patience 5"
        Invoke-Stage -Msg "TRAIN K=$K" -Cmd $trainCmd
    }

    foreach ($D in @('medalcare', 'ptbxl')) {
        foreach ($S in @('train', 'val', 'test')) {
            $OUT = "outputs\latents\${RUN_ID}_${D}_$S"
            $npz = "$OUT\latents.npz"
            if (Test-Path $npz) {
                "===  SKIP EXPORT K=$K $D/$S  (already done: $npz)" |
                    Tee-Object -FilePath $LOG -Append | Write-Host
                continue
            }
            $expCmd = "conda run -n ECGFounder python scripts\export_bottleneck_latents.py --checkpoint $BOTTLE_CKPT --dataset $D --split $S --outdir $OUT"
            Invoke-Stage -Msg "EXPORT K=$K dataset=$D split=$S" -Cmd $expCmd
        }
    }
}

$stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
"===  $stamp  ===  SWEEP COMPLETE" | Tee-Object -FilePath $LOG -Append | Write-Host
