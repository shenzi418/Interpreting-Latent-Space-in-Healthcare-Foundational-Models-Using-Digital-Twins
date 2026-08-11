# Tier 2 post-processing: export latents + run eval for a given run-id.
# Usage: powershell scripts/_t2_postprocess.ps1 <run_id> <eval_json_name>
param(
  [Parameter(Mandatory=$true)] [string] $RunId,
  [Parameter(Mandatory=$true)] [string] $EvalJsonName
)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$py = "F:\anaconda3\envs\ECGFounder\python.exe"
$ckpt = "outputs/$RunId/checkpoints/linear_best.pt"

# Export latents (4 splits x 2 datasets)
$splits = @(
  @("medalcare","train"), @("medalcare","val"), @("medalcare","test"),
  @("ptbxl","train"),     @("ptbxl","val"),     @("ptbxl","test")
)
foreach ($s in $splits) {
  $domain = $s[0]; $split = $s[1]
  $outdir = "outputs/latents/${RunId}_${domain}_${split}"
  Write-Host "[export] $RunId $domain $split -> $outdir"
  & $py -X utf8 scripts/export_bottleneck_latents.py `
    --checkpoint $ckpt `
    --dataset $domain `
    --split $split `
    --outdir $outdir `
    --batch-size 128 `
    --num-workers 0
  if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] export failed for $domain/$split"; exit 1 }
}

# Run Tier 2 eval (L1+L2+L3)
$evalOut = "outputs/inlp_lowK/$EvalJsonName"
Write-Host "[eval] $RunId -> $evalOut"
& $py -X utf8 analysis/eval_tier2.py `
  --run-prefix $RunId `
  --label $RunId `
  --out $evalOut
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] eval failed"; exit 1 }

Write-Host "[done] $RunId postprocess complete"
