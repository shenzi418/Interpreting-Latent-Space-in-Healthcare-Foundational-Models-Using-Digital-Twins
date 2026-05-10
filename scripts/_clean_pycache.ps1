# Recursively delete all __pycache__ directories and stray .pyc files in the repo root.
$root = Split-Path -Parent $PSScriptRoot

$caches = Get-ChildItem -Path $root -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue
foreach ($c in $caches) {
    Write-Output ("Removing dir : " + $c.FullName)
    Remove-Item -Recurse -Force $c.FullName
}

$pycs = Get-ChildItem -Path $root -Recurse -File -Include *.pyc -ErrorAction SilentlyContinue
foreach ($p in $pycs) {
    Write-Output ("Removing file: " + $p.FullName)
    Remove-Item -Force $p.FullName
}

Write-Output "DONE"
