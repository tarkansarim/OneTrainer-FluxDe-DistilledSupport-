$pythonPath = $env:SRPO_PYTHON
if ([string]::IsNullOrWhiteSpace($pythonPath) -or -not (Test-Path $pythonPath)) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCmd) {
        Write-Error "Python executable not found. Install Python and ensure it is on PATH."
        exit 1
    }
    $pythonPath = $pythonCmd.Source
}

$repoPath = $env:SRPO_REPO_PATH
if ([string]::IsNullOrWhiteSpace($repoPath) -or -not (Test-Path $repoPath)) {
    $repoPath = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$workingDir = $env:SRPO_WORKING_DIR
if ([string]::IsNullOrWhiteSpace($workingDir) -or -not (Test-Path $workingDir)) {
    $workingDir = $repoPath
}

if ($args.Length -lt 1) {
    Write-Error "Missing SRPO script path."
    exit 1
}

$forwardArgs = @()
if ($args.Length -gt 1) {
    $forwardArgs = $args[1..($args.Length - 1)]
}

$scriptPath = Join-Path $repoPath "fastvideo/SRPO.py"

Push-Location $workingDir
try {
    & $pythonPath $scriptPath @forwardArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
