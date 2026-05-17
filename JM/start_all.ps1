param(
    [int]$Port = 8000,
    [int]$FrontPort = 3000,
    [switch]$NoFront,
    [switch]$NoSTT,
    [switch]$WithSTT,
    [switch]$NoVision,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }

        $parts = $line.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ($name) {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

function Test-PythonCommand {
    param(
        [string]$Exe,
        [string[]]$Args
    )

    try {
        & $Exe @Args --version *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Resolve-Python {
    $candidates = @(
        @{ Exe = Join-Path $Root ".venv\Scripts\python.exe"; Args = @() },
        @{ Exe = Join-Path $Root "venv\Scripts\python.exe"; Args = @() },
        @{ Exe = Join-Path $Root "env\Scripts\python.exe"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        if ((Test-Path -LiteralPath $candidate.Exe) -and (Test-PythonCommand -Exe $candidate.Exe -Args $candidate.Args)) {
            return [PSCustomObject]$candidate
        }
    }

    $localPythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path -LiteralPath $localPythonRoot) {
        $localPythons = Get-ChildItem -LiteralPath $localPythonRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "Python*" } |
            Sort-Object Name -Descending

        foreach ($pythonDir in $localPythons) {
            $pythonExe = Join-Path $pythonDir.FullName "python.exe"
            if ((Test-Path -LiteralPath $pythonExe) -and (Test-PythonCommand -Exe $pythonExe -Args @())) {
                return [PSCustomObject]@{ Exe = $pythonExe; Args = @() }
            }
        }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py -and (Test-PythonCommand -Exe $py.Source -Args @("-3"))) {
        return [PSCustomObject]@{ Exe = $py.Source; Args = @("-3") }
    }

    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and (Test-PythonCommand -Exe $cmd.Source -Args @())) {
            return [PSCustomObject]@{ Exe = $cmd.Source; Args = @() }
        }
    }

    throw "Python was not found. Create .venv first or install Python 3."
}

function Set-DefaultEnv {
    param(
        [string]$Name,
        [string]$Value
    )

    if (-not (Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue)) {
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

function Test-LocalModel {
    param([string]$ModelPath)

    if (-not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
        return $false
    }

    $configPath = Join-Path $ModelPath "config.json"
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        return $false
    }

    $weights = Get-ChildItem -LiteralPath $ModelPath -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*.safetensors" -or $_.Name -like "*.bin" } |
        Select-Object -First 1

    return $null -ne $weights
}

function Read-SecretText {
    param([string]$Prompt)

    $secure = Read-Host $Prompt -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

function Ensure-HFToken {
    $existingToken = Get-Item -Path "Env:HF_TOKEN" -ErrorAction SilentlyContinue
    if ($existingToken -and $existingToken.Value) {
        return
    }

    $localModelPath = $env:LLM_LOCAL_MODEL_DIR
    $hasLocalModel = Test-LocalModel -ModelPath $localModelPath

    if ($hasLocalModel) {
        $token = Read-SecretText -Prompt "Hugging Face token (local model found, press Enter to skip)"
        if ($token) {
            Set-Item -Path "Env:HF_TOKEN" -Value $token.Trim()
        }
        return
    }

    do {
        $token = Read-SecretText -Prompt "Hugging Face token (required because local model was not found)"
        $token = $token.Trim()
    } while (-not $token)

    Set-Item -Path "Env:HF_TOKEN" -Value $token
}

function Start-ProjectWindow {
    param(
        [string]$Title,
        [string]$WorkingDirectory,
        [string]$CommandTail,
        [object]$Python
    )

    $pythonExe = $Python.Exe.Replace("'", "''")
    $pythonArgs = ($Python.Args | ForEach-Object { "'" + ($_.Replace("'", "''")) + "'" }) -join " "
    $workingDirectory = $WorkingDirectory.Replace("'", "''")
    $title = $Title.Replace("'", "''")

    $command = @"
`$host.UI.RawUI.WindowTitle = '$title'
Set-Location -LiteralPath '$workingDirectory'
& '$pythonExe' $pythonArgs $CommandTail
"@

    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        $encoded
    )
}

function Wait-LLMServer {
    param([int]$Port)

    $baseUrl = if ($env:LLM_BASE_URL) { $env:LLM_BASE_URL.TrimEnd("/") } else { "http://127.0.0.1:$Port" }
    $healthUrl = "$baseUrl/health"
    Write-Host "Waiting for LLM server: $healthUrl"

    for ($i = 1; $i -le 180; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Write-Host "LLM server is ready."
                return
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }

    Write-Warning "LLM server did not answer /health in time. Starting remaining workers anyway."
}

Import-DotEnv -Path (Join-Path $Root ".env")
Import-DotEnv -Path (Join-Path $Root "LLM\.env")

Set-DefaultEnv -Name "LLM_MODEL_ID" -Value "skt/A.X-4.0-Light"
Set-DefaultEnv -Name "LLM_LOCAL_MODEL_DIR" -Value "D:\models\skt_A.X-4.0-Light"
Set-DefaultEnv -Name "HF_HOME" -Value "D:\models\hf_cache"
Set-DefaultEnv -Name "LLM_HOST" -Value "127.0.0.1"
Set-Item -Path "Env:LLM_PORT" -Value $Port
Set-DefaultEnv -Name "LLM_BASE_URL" -Value "http://127.0.0.1:$Port"
Set-Item -Path "Env:FRONT_PORT" -Value $FrontPort

$python = Resolve-Python
Write-Host "Using Python: $($python.Exe) $($python.Args -join ' ')"
Ensure-HFToken

Start-ProjectWindow `
    -Title "NLP-jm LLM" `
    -WorkingDirectory (Join-Path $Root "LLM") `
    -CommandTail "-m uvicorn main:app --host $env:LLM_HOST --port $Port" `
    -Python $python

if (-not $NoWait) {
    Wait-LLMServer -Port $Port
}

if ($WithSTT -and -not $NoSTT) {
    Start-ProjectWindow `
        -Title "NLP-jm STT" `
        -WorkingDirectory (Join-Path $Root "STT") `
        -CommandTail "stt_worker.py" `
        -Python $python
}
else {
    Write-Host "Legacy STT worker skipped. The browser UI uses one-shot voice input."
}

if (-not $NoVision) {
    Start-ProjectWindow `
        -Title "NLP-jm Vision" `
        -WorkingDirectory (Join-Path $Root "Vision") `
        -CommandTail "vision_worker.py" `
        -Python $python
}

if (-not $NoFront) {
    $frontCandidates = @(
        (Join-Path $Root "..\Web-main\Front"),
        (Join-Path $Root "Web-main\Front")
    )
    $frontDir = $null
    foreach ($candidate in $frontCandidates) {
        $resolved = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue
        if ($resolved) {
            $frontDir = $resolved
            break
        }
    }

    if ($frontDir) {
        Start-ProjectWindow `
            -Title "Cooking Agent Front" `
            -WorkingDirectory $frontDir.Path `
            -CommandTail "abc.py" `
            -Python $python
        Write-Host "Frontend URL: http://127.0.0.1:$FrontPort"
    }
    else {
        Write-Warning "Frontend directory was not found."
    }
}

Write-Host "Started requested processes."
