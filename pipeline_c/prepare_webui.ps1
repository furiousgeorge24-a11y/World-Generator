[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 5002,

    [Parameter(Mandatory = $true)]
    [string]$ServerScript,

    [Parameter(Mandatory = $true)]
    [string]$Backend,

    [Parameter(Mandatory = $true)]
    [string]$Root,

    [switch]$InspectOnly
)

$ErrorActionPreference = "Stop"

$serverPath = (Resolve-Path -LiteralPath $ServerScript -ErrorAction Stop).Path
$rootPath = (Resolve-Path -LiteralPath $Root -ErrorAction Stop).Path

function Get-ExactTokenPattern([string]$Value) {
    $escaped = [Regex]::Escape($Value)
    if ($Value -match "\s") {
        return '"' + $escaped + '"'
    }
    '(?:"' + $escaped + '"|' + $escaped + ')'
}

function Get-ListenerProcessIds {
    $netstatPath = Join-Path $env:SystemRoot "System32\netstat.exe"
    $listenerPattern = "^\s*TCP\s+\S+:" +
        [Regex]::Escape([string]$Port) +
        "\s+\S+\s+LISTENING\s+([0-9]+)\s*$"
    $processIds = @(
        & $netstatPath -ano -p tcp | ForEach-Object {
            if ($_ -match $listenerPattern) {
                [int]$Matches[1]
            }
        }
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect TCP listeners before starting the Pipeline C WebUI."
    }
    @($processIds | Where-Object { $_ -gt 0 } | Sort-Object -Unique)
}

$listenerProcessIds = @(Get-ListenerProcessIds)
if ($listenerProcessIds.Count -eq 0) {
    return
}

$serverPattern = "(?i)(?:^|\s)" +
    (Get-ExactTokenPattern $serverPath) + "(?=\s|$)"
$rootPattern = "(?i)(?:^|\s)--root(?:\s+|=)" +
    (Get-ExactTokenPattern $rootPath) + "(?=\s|$)"
$legacyServerPattern = "(?i)(?:^|\s)" +
    (Get-ExactTokenPattern "..\webui\serve.py") + "(?=\s|$)"
$legacyRootPattern = "(?i)(?:^|\s)--root(?:\s+|=)" +
    (Get-ExactTokenPattern ".") + "(?=\s|$)"
$backendPattern = "(?i)(?:^|\s)--backend(?:\s+|=)" +
    (Get-ExactTokenPattern $Backend) + "(?=\s|$)"
$portPattern = "(?i)(?:^|\s)--port(?:\s+|=)" +
    (Get-ExactTokenPattern ([string]$Port)) + "(?=\s|$)"

# Before this launcher used absolute arguments, run.bat used the exact relative
# pair "..\webui\serve.py" and "--root .". A live legacy listener is accepted
# only when its registry also proves it is this Pipeline C inspection backend.
$legacyRegistryConfirmed = $false
try {
    $registry = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/registry" -TimeoutSec 2
    $caseIds = @($registry.inspection_cases | ForEach-Object { $_.id })
    $legacyRegistryConfirmed =
        $registry.name -eq "pipeline_c land-origin lab" -and
        $registry.default_mode -eq "inspection" -and
        $caseIds -contains "c5-c02-development-cohort-v1"
} catch {
    $legacyRegistryConfirmed = $false
}

function Test-SameWebUiProcess($ProcessRecord, [switch]$AllowLauncher) {
    $processName = [string]$ProcessRecord.Name
    $commandLine = [string]$ProcessRecord.CommandLine
    $isPython = $processName -match "(?i)^python(?:w)?(?:[0-9.]+)?\.exe$"
    $isLauncher = $AllowLauncher -and $processName -match "(?i)^py\.exe$"
    $hasCurrentIdentity =
        $commandLine -match $serverPattern -and
        $commandLine -match $rootPattern
    $hasLegacyIdentity =
        $legacyRegistryConfirmed -and
        $commandLine -match $legacyServerPattern -and
        $commandLine -match $legacyRootPattern
    ($isPython -or $isLauncher) -and
        ($hasCurrentIdentity -or $hasLegacyIdentity) -and
        $commandLine -match $backendPattern -and
        $commandLine -match $portPattern
}

$killRoots = @()

foreach ($ownerProcessId in $listenerProcessIds) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerProcessId"
    if ($null -eq $owner) {
        continue
    }

    if (-not (Test-SameWebUiProcess $owner)) {
        throw (
            "Port $Port is owned by PID $ownerProcessId ($($owner.Name)), " +
            "which is not the same Pipeline C WebUI server. Nothing was terminated."
        )
    }

    $killRoot = $owner
    while ([int]$killRoot.ParentProcessId -gt 0) {
        $parentProcessId = [int]$killRoot.ParentProcessId
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $parentProcessId"
        if ($null -eq $parent -or -not (
            Test-SameWebUiProcess $parent -AllowLauncher
        )) {
            break
        }
        $killRoot = $parent
    }
    $killRoots += [int]$killRoot.ProcessId
}

$uniqueKillRoots = @($killRoots | Sort-Object -Unique)
if ($InspectOnly) {
    Write-Host (
        "Matched Pipeline C WebUI listener process tree root(s): " +
        ($uniqueKillRoots -join ", ")
    )
    return
}

$taskkillPath = Join-Path $env:SystemRoot "System32\taskkill.exe"
foreach ($killRootProcessId in $uniqueKillRoots) {
    Write-Host "Stopping stale Pipeline C WebUI process tree $killRootProcessId on port $Port..."
    $taskkillOutput = @(& $taskkillPath /PID $killRootProcessId /T /F 2>&1)
    if ($LASTEXITCODE -ne 0 -and $null -ne (
        Get-Process -Id $killRootProcessId -ErrorAction SilentlyContinue
    )) {
        throw (
            "Could not terminate stale Pipeline C WebUI process tree " +
            "${killRootProcessId}: $($taskkillOutput -join ' ')"
        )
    }
}

$deadline = [DateTime]::UtcNow.AddSeconds(5)
do {
    $remaining = @(Get-ListenerProcessIds)
    if ($remaining.Count -eq 0) {
        return
    }
    Start-Sleep -Milliseconds 100
} while ([DateTime]::UtcNow -lt $deadline)

throw "The stale Pipeline C WebUI process did not release port $Port within 5 seconds."
