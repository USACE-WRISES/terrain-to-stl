[CmdletBinding()]
param(
    [string]$BundleVersion = 'dev',
    [string]$PythonVersion = '3.11.9',
    [string]$BundleName = 'terrain-to-stl-desktop-windows-x64',
    [string]$OutputRoot = 'dist\desktop'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function New-CleanDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }

    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Resolve-BuildPython {
    param([Parameter(Mandatory = $true)][string]$RequiredVersion)

    $majorMinor = ($RequiredVersion -split '\.')[0..1] -join '.'
    $candidates = @()

    $venvPython = Join-Path (Join-Path $repoRoot '.venv') 'Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) {
        $candidates += $venvPython
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        $resolved = & py "-$majorMinor" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $candidates += $resolved.Trim()
        }
    }

    $defaultPython = (Get-Command python -ErrorAction Stop).Source
    $candidates += $defaultPython

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        $version = & $candidate -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or $version.Trim() -ne $majorMinor) {
            continue
        }

        & $candidate -c "import pip" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }

    throw "Could not find a Python $majorMinor interpreter to vendor dependencies for the Python $RequiredVersion desktop runtime."
}

function Write-DesktopLauncher {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ScriptRelativePath,
        [string]$RuntimeExecutable = 'python.exe'
    )

    @(
        '@echo off'
        'setlocal'
        'set "APP_ROOT=%~dp0"'
        "%APP_ROOT%$RuntimeExecutable ""%APP_ROOT%$ScriptRelativePath"" %*"
        'endlocal'
    ) | Set-Content -LiteralPath $Path -Encoding Ascii
}

function Assert-FileContainsMarkers {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Markers,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description was not found at $Path"
    }

    $content = Get-Content -LiteralPath $Path -Raw
    foreach ($marker in $Markers) {
        if ($content -notmatch [regex]::Escape($marker)) {
            throw "$Description at $Path is missing required marker: $marker"
        }
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputDir = Join-Path $repoRoot $OutputRoot
$downloadDir = Join-Path $outputDir '_downloads'
$bundleRoot = Join-Path $outputDir $BundleName
$appDir = Join-Path $bundleRoot 'app'
$libDir = Join-Path $bundleRoot 'Lib'
$sitePackagesDir = Join-Path $libDir 'site-packages'
$exampleDir = Join-Path $repoRoot 'webapp\public\example'
$smokeOutputPath = Join-Path $outputDir 'smoke-example-output.stl'
$archivePath = Join-Path $outputDir "$BundleName.zip"
$hashPath = "$archivePath.sha256"
$pythonEmbedUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$pythonEmbedZip = Join-Path $downloadDir "python-$PythonVersion-embed-amd64.zip"
$bundlePython = Join-Path $bundleRoot 'python.exe'
$desktopGuiLauncherPath = Join-Path $bundleRoot 'Run Terrain-to-STL GUI.cmd'
$retiredLauncherPaths = @(
    (Join-Path $bundleRoot 'Run Console Converter.cmd'),
    (Join-Path $bundleRoot 'Run Desktop GUI.cmd'),
    (Join-Path $bundleRoot 'Run Terrain to STL.cmd'),
    (Join-Path $bundleRoot 'Run STL Viewer.cmd')
)
$appFiles = @(
    'desktop_backend.py',
    'desktop_console.py',
    'desktop_gui.py',
    'desktop_gui_support.py',
    'terrain_to_stl.py',
    'terrain_hdf_to_stl.py',
    'mesh_viewer.py',
    'requirements.txt'
)

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
New-CleanDirectory -Path $bundleRoot
New-Item -ItemType Directory -Path $appDir -Force | Out-Null
New-Item -ItemType Directory -Path $sitePackagesDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $pythonEmbedZip)) {
    Write-Host "Downloading Python embeddable runtime $PythonVersion..."
    Invoke-WebRequest -Uri $pythonEmbedUrl -OutFile $pythonEmbedZip
}

Write-Host "Extracting Python embeddable runtime..."
Expand-Archive -LiteralPath $pythonEmbedZip -DestinationPath $bundleRoot -Force

$buildPython = Resolve-BuildPython -RequiredVersion $PythonVersion
Write-Host "Using build Python: $buildPython"
Invoke-External $buildPython -m pip install --upgrade pip
Invoke-External $buildPython -m pip install --upgrade --target $sitePackagesDir -r (Join-Path $repoRoot 'requirements.txt')

foreach ($file in $appFiles) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $file) -Destination (Join-Path $appDir $file) -Force
}

Assert-FileContainsMarkers `
    -Path (Join-Path $appDir 'desktop_gui.py') `
    -Description 'Bundled desktop GUI source' `
    -Markers @(
        'class ViewerLoadWorker',
        'loading_overlay',
        '_start_async_load'
    )
Assert-FileContainsMarkers `
    -Path (Join-Path $appDir 'mesh_viewer.py') `
    -Description 'Bundled mesh viewer source' `
    -Markers @(
        'inspect_binary_stl',
        'build_preview_mesh_from_binary_stl'
    )

Copy-Item -LiteralPath (Join-Path $repoRoot 'bundle-assets\windows-desktop\README.txt') -Destination (Join-Path $bundleRoot 'README.txt') -Force
Write-DesktopLauncher -Path $desktopGuiLauncherPath -ScriptRelativePath 'app\desktop_gui.py' -RuntimeExecutable 'pythonw.exe'
foreach ($retiredLauncherPath in $retiredLauncherPaths) {
    if (Test-Path -LiteralPath $retiredLauncherPath) {
        throw "Unexpected retired launcher present in bundle output: $retiredLauncherPath"
    }
}

$pthFile = Get-ChildItem -LiteralPath $bundleRoot -Filter 'python*._pth' | Select-Object -First 1
if ($null -eq $pthFile) {
    throw 'Could not find the embeddable Python ._pth file.'
}

$zipEntry = Get-ChildItem -LiteralPath $bundleRoot -Filter 'python*.zip' | Select-Object -First 1
if ($null -eq $zipEntry) {
    throw 'Could not find the embeddable Python standard-library zip.'
}

@(
    $zipEntry.Name
    '.'
    'app'
    'Lib'
    'Lib/site-packages'
    'import site'
) | Set-Content -LiteralPath $pthFile.FullName -Encoding Ascii

if (Test-Path -LiteralPath $smokeOutputPath) {
    Remove-Item -LiteralPath $smokeOutputPath -Force
}

Write-Host 'Running desktop bundle smoke tests...'
Invoke-External $bundlePython -c "import desktop_backend, desktop_console, desktop_gui, desktop_gui_support, terrain_to_stl, mesh_viewer; print('Desktop bundle import smoke test passed.')"
Invoke-External $bundlePython `
    (Join-Path $appDir 'desktop_console.py') `
    convert `
    --input (Join-Path $exampleDir 'example.hdf') `
    --top-elevation 100 `
    --sample-step 8 `
    --output $smokeOutputPath

if (-not (Test-Path -LiteralPath $smokeOutputPath)) {
    throw "Smoke-test conversion did not produce $smokeOutputPath"
}

if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
if (Test-Path -LiteralPath $hashPath) {
    Remove-Item -LiteralPath $hashPath -Force
}

Write-Host "Creating desktop archive $archivePath..."
Compress-Archive -Path (Join-Path $bundleRoot '*') -DestinationPath $archivePath -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([System.IO.Path]::GetFileName($archivePath))" | Set-Content -LiteralPath $hashPath -Encoding Ascii

Write-Host "Bundle version: $BundleVersion"
Write-Host "Desktop bundle ready: $archivePath"
Write-Host "SHA256 file ready: $hashPath"
