# Runs the built binaries the way an install does: from the `bin` copy the
# installer puts on PATH. Two shipped releases broke there and not in the
# tests, because the tests never ran the program from that folder.
#
#   python build.py ; .\cpp\smoke-install.ps1
#
# ASCII only in here; a stray em-dash broke the parser once.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $repo 'dist\llmash'
if (-not (Test-Path (Join-Path $dist 'llmash.exe'))) { Write-Host 'build first: python build.py'; exit 2 }

$root = Join-Path ([System.IO.Path]::GetTempPath()) ("llmash-smoke-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
$bin  = Join-Path $root 'bin'
New-Item -ItemType Directory -Force $bin | Out-Null
Copy-Item (Join-Path $dist '*') $root
Copy-Item (Join-Path $dist 'llmash.exe') $bin
Copy-Item (Join-Path $dist '*.dll') $bin
'{"models_root": "", "pin": ["smoke-marker"]}' | Set-Content (Join-Path $root 'local.json') -Encoding ASCII

$fails = 0
function Check($ok, $what) {
    if ($ok) { Write-Host "  pass  $what" -ForegroundColor Green }
    else { Write-Host "  FAIL  $what" -ForegroundColor Red; $script:fails++ }
}

$env:LLMASH_PORT = '1'          # nothing is listening, so no server is touched
$env:LLMASH_PUBLIC_PORT = '1'

Write-Host ''
Write-Host "install at $root"
foreach ($from in @(@{n = 'root'; p = Join-Path $root 'llmash.exe' }, @{n = 'bin'; p = Join-Path $bin 'llmash.exe' })) {
    Write-Host ''
    Write-Host "> run from $($from.n)" -ForegroundColor Cyan
    $out = & $from.p 'doctor' 2>&1 | Out-String
    $line = ($out -split "`n" | Where-Object { $_ -match 'install' } | Select-Object -First 1)
    Check ($line -match [regex]::Escape($root) -and $line -notmatch 'bin') "doctor reports the install, not bin:$line"
    Check ($out -notmatch 'reinstall llmash') 'doctor asks for no reinstall'

    $ver = & $from.p '--version' 2>&1 | Out-String
    Check ($ver -match 'version is') "--version answers:  $($ver.Trim())"

    # `tray` starts the background program, so only its own resolution is
    # checked: the message it gives when it cannot find one.
    $tray = & $from.p 'tray' 2>&1 | Out-String
    Check ($tray -notmatch 'was not found') "tray finds llmashw.exe:$tray"
    Get-CimInstance Win32_Process -Filter "Name='llmashw.exe'" |
        Where-Object { $_.ExecutablePath -like "$root*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Remove-Item $root -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ''
if ($fails) { Write-Host "$fails check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host 'the installed layout works from both copies' -ForegroundColor Green
