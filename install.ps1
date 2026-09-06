# llmash installer
#
#   irm https://raw.githubusercontent.com/omgitsbase/llmash/main/install.ps1 | iex
#
#   -Uninstall        remove everything the installer created (models are kept)
#   -NoOllama         do not shadow the `ollama` command, do not stop Ollama
#   -NoStartup        do not start llmash at login
#   -Runtime <kind>   llama.cpp build to fetch: auto (default), cuda, vulkan, cpu, none
#   -Dir <path>       install somewhere other than %ProgramData%\llmash
#   -Mbps <n>         cap the download at n megabits per second (0 for no cap)
#   -Yes              answer yes to every prompt

[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$NoOllama,
    [switch]$NoStartup,
    [ValidateSet('auto', 'cuda', 'vulkan', 'cpu', 'none')]
    [string]$Runtime = 'auto',
    [string]$Dir,
    [double]$Mbps = 20,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Repo = if ($env:LLMASH_REPO) { $env:LLMASH_REPO } else { 'omgitsbase/llmash' }
$Asset = 'llmash-win-x64.zip'

$Startup   = [Environment]::GetFolderPath('Startup')
$Lnk       = Join-Path $Startup 'llmash.lnk'
$RegKey    = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\llmash'
$Shims     = @('llmash')
$Releases  = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10'

function Say  ($m) { Write-Host "  $m" }
function Step ($m) { Write-Host ''; Write-Host "> $m" -ForegroundColor Cyan }
function Good ($m) { Write-Host "  $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "  $m" -ForegroundColor Yellow }
function Die  ($m) { Write-Host ''; Write-Host "  $m" -ForegroundColor Red; exit 1 }

function Native ($exe, [string[]]$a) {
    $ErrorActionPreference = 'Continue'
    $out = & $exe @a 2>&1
    $code = $LASTEXITCODE
    $text = @($out | Where-Object { $_ -is [string] })
    return @{ code = $code; out = $text; all = @($out | ForEach-Object { "$_" }) }
}

function MB ($bytes) { return [math]::Round($bytes / 1MB) }

function Size ($bytes) {
    if ($bytes -lt 1MB)   { return "$([math]::Round($bytes / 1KB)) KB" }
    if ($bytes -lt 100MB) { return ('{0:N1} MB' -f ($bytes / 1MB)) }
    if ($bytes -lt 1GB)   { return "$([math]::Round($bytes / 1MB)) MB" }
    return ('{0:N2} GB' -f ($bytes / 1GB))
}

function Bar ($label, $done, $total, $rate) {
    $w = 28
    $done = [long]$done; $total = [long]$total; $rate = [double]$rate
    $known = $total -gt 0
    $pct = if ($known) { [math]::Min(100, [math]::Floor(100 * $done / $total)) } else { 0 }
    $fill = if ($known) { [math]::Floor($w * $pct / 100) } else { 0 }
    $bar = ([string][char]0x2588) * $fill + ' ' * ($w - $fill)
    $pctText = if ($known) { '{0,3}%' -f $pct } else { '    ' }
    $size = if ($known) { "$(Size $done) / $(Size $total)" } else { Size $done }
    $speed = if ($rate -gt 0) { "  $(Size $rate)/s" } else { '' }
    $left = if ($rate -gt 0 -and $known -and $done -lt $total) { "  $([math]::Ceiling(($total - $done) / $rate))s left" } else { '' }
    $line = "  {0,-38} {1} {2}{3}{4}  {5}{6}" -f $label, $pctText, [char]0x2595, $bar, [char]0x258F, $size, ($speed + $left)
    return $line.PadRight(110)
}

function Download ($url, $dest, $label) {
    $req = [System.Net.HttpWebRequest]::Create($url)
    $req.UserAgent = 'llmash-installer'
    $req.Timeout = 60000
    $resp = $req.GetResponse()
    $total = $resp.ContentLength
    $in = $resp.GetResponseStream()
    $out = [System.IO.File]::Create($dest)
    $buf = New-Object byte[] (1MB)
    $done = 0L; $t0 = Get-Date; $tick = $t0; $tickDone = 0L; $rate = 0.0
    try {
        Write-Host -NoNewline ("`r" + (Bar $label 0 $total 0))
        while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
            $out.Write($buf, 0, $n)
            $done += $n
            if ($Mbps -gt 0) {
                $target = $done / ($Mbps * 1000000 / 8)
                $ahead  = $target - ((Get-Date) - $t0).TotalSeconds
                if ($ahead -gt 0.01) { Start-Sleep -Milliseconds ([int][math]::Min(500, $ahead * 1000)) }
            }
            $now = Get-Date
            $dt = ($now - $tick).TotalSeconds
            if ($dt -ge 0.25) {
                $inst = ($done - $tickDone) / $dt
                $rate = if ($rate -eq 0) { $inst } else { 0.7 * $rate + 0.3 * $inst }
                $tick = $now; $tickDone = $done
                Write-Host -NoNewline ("`r" + (Bar $label $done $total $rate))
            }
        }
    } finally {
        $out.Close(); $in.Close(); $resp.Close()
    }
    $secs = [math]::Max(0.1, ((Get-Date) - $t0).TotalSeconds)
    Write-Host ("`r" + (Bar $label $done $done ($done / $secs)).TrimEnd())
}

# ---------------------------------------------------------------- location
if ($Dir) {
    $Root = $Dir
} else {
    $Root = Join-Path $env:ProgramData 'llmash'
    try {
        New-Item -ItemType Directory -Force $Root | Out-Null
        $probe = Join-Path $Root '.write-test'
        Set-Content $probe 'ok'; Remove-Item $probe
    } catch {
        $Root = Join-Path $env:LOCALAPPDATA 'llmash'
    }
}
$BinDir  = Join-Path $Root 'bin'
$RtDir   = Join-Path $Root 'runtime'
$RtExe   = Join-Path $RtDir 'llama-server.exe'
$exe     = Join-Path $Root 'llmash.exe'
$exew    = Join-Path $Root 'llmashw.exe'

Write-Host ''
Write-Host '  llmash' -ForegroundColor Cyan -NoNewline
Write-Host '  -  Ollama-compatible, less RAM, faster on llama.cpp'

# ---------------------------------------------------------------- uninstall
if ($Uninstall) {
    if (Test-Path $exe) {
        $r = Native $exe @('uninstall'); $r.all | ForEach-Object { Write-Host $_ }; return
    }
    Step 'Removing'
    foreach ($n in $Shims + @('ollama', 'llamash')) {
        $f = Join-Path $BinDir "$n.cmd"
        if (Test-Path $f) { Remove-Item $f -Force; Say "removed $n" }
    }
    if (Test-Path $Lnk) {
        $target = (New-Object -ComObject WScript.Shell).CreateShortcut($Lnk).TargetPath
        if ($target -like "$Root*") { Remove-Item $Lnk -Force; Say 'removed the startup entry' }
        else { Say "left the startup entry alone (it starts $target)" }
    }
    $disabled = Join-Path $Startup 'Ollama.lnk.disabled'
    if (Test-Path $disabled) { Move-Item $disabled (Join-Path $Startup 'Ollama.lnk') -Force; Say "restored Ollama's startup entry" }
    if (Test-Path $RegKey) { Remove-Item $RegKey -Recurse -Force; Say 'removed from Settings > Apps' }
    Good "done. $Root left in place; delete it by hand if you want it gone."
    return
}

function Ask ($question) {
    if ($Yes) { return $true }
    if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        while ($true) {
            Write-Host "  $question " -NoNewline
            Write-Host '[y/n]' -ForegroundColor Cyan -NoNewline
            Write-Host ' ' -NoNewline
            switch ([Console]::ReadKey($true).KeyChar) {
                'y' { Write-Host 'y'; return $true }
                'n' { Write-Host 'n'; return $false }
            }
            Write-Host ''
        }
    }
    Say "$question  assuming yes (nothing is reading the keyboard)"
    return $true
}

# ------------------------------------------------------------------ sources
# An install somewhere other than here is offered for removal, so two copies
# do not end up fighting over the port and the PATH. -Dir means the location
# was chosen deliberately, so nothing outside it is touched.
$elsewhere = @()
if (-not $Dir) {
    $elsewhere = @((Join-Path $env:ProgramData 'llmash'), (Join-Path $env:LOCALAPPDATA 'llmash')) |
        Select-Object -Unique |
        Where-Object { $_ -ne $Root -and (Test-Path (Join-Path $_ 'llmashw.exe')) }
}

foreach ($old in $elsewhere) {
    $v = 'an unknown version'
    if (Test-Path (Join-Path $old 'VERSION')) { $v = (Get-Content (Join-Path $old 'VERSION') -Raw).Trim() }
    Write-Host ''
    Warn "llmash $v is already installed in $old"
    if (Ask "Remove it before installing here?") {
        $oldExe = Join-Path $old 'llmash.exe'
        if (Test-Path $oldExe) { $r = Native $oldExe @('uninstall'); $r.all | ForEach-Object { Say $_ } }
        Remove-Item $old -Recurse -Force -EA SilentlyContinue
        Good "removed $old"
    } else {
        Say 'left it alone; the copy installed last is the one on PATH'
    }
}

$upgrade = Test-Path (Join-Path $Root 'llmashw.exe')
Step $(if ($upgrade) { 'Upgrading llmash' } else { 'Fetching llmash' })
New-Item -ItemType Directory -Force $Root   | Out-Null
New-Item -ItemType Directory -Force $BinDir | Out-Null

if ($upgrade) {
    Get-CimInstance Win32_Process -Filter "Name='llmashw.exe' OR Name='llmash.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$Root\llmashw.exe*" -or ($_.CommandLine -like "*$Root\llmash.exe*" -and $_.CommandLine -like "* serve*") } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
    Start-Sleep -Milliseconds 500

    Get-ChildItem $Root, $BinDir -Filter '*.old-*' -File -EA SilentlyContinue |
        Remove-Item -Force -EA SilentlyContinue
    foreach ($f in @(Get-ChildItem $Root, $BinDir -Filter '*.exe' -File -EA SilentlyContinue)) {
        try {
            $h = [IO.File]::Open($f.FullName, 'Open', 'ReadWrite', 'None')
            $h.Close()
        } catch {
            Move-Item $f.FullName "$($f.FullName).old-$(Get-Random)" -Force -EA SilentlyContinue
        }
    }
}

$zip = Join-Path $env:TEMP $Asset
try {
    $rel = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest" `
        -Headers @{ 'User-Agent' = 'llmash-installer'; Accept = 'application/vnd.github+json' } -TimeoutSec 30
} catch {
    Die "could not read the latest release of $Repo  ($($_.Exception.Message))"
}
$url = ($rel.assets | Where-Object { $_.name -eq $Asset } | Select-Object -First 1).browser_download_url
if (-not $url) { Die "release $($rel.tag_name) has no $Asset" }
Say "release $($rel.tag_name)"
try {
    Download $url $zip $Asset
} catch {
    Die "download failed  ($($_.Exception.Message))"
}
Expand-Archive -Path $zip -DestinationPath $Root -Force
Remove-Item $zip -Force
Good "unpacked into $Root"

$version = '0.0.0'
if (Test-Path (Join-Path $Root 'VERSION')) { $version = (Get-Content (Join-Path $Root 'VERSION') -Raw).Trim() }
if (-not (Test-Path $exew) -or -not (Test-Path $exe)) {
    Die "that release asset is not a compiled build: llmash.exe / llmashw.exe are missing"
}
Good "llmash $version, nothing else to install"

# ----------------------------------------------------------------- llama.cpp
Step 'Checking llama.cpp'
$rtInfo = $null
$have = (Test-Path $RtExe) -or (Get-Command llama-server -ErrorAction SilentlyContinue) -or $env:LLAMA_BIN
if ($Runtime -eq 'none') {
    if ($have) { Good 'found' } else { Warn ('skipped; set LLAMA_BIN or drop llama-server.exe into ' + $RtDir) }
} elseif ($have -and $Runtime -eq 'auto') {
    Good 'found'
} else {
    $kind = $Runtime
    $driver = $null
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) { $smi = $smi.Source } elseif (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe") { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
    if ($smi) {
        $r = Native $smi @('--query-gpu=driver_version', '--format=csv,noheader')
        if ($r.code -eq 0 -and $r.out.Count) { $driver = $r.out[0].Trim() }
    }
    if ($kind -eq 'auto') {
        if ($driver) {
            $kind = 'cuda'
        } else {
            $gpu = Get-CimInstance Win32_VideoController -EA SilentlyContinue |
                   Where-Object { $_.Name -notlike '*Microsoft Basic*' -and $_.Name -notlike '*Remote*' }
            $kind = if ($gpu) { 'vulkan' } else { 'cpu' }
        }
    }
    $cudaMajor = '13'
    if ($kind -eq 'cuda' -and $driver) {
        try { if (([version]($driver -replace '[^0-9.].*$', '')).Major -lt 580) { $cudaMajor = '12' } } catch {}
    }
    $pattern = switch ($kind) {
        'cuda'   { "^llama-.*-bin-win-cuda-$cudaMajor\.[0-9]+-x64\.zip$" }
        'vulkan' { '^llama-.*-bin-win-vulkan-x64\.zip$' }
        'cpu'    { '^llama-.*-bin-win-cpu-x64\.zip$' }
    }
    $label = if ($kind -eq 'cuda') { "cuda $cudaMajor (driver $driver)" } else { $kind }
    Say "this machine gets the $label build"

    $rel = $null; $asset = $null; $cudart = $null
    try {
        foreach ($cand in (Invoke-RestMethod $Releases -Headers @{ 'User-Agent' = 'llmash-installer' } -TimeoutSec 30)) {
            $asset = $cand.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
            if ($asset) { $rel = $cand; break }
        }
    } catch {
        Warn "could not read llama.cpp's releases ($($_.Exception.Message))"
    }
    if (-not $asset) {
        Warn 'no matching llama.cpp build found; get one from https://github.com/ggml-org/llama.cpp/releases'
        Say  "and unzip it into  $RtDir"
    } else {
        if ($kind -eq 'cuda') {
            $haveCudart = Get-Command "cublas64_$cudaMajor.dll" -ErrorAction SilentlyContinue
            if (-not $haveCudart) {
                $cudart = $rel.assets | Where-Object { $_.name -match "^cudart-llama-bin-win-cuda-$cudaMajor\.[0-9]+-x64\.zip$" } | Select-Object -First 1
            }
        }
        $total = $asset.size + $(if ($cudart) { $cudart.size } else { 0 })
        Say "llama.cpp $($rel.tag_name), about $(MB $total) MB"
        if (Test-Path $RtDir) { Remove-Item $RtDir -Recurse -Force }
        New-Item -ItemType Directory -Force $RtDir | Out-Null
        try {
            foreach ($a in @($asset, $cudart) | Where-Object { $_ }) {
                $tmp = Join-Path $env:TEMP $a.name
                Download $a.browser_download_url $tmp $a.name
                Expand-Archive -Path $tmp -DestinationPath $RtDir -Force
                Remove-Item $tmp -Force
            }
            if (-not (Test-Path $RtExe)) {
                $inner = Get-ChildItem $RtDir -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
                if ($inner) {
                    Get-ChildItem $inner.DirectoryName | Move-Item -Destination $RtDir -Force
                    Get-ChildItem $RtDir -Directory | Where-Object { -not (Get-ChildItem $_.FullName -Recurse -File) } | Remove-Item -Recurse -Force
                }
            }
            if (Test-Path $RtExe) {
                $rtInfo = @{ tag = $rel.tag_name; kind = $kind; asset = $asset.name }
                Good "llama.cpp $($rel.tag_name) ($kind) in $RtDir"
            } else {
                Warn 'the download did not contain llama-server.exe'
            }
        } catch {
            Warn "download failed ($($_.Exception.Message))"
            Say  "get $($asset.name) from https://github.com/ggml-org/llama.cpp/releases and unzip it into  $RtDir"
        }
    }
}

@{ version = $version; repo = $Repo; installed = (Get-Date).ToString('o'); runtime = $rtInfo } |
    ConvertTo-Json | Set-Content (Join-Path $Root 'install.json') -Encoding UTF8

# --------------------------------------------------------------------- shims
Step 'Installing commands'
$names = $Shims
if (-not $NoOllama) { $names += 'ollama' }
foreach ($n in $names) {
    Remove-Item (Join-Path $BinDir "$n.cmd") -Force -EA SilentlyContinue
    Copy-Item $exe (Join-Path $BinDir "$n.exe") -Force
    Say $n
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable('Path', "$BinDir;$userPath", 'User')
    $env:Path = "$BinDir;$env:Path"
    Good 'added to PATH'
} else {
    Good 'already on PATH'
}

# --------------------------------------------------------- Settings > Apps
New-Item -Path $RegKey -Force | Out-Null
$reg = @{
    DisplayName     = 'llmash'
    DisplayVersion  = $version
    Publisher       = 'llmash'
    InstallLocation = $Root
    InstallDate     = (Get-Date).ToString('yyyyMMdd')
    UninstallString = "`"$exe`" uninstall"
    DisplayIcon     = "$Root\llmash.ico"
    URLInfoAbout    = $Origin
    NoModify        = 1
    NoRepair        = 1
}
foreach ($k in $reg.Keys) {
    $t = if ($reg[$k] -is [int]) { 'DWord' } else { 'String' }
    Set-ItemProperty -Path $RegKey -Name $k -Value $reg[$k] -Type $t
}
$size = [int]((Get-ChildItem $Root -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum / 1KB)
Set-ItemProperty -Path $RegKey -Name EstimatedSize -Value $size -Type DWord
Good 'registered in Settings > Apps'

# ------------------------------------------------------------------- startup
if (-not $NoStartup) {
    Step 'Starting at login'
    $ws = New-Object -ComObject WScript.Shell
    $s  = $ws.CreateShortcut($Lnk)
    $s.TargetPath       = $exew
    $s.Arguments        = 'tray'
    $s.WorkingDirectory = $Root
    $s.WindowStyle      = 7
    $s.IconLocation     = "$Root\llmash.ico"
    $s.Description      = 'llmash'
    $s.Save()
    Good 'enabled, silent, with the tray icon'

} else {
    Set-Content (Join-Path $Root 'tray.json') '{"startup": false}' -Encoding ASCII
}

# ---------------------------------------------------------------------- ollama
# Both want port 11434, so only one of them can be the server.
$ollamaLnk = Join-Path $Startup 'Ollama.lnk'
$ollamaProcs = @(Get-Process -Name 'ollama', 'ollama app', 'ollama_llama_server' -ErrorAction SilentlyContinue)
$ollamaAtLogin = Test-Path $ollamaLnk
if (-not $NoOllama -and ($ollamaProcs.Count -or $ollamaAtLogin)) {
    Step 'Ollama'
    if ($ollamaProcs.Count) { Say 'Ollama is running' }
    if ($ollamaAtLogin)     { Say 'Ollama starts at login' }
    Say 'llmash serves the same API on the same port, so they cannot both run.'
    if (Ask 'Close Ollama and take it off startup?') {
        if ($ollamaProcs.Count) {
            $ollamaProcs | Stop-Process -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
            Good 'closed Ollama'
        }
        if ($ollamaAtLogin) {
            Move-Item $ollamaLnk "$ollamaLnk.disabled" -Force
            Good 'took Ollama off startup (put back by llmash uninstall)'
        }
    } else {
        Warn 'left Ollama alone; whichever starts first will hold port 11434'
    }
}

# --------------------------------------------------------------------- start
Step 'Starting llmash'
function PortBusy { try { $null = Invoke-WebRequest 'http://127.0.0.1:11434/' -UseBasicParsing -TimeoutSec 2; return $true } catch { return $false } }
$busy = PortBusy
if ($busy) {
    if ($upgrade) {
        Warn 'something else is already on 11434 (an older llmash?); restart it with:  llmash tray'
    } else {
        Warn 'port 11434 is already in use, not starting a second server'
        Say  'stop whatever holds it, then run:  llmash tray'
    }
} else {
    Start-Process -FilePath $exew -ArgumentList 'tray' -WorkingDirectory $Root -WindowStyle Hidden
    $ok = $false
    foreach ($i in 1..40) {
        Start-Sleep -Milliseconds 400
        if (PortBusy) { $ok = $true; break }
    }
    if ($ok) { Good 'listening on 127.0.0.1:11434 - look for the icon in the tray' } else { Warn 'not answering yet; try:  llmash tray' }
}

Write-Host ''
Write-Host "  llmash $version is ready." -ForegroundColor Green
Write-Host ''
Write-Host '    llmash install qwen3.6:a3b'
Write-Host '    llmash run qwen3.6:a3b'
Write-Host '    llmash list'
Write-Host '    llmash tray        (if you closed it)'
Write-Host ''
if (-not $NoOllama) { Say '`ollama` now runs llmash too, so anything already pointed at Ollama keeps working.' }
Say 'Open a new terminal to pick up the PATH change.   Remove with:  llmash uninstall'
Write-Host ''
