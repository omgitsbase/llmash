#!/bin/sh
# llmash installer for Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/omgitsbase/llmash/main/install.sh | sh
#
#   --dir <path>    install somewhere other than /usr/local
#   --user          install for this user only, into ~/.local, no service
#   --no-service    skip the systemd unit
#   --no-ollama     do not take over the `ollama` command
#   --runtime <k>   llama.cpp build to fetch: auto, vulkan, rocm, cpu, none
#   --dry-run       say what it would do, download nothing
#   --uninstall     remove what this installed (models are kept)
#   --yes           answer yes to every prompt

set -eu

REPO=${LLMASH_REPO:-omgitsbase/llmash}
PREFIX=/usr/local
SERVICE=1
SYSTEM=1
SHADOW_OLLAMA=1
RUNTIME=auto
UNINSTALL=0
DOWNLOAD=1

while [ $# -gt 0 ]; do
    case $1 in
        --dir) PREFIX=$2; shift 2 ;;
        --user) SYSTEM=0; PREFIX=$HOME/.local; shift ;;
        --no-service) SERVICE=0; shift ;;
        --no-ollama) SHADOW_OLLAMA=0; shift ;;
        --runtime) RUNTIME=$2; shift 2 ;;
        --dry-run) DOWNLOAD=0; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --yes|-y) shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

say()  { printf '  %s\n' "$1"; }
step() { printf '\n\033[36m> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m  %s\033[0m\n' "$1"; }
die()  { printf '\n\033[31m  %s\033[0m\n' "$1" >&2; exit 1; }

SUDO=
if [ "$SYSTEM" = 1 ] && [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "this needs root; install sudo or re-run with --user"
    SUDO=sudo
fi

BIN=$PREFIX/bin
LIB=$PREFIX/lib/llmash
UNIT=/etc/systemd/system/llmash.service
USER_UNIT=$HOME/.config/systemd/user/llmash.service
ROOT=$LIB

if [ "$UNINSTALL" = 1 ]; then
    step 'Removing llmash'
    if [ "$SYSTEM" = 1 ]; then
        $SUDO systemctl disable --now llmash 2>/dev/null || true
        $SUDO rm -f "$UNIT"
        $SUDO systemctl daemon-reload 2>/dev/null || true
    else
        systemctl --user disable --now llmash 2>/dev/null || true
        rm -f "$USER_UNIT"
    fi
    $SUDO rm -f "$BIN/llmash"
    [ -L "$BIN/ollama" ] && $SUDO rm -f "$BIN/ollama"
    $SUDO rm -rf "$LIB"
    say 'removed; your models were left alone'
    exit 0
fi

step 'Checking this machine'
case $(uname -s) in
    Linux) ;;
    *) die "this installer is for Linux; on Windows use install.ps1" ;;
esac
case $(uname -m) in
    x86_64|amd64) ARCH=x64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) die "no llmash build for $(uname -m)" ;;
esac
say "linux $ARCH"
command -v curl >/dev/null 2>&1 || die 'curl is needed to download llmash'
command -v tar  >/dev/null 2>&1 || die 'tar is needed to unpack llmash'

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

step 'Fetching llmash'
ASSET=llmash-linux-$ARCH.tar.gz
if [ "$DOWNLOAD" = 0 ]; then
    say "would install $ASSET into $LIB"
else
URL=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
      | sed -n 's/.*"browser_download_url": *"\([^"]*'"$ASSET"'\)".*/\1/p' | head -1)
[ -n "$URL" ] || die "the latest release of $REPO has no $ASSET"

curl -fsSL --progress-bar "$URL" -o "$TMP/$ASSET" || die 'download failed'
$SUDO mkdir -p "$BIN" "$LIB" "$ROOT"
$SUDO tar -xzf "$TMP/$ASSET" -C "$LIB"
$SUDO ln -sf "$LIB/llmash" "$BIN/llmash"
say "unpacked into $LIB"

miss=$(ldd "$LIB/llmash" 2>/dev/null | awk '/not found/{print $1}')
if [ -n "$miss" ]; then
    warn "llmash needs shared libraries this machine does not have:"
    for m in $miss; do say "    $m"; done
    say 'try one of:  apt install libcurl4   |   dnf install libcurl   |   pacman -S curl'
    die 'install stopped'
fi
"$LIB/llmash" --version >/dev/null 2>&1 || die "the downloaded llmash does not run here"
fi

if [ "$SHADOW_OLLAMA" = 1 ]; then
    if [ -e "$BIN/ollama" ] && [ ! -L "$BIN/ollama" ]; then
        warn "$BIN/ollama is a real Ollama; leaving it alone"
    elif [ "$DOWNLOAD" = 0 ]; then
        say "would also answer to \`ollama\`"
    else
        $SUDO ln -sf "$LIB/llmash" "$BIN/ollama"
        say 'also answers to `ollama`'
    fi
fi

# ------------------------------------------------------------ ollama's models
# OLLAMA_MODELS (LLMASH_MODELS is the same setting under llmash's name) points
# at either an Ollama store, which llmash reads and pulls into, or a folder of
# GGUFs, which it reads where they are. A shell's variables say nothing about a
# service's, so an installed ollama.service is read for its own.
is_store()  { [ -n "${1:-}" ] && [ -d "$1/manifests" ]; }
is_ggufs()  { [ -n "${1:-}" ] && [ -d "$1" ] && ! is_store "$1" \
              && [ -n "$(find "$1" -name '*.gguf' -print -quit 2>/dev/null)" ]; }

unit_models() {
    for f in /etc/systemd/system/ollama.service \
             /etc/systemd/system/ollama.service.d/*.conf \
             "$HOME/.config/systemd/user/ollama.service"; do
        [ -f "$f" ] || continue
        v=$(sed -n 's/^[[:space:]]*Environment=.*"\?OLLAMA_MODELS=\([^"]*\)"\?.*/\1/p' "$f" | tail -1)
        [ -n "$v" ] && { printf '%s' "$v"; return; }
    done
}

step 'Models'
OLLAMA_HOME_STORE=$HOME/.ollama/models
OLLAMA_SVC_STORE=/usr/share/ollama/.ollama/models

MODELS_VAR=${OLLAMA_MODELS:-${LLMASH_MODELS:-}}
[ -z "$MODELS_VAR" ] && MODELS_VAR=$(unit_models || true)
[ -n "$MODELS_VAR" ] && say "OLLAMA_MODELS points at $MODELS_VAR"

MODELS_ROOT=$OLLAMA_HOME_STORE
if [ -n "$MODELS_VAR" ]; then
    MODELS_ROOT=$MODELS_VAR
elif is_store "$OLLAMA_SVC_STORE"; then
    MODELS_ROOT=$OLLAMA_SVC_STORE
    say "found Ollama's service store"
fi

EXTRAS=
for cand in "$OLLAMA_HOME_STORE" "$OLLAMA_SVC_STORE"; do
    [ "$cand" = "$MODELS_ROOT" ] && continue
    is_store "$cand" && EXTRAS="$EXTRAS\"$cand\","
done
EXTRAS=${EXTRAS%,}

$SUDO mkdir -p "$ROOT"
$SUDO sh -c "cat > $ROOT/local.json" <<JSON
{
  "models_root": "$MODELS_ROOT"$([ -n "$EXTRAS" ] && printf ',\n  "extra_roots": [%s]' "$EXTRAS")
}
JSON

COUNT=$(find "$MODELS_ROOT/manifests" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "${COUNT:-0}" -gt 0 ]; then
    say "$COUNT model(s) already in $MODELS_ROOT; llmash serves them from there"
elif is_ggufs "$MODELS_ROOT"; then
    say "reading the GGUFs in $MODELS_ROOT where they are"
else
    say "models go in $MODELS_ROOT"
fi
[ -n "$EXTRAS" ] && say "also reading $(printf '%s' "$EXTRAS" | tr -d '\"')"

# --------------------------------------------------------------- llama.cpp
# llmash runs models through llama-server, so the installer fetches a build the
# way the Windows one does. There is no Linux CUDA release upstream; Vulkan is
# what covers an NVIDIA or AMD card, and everything else gets the CPU build.
has_gpu() {
    [ -e /proc/driver/nvidia/version ] && return 0
    [ -d /sys/module/amdgpu ] && return 0
    { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; } && return 0
    return 1
}
has_vulkan() { ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1'; }

runtime_asset() {
    if [ "$ARCH" = arm64 ]; then cpu=ubuntu-arm64; vk=ubuntu-vulkan-arm64
    else                         cpu=ubuntu-x64;   vk=ubuntu-vulkan-x64; fi
    case $RUNTIME in
        cpu)    printf '%s' "$cpu" ;;
        vulkan) printf '%s' "$vk" ;;
        rocm)   printf '%s' 'ubuntu-rocm-10.0-x64' ;;
        auto)
            if has_gpu && has_vulkan; then printf '%s' "$vk"
            else printf '%s' "$cpu"; fi ;;
        *) die "--runtime takes auto, vulkan, rocm, cpu or none" ;;
    esac
}

step 'llama.cpp'
if [ "$RUNTIME" = none ]; then
    say 'skipped; point LLAMA_BIN at your own build'
elif [ "$DOWNLOAD" = 0 ]; then
    say "would fetch the $(runtime_asset) build"
else
    asset=$(runtime_asset)
    case $asset in
        *vulkan*) say 'a GPU is present, taking the Vulkan build' ;;
        *rocm*)   say 'taking the ROCm build' ;;
        *) has_gpu && say 'a GPU is present but libvulkan is not installed, taking the CPU build'                    || say 'no GPU found, taking the CPU build' ;;
    esac
    url=$(curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"           | grep -o "https://[^\"]*bin-${asset}\.tar\.gz" | head -1)
    if [ -z "$url" ]; then
        warn "llama.cpp publishes no $asset build; llmash cannot run a model until"
        say  "you put llama-server in $LIB/runtime or set LLAMA_BIN"
    else
        curl -fsSL --progress-bar "$url" -o "$TMP/llama.tar.gz" || die 'llama.cpp download failed'
        mkdir -p "$TMP/lc"
        tar -xzf "$TMP/llama.tar.gz" -C "$TMP/lc"
        src=$(find "$TMP/lc" -name llama-server -type f | head -1)
        [ -n "$src" ] || die 'that llama.cpp build has no llama-server in it'
        $SUDO rm -rf "$LIB/runtime"
        $SUDO mkdir -p "$LIB/runtime"
        $SUDO cp -a "$(dirname "$src")/." "$LIB/runtime/"
        $SUDO chmod +x "$LIB/runtime/llama-server"
        say "$(basename "$url")"

        lmiss=$(ldd "$LIB/runtime/llama-server" 2>/dev/null | awk '/not found/{print $1}')
        if [ -n "$lmiss" ]; then
            warn 'llama-server needs shared libraries this machine does not have:'
            for m in $lmiss; do say "    $m"; done
            case $lmiss in
                *libgomp*) say 'try one of:  apt install libgomp1  |  dnf install libgomp  |  pacman -S gcc-libs' ;;
                *libvulkan*) say 'try one of:  apt install libvulkan1  |  dnf install vulkan-loader  |  pacman -S vulkan-icd-loader' ;;
            esac
            say 'llmash will install, but cannot run a model until those are there'
        elif "$LIB/runtime/llama-server" --version >/dev/null 2>&1; then
            say 'it runs'
        fi
    fi
fi

if [ "$SERVICE" = 1 ] && [ "$DOWNLOAD" = 1 ] && command -v systemctl >/dev/null 2>&1; then
    step 'Starting at boot'
    if [ "$SYSTEM" = 1 ]; then
        $SUDO sh -c "cat > $UNIT" <<UNIT
[Unit]
Description=llmash
After=network-online.target

[Service]
ExecStart=$BIN/llmash serve
WorkingDirectory=$ROOT
Environment="OLLAMA_MODELS=$MODELS_ROOT"
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
        $SUDO systemctl daemon-reload
        $SUDO systemctl enable --now llmash
        say 'running as a system service'
        say 'logs:  journalctl -u llmash -f'
    else
        mkdir -p "$(dirname "$USER_UNIT")"
        cat > "$USER_UNIT" <<UNIT
[Unit]
Description=llmash

[Service]
ExecStart=$BIN/llmash serve
WorkingDirectory=$ROOT
Environment="OLLAMA_MODELS=$MODELS_ROOT"
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT
        systemctl --user daemon-reload
        systemctl --user enable --now llmash
        say 'running as a user service'
        say 'logs:  journalctl --user -u llmash -f'
    fi
    if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama 2>/dev/null; then
        warn 'ollama.service is running and holds port 11434; stop it or llmash cannot listen'
        say  '  sudo systemctl disable --now ollama'
    fi
else
    step 'Starting llmash'
    say 'no service installed; start it yourself with:  llmash serve'
fi

step 'Done'
say "$BIN/llmash"
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) say "$BIN is not on your PATH; add it to your shell profile" ;;
esac
warn 'pull is not available on Linux yet; models already on disk are served normally'
say 'try:  llmash list'
