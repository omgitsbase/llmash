#!/bin/sh
# llmash installer for Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/omgitsbase/llmash/main/install.sh | sh
#
#   --dir <path>   install somewhere other than /usr/local
#   --user         install for this user only, into ~/.local, with no service
#   --no-service   skip the systemd unit
#   --uninstall    remove everything this script created (models are kept)
#   --yes          answer yes to every prompt

set -eu

REPO=${LLMASH_REPO:-omgitsbase/llmash}
PREFIX=/usr/local
SERVICE=1
SYSTEM=1
ASSUME_YES=0
UNINSTALL=0

while [ $# -gt 0 ]; do
    case $1 in
        --dir) PREFIX=$2; shift 2 ;;
        --user) SYSTEM=0; PREFIX=$HOME/.local; shift ;;
        --no-service) SERVICE=0; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

say()  { printf '  %s\n' "$1"; }
step() { printf '\n\033[36m> %s\033[0m\n' "$1"; }
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
    $SUDO rm -f "$BIN/llmash" "$BIN/ollama"
    $SUDO rm -rf "$LIB"
    say 'removed; your models were left alone'
    exit 0
fi

step 'Checking this machine'
case $(uname -m) in
    x86_64|amd64) ARCH=x64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) die "no llmash build for $(uname -m)" ;;
esac
say "linux $ARCH"

command -v curl >/dev/null 2>&1 || die 'curl is needed to download llmash'
command -v tar  >/dev/null 2>&1 || die 'tar is needed to unpack llmash'

step 'Fetching llmash'
ASSET=llmash-linux-$ARCH.tar.gz
API=https://api.github.com/repos/$REPO/releases/latest
URL=$(curl -fsSL "$API" | sed -n 's/.*"browser_download_url": *"\([^"]*'"$ASSET"'\)".*/\1/p' | head -1)
[ -n "$URL" ] || die "the latest release of $REPO has no $ASSET"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fsSL --progress-bar "$URL" -o "$TMP/$ASSET" || die 'download failed'
$SUDO mkdir -p "$BIN" "$LIB"
$SUDO tar -xzf "$TMP/$ASSET" -C "$LIB"
$SUDO ln -sf "$LIB/llmash" "$BIN/llmash"
say "installed into $LIB"

if [ ! -e "$BIN/ollama" ]; then
    $SUDO ln -sf "$LIB/llmash" "$BIN/ollama"
    say 'also answers to `ollama`'
fi

VERSION=$("$BIN/llmash" -v 2>/dev/null | awk '{print $NF}')
say "llmash ${VERSION:-installed}"

step 'llama.cpp'
if command -v llama-server >/dev/null 2>&1; then
    say "found $(command -v llama-server)"
elif [ -x "$LIB/runtime/llama-server" ]; then
    say "found $LIB/runtime/llama-server"
else
    say 'not found. Build llama.cpp with your GPU backend and put llama-server on PATH,'
    say "or drop it in $LIB/runtime, or set LLAMA_BIN."
    say '  https://github.com/ggml-org/llama.cpp'
fi

if [ "$SERVICE" = 1 ] && command -v systemctl >/dev/null 2>&1; then
    step 'Starting at boot'
    if [ "$SYSTEM" = 1 ]; then
        # Ollama's shape: its own account, its own home, models under it
        if ! id llmash >/dev/null 2>&1; then
            $SUDO useradd -r -s /bin/false -U -m -d /usr/share/llmash llmash
            say 'created the llmash user'
        fi
        $SUDO sh -c "cat > $UNIT" <<UNIT
[Unit]
Description=llmash
After=network-online.target

[Service]
ExecStart=$BIN/llmash serve
User=llmash
Group=llmash
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=multi-user.target
UNIT
        $SUDO systemctl daemon-reload
        $SUDO systemctl enable --now llmash
        say 'running as a system service'
        say 'models live in /usr/share/llmash/.ollama/models; set OLLAMA_MODELS in the unit to move them'
        say 'logs:  journalctl -u llmash -f'
    else
        mkdir -p "$(dirname "$USER_UNIT")"
        cat > "$USER_UNIT" <<UNIT
[Unit]
Description=llmash

[Service]
ExecStart=$BIN/llmash serve
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
else
    step 'Starting llmash'
    say 'no service was installed; start it yourself with:  llmash serve'
fi

step 'Done'
say "$BIN/llmash"
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) say "$BIN is not on your PATH; add it to your shell profile" ;;
esac
say 'try:  llmash list'
