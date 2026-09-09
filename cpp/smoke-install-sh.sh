#!/usr/bin/env bash
# Installs llmash the way a stranger would, from the published release, and
# runs the binary that lands. Needs the network.
set -u
fail=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s  %s\n' "$1" "${2:-}"; fail=1; }

SRC=${1:-https://raw.githubusercontent.com/omgitsbase/llmash/main/install.sh}
case $SRC in
    http*) curl -fsSL "$SRC" -o /tmp/install.sh || { echo "cannot fetch $SRC"; exit 1; } ;;
    *) cp "$SRC" /tmp/install.sh ;;
esac

# an Ollama store with one model in it, which the install must find and keep
export HOME=/root
mkdir -p "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3"
echo '{}' > "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3/latest"

out=$(sh /tmp/install.sh 2>&1); rc=$?
echo "$out" | sed 's/^/    | /'
[ "$rc" -eq 0 ] && ok "the installer exits clean" || bad "installer rc=$rc"

[ -x /usr/local/lib/llmash/llmash ] && ok "the binary is installed" || bad "binary missing"
[ -L /usr/local/bin/llmash ] && ok "llmash is on PATH" || bad "llmash symlink"
[ -L /usr/local/bin/ollama ] && ok "ollama is answered too" || bad "ollama symlink"

echo "$out" | grep -q "1 model(s) already in $HOME/.ollama/models" \
    && ok "the existing Ollama store was found" || bad "ollama store not found"

cfg=/var/lib/llmash/local.json
if [ -f "$cfg" ]; then
    cat "$cfg" | sed 's/^/    | /'
    grep -q "\"models_root\": \"$HOME/.ollama/models\"" "$cfg" \
        && ok "local.json points at that store" || bad "local.json models_root"
    python3 -c "import json,sys; json.load(open('$cfg'))" 2>/dev/null \
        && ok "local.json is valid JSON" || bad "local.json is not valid JSON"
else
    bad "local.json was not written"
fi

# no systemd in a container: the installer must say so, not fall over
echo "$out" | grep -q 'no service installed' \
    && ok "a machine without systemd is handled" || bad "systemd fallback"

v=$(llmash --version 2>&1 || llmash version 2>&1)
echo "$v" | grep -qi 'version' && ok "runs: $v" || bad "the installed binary does not run" "$v"

llmash list >/dev/null 2>&1 && ok "\`llmash list\` works" || bad "llmash list"

out=$(sh /tmp/install.sh --uninstall 2>&1)
{ [ ! -e /usr/local/bin/llmash ] && [ ! -e /usr/local/lib/llmash ]; } \
    && ok "uninstall removes it" || bad "uninstall left files"
[ -f "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3/latest" ] \
    && ok "uninstall kept the models" || bad "uninstall deleted models"

echo
[ "$fail" -eq 0 ] && echo "install.sh smoke passes" || echo "install.sh smoke FAILED"
exit $fail
