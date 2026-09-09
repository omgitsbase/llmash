#!/usr/bin/env bash
# Drives install.sh's Ollama detection with fake stores, without a network.
set -u
fail=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s  %s\n' "$1" "${2:-}"; fail=1; }

I=/src/install.sh
sh -n "$I" && ok "install.sh parses" || bad "install.sh parses"

out=$(sh "$I" --bogus 2>&1); rc=$?
{ [ "$rc" -ne 0 ] && echo "$out" | grep -q "unknown option"; } \
    && ok "an unknown option is refused" || bad "unknown option" "$out"

out=$(sh "$I" --uninstall --user 2>&1); rc=$?
{ [ "$rc" -eq 0 ] && echo "$out" | grep -q "were left alone"; } \
    && ok "uninstall keeps models" || bad "uninstall" "$out"

# a home-directory Ollama store is found on its own
export HOME=/tmp/h1; rm -rf $HOME; mkdir -p "$HOME/.ollama/models/manifests/registry/x/y"
touch "$HOME/.ollama/models/manifests/registry/x/y/latest"
out=$(LLMASH_REPO=omgitsbase/nope sh "$I" --user --no-service --dry-run 2>&1)
echo "$out" | grep -q "1 model(s) already in $HOME/.ollama/models" \
    && ok "finds Ollama's store under \$HOME" || bad "home store" "$(echo "$out" | tail -3)"

# OLLAMA_MODELS wins over the default
export HOME=/tmp/h2; rm -rf $HOME; mkdir -p $HOME /tmp/elsewhere/manifests
out=$(OLLAMA_MODELS=/tmp/elsewhere LLMASH_REPO=omgitsbase/nope sh "$I" --user --no-service --dry-run 2>&1)
echo "$out" | grep -q "OLLAMA_MODELS points at /tmp/elsewhere" \
    && ok "OLLAMA_MODELS is read from the environment" || bad "OLLAMA_MODELS" "$(echo "$out" | tail -3)"

# LLMASH_MODELS is the same setting under llmash's name
out=$(LLMASH_MODELS=/tmp/elsewhere LLMASH_REPO=omgitsbase/nope sh "$I" --user --no-service --dry-run 2>&1)
echo "$out" | grep -q "points at /tmp/elsewhere" \
    && ok "LLMASH_MODELS is read too" || bad "LLMASH_MODELS" "$(echo "$out" | tail -3)"

# a service's variable, which a shell never shows
mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_MODELS=/srv/models"\n' > /etc/systemd/system/ollama.service
out=$(env -u OLLAMA_MODELS -u LLMASH_MODELS LLMASH_REPO=omgitsbase/nope sh "$I" --user --no-service --dry-run 2>&1)
echo "$out" | grep -q "points at /srv/models" \
    && ok "reads OLLAMA_MODELS out of ollama.service" || bad "unit env" "$(echo "$out" | tail -3)"
rm -f /etc/systemd/system/ollama.service

# a folder of loose GGUFs is read where it is
export HOME=/tmp/h3; rm -rf $HOME; mkdir -p $HOME /tmp/ggufs; : > /tmp/ggufs/a.gguf
out=$(OLLAMA_MODELS=/tmp/ggufs LLMASH_REPO=omgitsbase/nope sh "$I" --user --no-service --dry-run 2>&1)
echo "$out" | grep -q "reading the GGUFs in /tmp/ggufs" \
    && ok "a GGUF folder is read in place" || bad "gguf folder" "$(echo "$out" | tail -3)"

# a missing release stops rather than half-installing
out=$(LLMASH_REPO=omgitsbase/does-not-exist sh "$I" --user --no-service 2>&1); rc=$?
[ "$rc" -ne 0 ] && ok "a missing release stops the install" || bad "missing release" "$out"

echo
[ "$fail" -eq 0 ] && echo "install.sh checks pass" || echo "install.sh checks FAILED"
exit $fail
