#!/usr/bin/env bash
# Checks install.sh parses, and that its uninstall and --user paths behave
# without a network or a release to download.
set -u
fail=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fail=1; }

sh -n /src/install.sh && ok "install.sh parses" || bad "install.sh parses"

if command -v shellcheck >/dev/null 2>&1; then
    shellcheck -S error /src/install.sh && ok "shellcheck finds no errors" || bad "shellcheck"
fi

out=$(sh /src/install.sh --bogus 2>&1); rc=$?
[ "$rc" -ne 0 ] && echo "$out" | grep -q "unknown option" \
    && ok "an unknown option is refused" || bad "an unknown option is refused"

out=$(sh /src/install.sh --uninstall --user 2>&1); rc=$?
[ "$rc" -eq 0 ] && echo "$out" | grep -q "models were left alone" \
    && ok "uninstall runs and keeps models" || bad "uninstall runs ($rc): $out"

out=$(HOME=/tmp/fakehome LLMASH_REPO=omgitsbase/does-not-exist sh /src/install.sh --user --no-service 2>&1); rc=$?
[ "$rc" -ne 0 ] && ok "a missing release stops the install" || bad "a missing release stops the install ($rc)"

echo
[ "$fail" -eq 0 ] && echo "install.sh checks pass" || echo "install.sh checks FAILED"
exit $fail
