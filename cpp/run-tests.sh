#!/usr/bin/env bash
# Builds llmash for Linux and runs its tests. Meant to be run in the
# container from cpp/Dockerfile.test, with the repository at /src.
set -u

BUILD=/tmp/build
cmake -S /src/cpp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/tmp/cmake.log 2>&1 || {
    echo "configure failed"; tail -20 /tmp/cmake.log; exit 2;
}

cmake --build "$BUILD" -j "$(nproc)" -- -k > /tmp/build.log 2>&1
errors=$(grep -cE "error:" /tmp/build.log || true)
echo "build: $errors error(s)"
if [ "$errors" -gt 0 ]; then
    grep -E "error:" /tmp/build.log | sed 's|/src/cpp/src/||' | sort -u | head -30
    echo
    echo "by file:"
    grep -oE "/src/cpp/src/[A-Za-z0-9_.]+:[0-9]+:[0-9]+: (error|fatal error)" /tmp/build.log \
        | sed 's|/src/cpp/src/||' | cut -d: -f1 | sort | uniq -c | sort -rn
    exit 1
fi

echo
cd "$BUILD" || exit 2
pass=0
total=0
for t in "$BUILD"/*_test "$BUILD"/test_main; do
    [ -x "$t" ] || continue
    name=$(basename "$t")
    total=$((total + 1))
    if out=$("$t" 2>&1); then
        pass=$((pass + 1))
        printf '  %-22s pass   %s\n' "$name" "$(echo "$out" | tail -1)"
    else
        printf '  %-22s FAIL\n' "$name"
        echo "$out" | grep -iE "^fail" | head -8 | sed 's/^/      /'
    fi
done
echo
echo "$pass/$total suites pass on linux"
[ "$pass" -eq "$total" ]
