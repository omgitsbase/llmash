#!/usr/bin/env bash
# Builds the Linux release binary in the container from cpp/Dockerfile.test,
# runs the tests on that same build, and leaves dist/llmash-linux-x64.tar.gz
# beside the Windows zip. A volume mounted at /build keeps the build between
# runs, so a small change rebuilds in seconds. --no-tests skips the tests.
set -eu

# Linked against a libcurl.so.4 whose symbols carry no version, so the binary
# asks for plain curl_* names: Debian's libcurl tags them CURL_OPENSSL_4 and
# Fedora's tags nothing, and a binary that wants the tag makes Fedora's loader
# warn on every run.
STUB=/tmp/curlstub
REAL=$(ls /usr/lib/*/libcurl.so.4 | head -1)
mkdir -p "$STUB"
nm -D --defined-only "$REAL" | awk '$2 == "T" { sub(/@.*/, "", $3); print "void " $3 "(void) {}" }' | sort -u > "$STUB/stub.c"
gcc -shared -fPIC -o "$STUB/libcurl.so.4" -Wl,-soname,libcurl.so.4 "$STUB/stub.c"
ln -sf libcurl.so.4 "$STUB/libcurl.so"

BUILD=/tmp/relbuild
if [ -d /build ]; then
    BUILD=/build/release
fi
TESTS=1
if [ "${1:-}" = "--no-tests" ]; then
    TESTS=0
fi
cmake -S /src/cpp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCURL_LIBRARY="$STUB/libcurl.so" \
      -DCMAKE_CXX_FLAGS="-static-libstdc++ -static-libgcc" >/tmp/cmake.log 2>&1 || {
    echo "configure failed"; tail -20 /tmp/cmake.log; exit 2;
}
TARGET="--target llmash"
if [ "$TESTS" = 1 ]; then
    TARGET=""
fi
cmake --build "$BUILD" -j "$(nproc)" $TARGET > /tmp/build.log 2>&1 || {
    echo "build failed"; grep -E "error:" /tmp/build.log | head -20; exit 1;
}

if [ "$TESTS" = 1 ]; then
    pass=0
    total=0
    for t in "$BUILD"/*_test "$BUILD"/test_main; do
        [ -x "$t" ] || continue
        total=$((total + 1))
        if out=$(cd "$BUILD" && "$t" 2>&1); then
            pass=$((pass + 1))
        else
            echo "  $(basename "$t") FAIL"
            echo "$out" | grep -iE "^fail" | head -8 | sed 's/^/      /'
        fi
    done
    echo "$pass/$total suites pass on linux"
    [ "$pass" -eq "$total" ] || exit 1
fi

# the kept build stays unstripped so the next link is incremental
cp "$BUILD/llmash" /tmp/llmash
strip /tmp/llmash
if objdump -T /tmp/llmash | grep -q 'CURL_'; then
    echo "llmash still wants versioned curl symbols"; exit 1
fi
mkdir -p /src/dist
tar -czf /src/dist/llmash-linux-x64.tar.gz -C /tmp llmash
echo "built $(LD_LIBRARY_PATH= /tmp/llmash --version)"
ls -l /src/dist/llmash-linux-x64.tar.gz
