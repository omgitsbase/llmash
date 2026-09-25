#!/usr/bin/env bash
# Builds the Linux release binary in the container from cpp/Dockerfile.test
# and leaves dist/llmash-linux-x64.tar.gz beside the Windows zip.
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
cmake -S /src/cpp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCURL_LIBRARY="$STUB/libcurl.so" \
      -DCMAKE_CXX_FLAGS="-static-libstdc++ -static-libgcc" >/tmp/cmake.log 2>&1 || {
    echo "configure failed"; tail -20 /tmp/cmake.log; exit 2;
}
cmake --build "$BUILD" -j "$(nproc)" --target llmash > /tmp/build.log 2>&1 || {
    echo "build failed"; grep -E "error:" /tmp/build.log | head -20; exit 1;
}

strip "$BUILD/llmash"
if objdump -T "$BUILD/llmash" | grep -q 'CURL_'; then
    echo "llmash still wants versioned curl symbols"; exit 1
fi
mkdir -p /src/dist
tar -czf /src/dist/llmash-linux-x64.tar.gz -C "$BUILD" llmash
echo "built $(LD_LIBRARY_PATH= "$BUILD/llmash" --version)"
ls -l /src/dist/llmash-linux-x64.tar.gz
