#!/usr/bin/env bash
# Builds the Linux release binary in the container from cpp/Dockerfile.test
# and leaves dist/llmash-linux-x64.tar.gz beside the Windows zip.
set -eu

BUILD=/tmp/relbuild
cmake -S /src/cpp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-static-libstdc++ -static-libgcc" >/tmp/cmake.log 2>&1 || {
    echo "configure failed"; tail -20 /tmp/cmake.log; exit 2;
}
cmake --build "$BUILD" -j "$(nproc)" --target llmash > /tmp/build.log 2>&1 || {
    echo "build failed"; grep -E "error:" /tmp/build.log | head -20; exit 1;
}

strip "$BUILD/llmash"
mkdir -p /src/dist
tar -czf /src/dist/llmash-linux-x64.tar.gz -C "$BUILD" llmash
echo "built $("$BUILD/llmash" --version)"
ls -l /src/dist/llmash-linux-x64.tar.gz
