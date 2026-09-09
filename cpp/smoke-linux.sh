#!/usr/bin/env bash
# Exercises the Linux binary in the container. Nothing here needs a GPU or a
# model; it checks the program runs and says sensible things.
set -u
B=/tmp/b
cmake -S /src/cpp -B "$B" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build "$B" -j "$(nproc)" >/dev/null 2>&1
cd /tmp || exit 2

echo "== version"
timeout 20 "$B/llmash" -v

echo
echo "== help, first lines"
timeout 20 "$B/llmash" help 2>&1 | head -8

echo
echo "== an unknown command"
timeout 20 "$B/llmash" bogus 2>&1
echo "exit=$?"

echo
echo "== models"
timeout 30 "$B/llmash" models 2>&1 | head -6

echo
echo "== serve, with no engine installed"
timeout 30 "$B/llmash" serve 2>&1 | head -4

echo
echo "== pull, which is not ported"
timeout 20 "$B/llmash" pull something 2>&1 | head -3
