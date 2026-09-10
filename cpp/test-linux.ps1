# Builds llmash for Linux in a container and runs its tests.
#
#   .\cpp\test-linux.ps1

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$tag = 'llmash-linux-test'

docker build -q -t $tag -f "$repo\cpp\Dockerfile.test" "$repo\cpp" | Out-Null
docker run --rm -v "${repo}:/src" $tag bash /src/cpp/run-tests.sh
exit $LASTEXITCODE
