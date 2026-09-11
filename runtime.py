import pathlib
import subprocess
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "dist" / "llmash-runtime-win-cuda-13-x64.zip"

NEEDED = ("llama-server.exe", "llama-server-impl.dll",
          "llama-quantize.exe", "llama-quantize-impl.dll", "llama.dll", "llama-common.dll", "mtmd.dll",
          "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll",
          "cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = pathlib.Path(sys.argv[1])
    missing = [n for n in NEEDED if not (src / n).exists()]
    if missing:
        print("missing from the build:", ", ".join(missing))
        return 1
    version = subprocess.run([str(src / "llama-server.exe"), "--version"], capture_output=True, text=True)
    stamp = (version.stdout + version.stderr).strip().splitlines()[0] if version.returncode == 0 else "unknown"
    OUT.parent.mkdir(exist_ok=True)
    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in NEEDED:
            z.write(src / n, n)
        z.writestr("RUNTIME.txt", stamp + "\n")
    size = sum((src / n).stat().st_size for n in NEEDED)
    print(f"{stamp}: {size / 1e6:.0f} MB unpacked, {OUT.stat().st_size / 1e6:.0f} MB zipped -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
