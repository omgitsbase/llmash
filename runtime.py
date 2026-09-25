import pathlib
import subprocess
import sys
import zipfile

from build import crt_dlls

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "dist" / "llmash-runtime-win-cuda-13-x64.zip"

NEEDED = ("llama-server.exe", "llama-server-impl.dll",
          "llama-quantize.exe", "llama-quantize-impl.dll", "llama.dll", "llama-common.dll", "mtmd.dll",
          "ggml.dll", "ggml-base.dll", "ggml-cuda.dll",
          "cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = pathlib.Path(sys.argv[1])
    missing = [n for n in NEEDED if not (src / n).exists()]
    # the CPU backend comes once per instruction set level, and ggml loads the best one the machine runs
    cpus = sorted(p.name for p in src.glob("ggml-cpu-*.dll"))
    if not any(n in cpus for n in ("ggml-cpu-x64.dll", "ggml-cpu-sse42.dll")):
        missing.append("ggml-cpu-x64.dll")
    if missing:
        print("missing from the build:", ", ".join(missing))
        return 1
    files = list(NEEDED) + cpus
    # the Visual C++ runtime and its OpenMP go beside llama-server, for a machine that never installed them
    crt = crt_dlls()
    omp = sorted(crt[0].parent.parent.glob("Microsoft.VC*.OpenMP"))
    if not omp:
        print("no Visual C++ OpenMP runtime beside", crt[0].parent)
        return 1
    extras = crt + [omp[-1] / "vcomp140.dll"]
    version = subprocess.run([str(src / "llama-server.exe"), "--version"], capture_output=True, text=True)
    # the backends it loads report themselves first
    lines = [l for l in (version.stdout + version.stderr).splitlines() if l.startswith("version:")]
    stamp = lines[0].strip() if version.returncode == 0 and lines else "unknown"
    OUT.parent.mkdir(exist_ok=True)
    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in files:
            z.write(src / n, n)
        for f in extras:
            z.write(f, f.name)
        z.writestr("RUNTIME.txt", stamp + "\n")
    # the release's own RUNTIME.txt asset, which `llmash update` compares with the installed one
    (OUT.parent / "RUNTIME.txt").write_text(stamp + "\n", encoding="utf-8", newline="\n")
    size = sum((src / n).stat().st_size for n in files)
    print(f"{stamp}: {size / 1e6:.0f} MB unpacked, {OUT.stat().st_size / 1e6:.0f} MB zipped -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
