"""Compile llmash and package it for a release.

    python build.py            build dist/llmash-win-x64.zip
    python build.py --here     ...and run this checkout on the build

One Go program is built twice: llmash.exe is the command line and `llmash
serve`; llmashw.exe is the same program without a console, for the tray and the
background server. Both need MinGW's windres for the icon and version resource.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE / "cmd" / "llmash"
ASSETS = HERE / "assets"
OUT = HERE / "dist" / "llmash"
ZIP = HERE / "dist" / "llmash-win-x64.zip"


def version() -> str:
    m = re.search(r'serverVersion\s*=\s*"([^"]+)"', (PKG / "config.go").read_text("utf-8"))
    if not m:
        raise SystemExit("serverVersion not found in config.go")
    return m.group(1)


def go_build(name: str, windowed: bool) -> None:
    ld = "-s -w" + (" -H windowsgui" if windowed else "")
    env = dict(os.environ, CGO_ENABLED="0", GOOS="windows", GOARCH="amd64")
    subprocess.run(["go", "build", "-trimpath", "-ldflags", ld, "-o", str(OUT / name), "./cmd/llmash"],
                   cwd=HERE, env=env, check=True)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / "VERSION").write_text(version() + "\n", encoding="utf-8")
    subprocess.run(["windres", "llmash.rc", "-O", "coff", "-o", str(PKG / "rsrc.syso")],
                   cwd=ASSETS, check=True)
    go_build("llmash.exe", windowed=False)
    go_build("llmashw.exe", windowed=True)
    shutil.copy(ASSETS / "icon.ico", OUT / "llmash.ico")
    shutil.copy(ASSETS / "icon.png", OUT / "llmash.png")
    shutil.copy(HERE / "install.ps1", OUT / "install.ps1")
    shutil.copy(HERE / "VERSION", OUT / "VERSION")


def stop_running(root: pathlib.Path) -> None:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe' OR Name='llmash.exe'\" | "
          "Where-Object { $_.CommandLine -like '*" + str(root) + "*' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=30)
    time.sleep(1.0)


def install_here() -> None:
    """Run this checkout on the build it just produced."""
    stop_running(HERE)
    for name in ("llmash.exe", "llmashw.exe", "VERSION", "llmash.ico", "llmash.png"):
        shutil.copy(OUT / name, HERE / name)
    lnk = pathlib.Path(os.environ.get("APPDATA", "")) / \
        "Microsoft/Windows/Start Menu/Programs/Startup/llmash.lnk"
    lnk.unlink(missing_ok=True)
    subprocess.Popen([str(HERE / "llmashw.exe"), "tray"], cwd=str(HERE), close_fds=True,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"running {HERE} on the build; the tray is up and it starts the server")


def main() -> int:
    t0 = time.time()
    if OUT.exists():
        shutil.rmtree(OUT)
    build()
    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
    unpacked = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"llmash {version()}: {unpacked / 1e6:.1f} MB unpacked, "
          f"{ZIP.stat().st_size / 1e6:.1f} MB zipped, {time.time() - t0:.0f}s -> {ZIP}")
    if "--here" in sys.argv:
        install_here()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
