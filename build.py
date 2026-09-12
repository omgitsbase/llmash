import os
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
CPP = HERE / "cpp"
BUILD = CPP / "build-release"
ASSETS = HERE / "assets"
OUT = HERE / "dist" / "llmash"
ZIP = HERE / "dist" / "llmash-win-x64.zip"
TGZ = HERE / "dist" / "llmash-linux-x64.tar.gz"


def version() -> str:
    return (HERE / "VERSION").read_text("utf-8").strip()


def cmake() -> str:
    if shutil.which("cmake"):
        return "cmake"
    vswhere = pathlib.Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.exists():
        vs = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                            capture_output=True, text=True).stdout.strip()
        candidate = pathlib.Path(vs) / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
        if candidate.exists():
            return str(candidate)
    raise SystemExit("cmake not found; install Visual Studio 2022 with the C++ workload")


def crt_dlls() -> list:
    vswhere = pathlib.Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    vs = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                        capture_output=True, text=True).stdout.strip()
    redist = sorted((pathlib.Path(vs) / "VC/Redist/MSVC").glob("14.*"))
    if not redist:
        raise SystemExit("no Visual C++ redistributable found next to the compiler")
    crt = redist[-1] / "x64/Microsoft.VC143.CRT"
    names = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
    return [crt / n for n in names]


def build() -> None:
    cm = cmake()
    # CMake reads VERSION when it configures, not when it builds, so a bumped
    # VERSION with a warm cache ships a binary reporting the old number.
    cache = BUILD / "CMakeCache.txt"
    if not cache.exists() or (HERE / "VERSION").stat().st_mtime > cache.stat().st_mtime:
        subprocess.run([cm, "-S", str(CPP), "-B", str(BUILD), "-G", "Visual Studio 17 2022", "-A", "x64"],
                       check=True)
    subprocess.run([cm, "--build", str(BUILD), "--config", "Release", "--target", "llmash", "llmashw"],
                   check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("llmash.exe", "llmashw.exe"):
        shutil.copy(BUILD / "Release" / name, OUT / name)
    shutil.copy(ASSETS / "icon.ico", OUT / "llmash.ico")
    shutil.copy(ASSETS / "icon.png", OUT / "llmash.png")
    shutil.copy(HERE / "install.ps1", OUT / "install.ps1")
    shutil.copy(HERE / "VERSION", OUT / "VERSION")
    shutil.copy(HERE / "LICENSE", OUT / "LICENSE")
    for dll in crt_dlls():
        shutil.copy(dll, OUT / dll.name)


def stop_running(root: pathlib.Path) -> None:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe' OR Name='llmash.exe'\" | "
          "Where-Object { $_.CommandLine -like '*" + str(root) + "*' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=30)
    time.sleep(1.0)


def installed_root() -> pathlib.Path:
    """Where the installer put llmash, so --here can say when it is stale."""
    return pathlib.Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "llmash"


def warn_if_install_is_older() -> None:
    """--here runs the build from the repository; the installed copy keeps the
    port. A CLI talking to an older server silently gets the older behaviour,
    which is how a pull asking for RCO-3 came back with the plain Q8_0."""
    root = installed_root()
    vf = root / "VERSION"
    if not vf.exists():
        return
    there = vf.read_text(encoding="utf-8").strip()
    if there != version():
        print(f"  note: {root} is {there}, this build is {version()}")
        print("  the installed copy owns port 11434, so `llmash` on PATH still")
        print(f"  answers as {there}; copy this build over it to test the new behaviour")


def install_here() -> None:
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
    warn_if_install_is_older()


def build_linux() -> None:
    """The Linux tarball, built in the container from cpp/Dockerfile.test.

    A release that ships only the zip leaves Linux on whatever version last
    had one; 0.4.11 through 0.4.15 went out Windows-only that way.
    """
    if shutil.which("docker") is None:
        print("  no docker, skipping the linux build")
        return
    image = "llmash-linux-test:latest"
    have = subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0
    if not have:
        print(f"  no {image}, skipping the linux build")
        return
    r = subprocess.run(["docker", "run", "--rm", "-v", f"{HERE}:/src", image,
                        "bash", "/src/cpp/build-linux.sh"], capture_output=True, text=True)
    if r.returncode != 0 or not TGZ.exists():
        print("  linux build failed:")
        print("   ", (r.stderr or r.stdout).strip().splitlines()[-1:] or "")
        return
    print(f"  linux {TGZ.stat().st_size / 1e6:.1f} MB -> {TGZ}")


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
    if "--no-linux" not in sys.argv:
        build_linux()
    if "--here" in sys.argv:
        install_here()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
