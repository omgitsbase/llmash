"""Rebuild llmash.png and llmash.ico from a source image.

    python tools/make_icon.py C:\\Users\\reedk\\Downloads\\llmash.png

The .ico carries every size Windows asks for: the notification area uses 16 and
20, the taskbar 24 to 40, Explorer up to 256.
"""
import sys
import pathlib

from PIL import Image

SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
HERE = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "llmash.png")
    im = Image.open(src).convert("RGBA")
    if im.width != im.height:
        side = max(im.size)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        im = square
    print("source %s %s" % (src, im.size))

    png = im.resize((256, 256), Image.LANCZOS)
    png.save(HERE / "llmash.png", optimize=True)

    frames = [im.resize((s, s), Image.LANCZOS) for s in SIZES]
    frames[-1].save(HERE / "llmash.ico", format="ICO",
                    sizes=[(s, s) for s in SIZES],
                    append_images=frames[:-1])
    ico = HERE / "llmash.ico"
    print("wrote %s (%d bytes) and llmash.png (%d bytes)"
          % (ico, ico.stat().st_size, (HERE / "llmash.png").stat().st_size))
    with Image.open(ico) as check:
        print("ico sizes: %s" % sorted(check.info["sizes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
