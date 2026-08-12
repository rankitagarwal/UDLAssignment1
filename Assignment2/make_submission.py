#!/usr/bin/env python3
"""Build the portal submission bundle: submission/*.{ipynb,html,pdf} + the zip.

The WILP portal caps uploads at 10 MB per file. An executed notebook here is
~12 MB, essentially all of it base64 PNG figures -- PNG barely compresses grids
of CIFAR samples. This re-encodes the *stored image bytes only*: code cells,
execution counts, stdout and text/plain outputs come through byte-identical, and
the HTML twins get the same substitution so they stay in sync with the notebooks.

Re-running a notebook regenerates plain PNG figures (matplotlib's inline backend
default), so the source files go back to full size. Re-run this script after any
re-execution, before uploading.

    python3 make_submission.py
"""
import base64
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission"
ZIP = HERE / "UDL_Assignment2_Group_123.zip"

# Photo grids re-encode to JPEG; charts stay lossless PNG. Palette-quantised PNG
# was tried and rejected -- it posterises the sample grids, and a grader marking
# perceptual quality would read that banding as a model artefact.
JPEG_Q = 92
PHOTO_COLOR_THRESHOLD = 5000

PARTS = [
    ("PartA_VAE_VQVAE.ipynb", "PartA_VAE_VQVAE.html", "parta"),
    ("PartB_WGAN.ipynb", "PartB_WGAN.html", "partb"),
]
PDF_SRC = "PartC_Comparative_Analysis.pdf"

remap = {}  # original b64 (whitespace-stripped) -> (mime, new b64)


def recompress(raw):
    im = Image.open(io.BytesIO(raw))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im)
    rgb = im.convert("RGB")

    ncolors = len(rgb.getcolors(maxcolors=1 << 20) or [None] * (1 << 20))
    buf = io.BytesIO()
    if ncolors > PHOTO_COLOR_THRESHOLD:
        rgb.save(buf, "JPEG", quality=JPEG_Q, optimize=True, subsampling=0)
        mime = "image/jpeg"
    else:
        rgb.save(buf, "PNG", optimize=True)
        mime = "image/png"

    out = buf.getvalue()
    return None if len(out) >= len(raw) else (mime, out)


def convert_notebook(name, dst):
    nb = json.loads((HERE / name).read_text())
    before = after = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data")
            if not data or "image/png" not in data:
                continue
            value = data["image/png"]
            key = re.sub(r"\s", "", "".join(value) if isinstance(value, list) else value)
            raw = base64.b64decode(key)
            before += len(raw)

            if key not in remap:
                result = recompress(raw)
                if result is None:
                    after += len(raw)
                    continue
                remap[key] = (result[0], base64.b64encode(result[1]).decode())
            mime, new_b64 = remap[key]
            after += len(base64.b64decode(new_b64))

            del data["image/png"]
            data[mime] = new_b64
            meta = output.get("metadata")
            if isinstance(meta, dict) and "image/png" in meta:
                meta[mime] = meta.pop("image/png")

    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"  {name}: figures {before / 1e6:.2f} -> {after / 1e6:.2f} MB")
    return nb


def convert_html(name, dst):
    def sub(m):
        hit = remap.get(m.group(1))
        return f"data:{hit[0]};base64,{hit[1]}" if hit else m.group(0)

    html = (HERE / name).read_text()
    dst.write_text(re.sub(r"data:image/png;base64,([A-Za-z0-9+/=]+)", sub, html))


def signature(nb):
    """Everything except the image payloads -- must survive untouched."""
    return [
        (
            c["cell_type"],
            "".join(c.get("source", [])),
            c.get("execution_count"),
            [
                (
                    o.get("output_type"),
                    "".join(o.get("text", [])),
                    "".join(o.get("data", {}).get("text/plain", [])),
                )
                for o in c.get("outputs", [])
            ],
        )
        for c in nb["cells"]
    ]


def strip_images(html):
    return re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "IMG", html)


def verify(nb_name, new_nb, html_name, new_html_path):
    old = json.loads((HERE / nb_name).read_text())
    assert old["nbformat"] == new_nb["nbformat"], nb_name
    assert signature(old) == signature(new_nb), f"code/text outputs changed in {nb_name}"

    for cell in new_nb["cells"]:
        for output in cell.get("outputs", []):
            for mime, value in output.get("data", {}).items():
                if mime.startswith("image/"):
                    blob = "".join(value) if isinstance(value, list) else value
                    Image.open(io.BytesIO(base64.b64decode(blob))).load()

    assert strip_images((HERE / html_name).read_text()) == \
        strip_images(new_html_path.read_text()), f"markup changed in {html_name}"


def shrink_pdf(dst):
    """Recompress embedded images to JPEG without downsampling -- full resolution."""
    if not shutil.which("gs"):
        print("  ghostscript not found; copying PDF unchanged")
        shutil.copy(HERE / PDF_SRC, dst)
        return
    subprocess.run(
        ["gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5", "-dNOPAUSE", "-dQUIET",
         "-dBATCH", "-dDetectDuplicateImages=true", "-dAutoFilterColorImages=false",
         "-dColorImageFilter=/DCTEncode", f"-dJPEGQ={JPEG_Q}",
         "-dDownsampleColorImages=false", f"-sOutputFile={dst}", str(HERE / PDF_SRC)],
        check=True,
    )
    print(f"  {PDF_SRC}: {(HERE / PDF_SRC).stat().st_size / 1e6:.2f} -> "
          f"{dst.stat().st_size / 1e6:.2f} MB")


def main():
    OUT.mkdir(exist_ok=True)
    print("recompressing embedded figures")

    for nb_name, html_name, stem in PARTS:
        new_nb = convert_notebook(nb_name, OUT / f"{stem}.ipynb")
        convert_html(html_name, OUT / f"{stem}.html")
        verify(nb_name, new_nb, html_name, OUT / f"{stem}.html")

    shrink_pdf(OUT / "partc.pdf")

    names = ["parta.ipynb", "partb.ipynb", "parta.html", "partb.html", "partc.pdf"]
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in names:
            z.write(OUT / n, n)

    print("\nverified: code cells, execution counts and text outputs identical to source\n")
    over = []
    for n in names:
        size = (OUT / n).stat().st_size / 1e6
        flag = "  OVER 10 MB" if size > 10 else ""
        over += [n] if size > 10 else []
        print(f"  {n:14s} {size:6.2f} MB{flag}")
    zsize = ZIP.stat().st_size / 1e6
    print(f"  {ZIP.name:14s} {zsize:6.2f} MB{'  OVER 10 MB' if zsize > 10 else ''}")
    if over or zsize > 10:
        print("\nstill over the portal limit -- lower JPEG_Q and re-run")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
