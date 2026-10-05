"""
Build the manuscript and (optionally) render page previews.

    python paper/build.py              # pdflatex + bibtex + 2x pdflatex -> paper/main.pdf
    python paper/build.py --preview    # also writes paper/preview/page-XX.png (needs PyMuPDF)

The dataset numbers and tables are regenerated first by
scripts/tools/paper_dataset_stats.py unless --no-stats is given.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent


def find_tex(name: str) -> str:
    """Locate a TeX binary on PATH or in the default per-user MiKTeX location."""
    exe = shutil.which(name)
    if exe:
        return exe
    miktex = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64" / f"{name}.exe"
    if miktex.exists():
        return str(miktex)
    sys.exit(f"{name} not found - install a TeX distribution (MiKTeX / TeX Live) or use Overleaf")


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        log = (PAPER / "main.log").read_text(errors="replace") if (PAPER / "main.log").exists() else res.stdout
        errors = [line for line in log.splitlines() if line.startswith("!")]
        sys.exit(f"{Path(cmd[0]).name} failed:\n" + "\n".join(errors[:10] or log.splitlines()[-30:]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="render PNG previews of every page")
    ap.add_argument("--no-stats", action="store_true", help="do not regenerate the dataset statistics")
    args = ap.parse_args()

    if not args.no_stats:
        subprocess.run([sys.executable, str(ROOT / "scripts/tools/paper_dataset_stats.py")], check=True,
                       capture_output=True)
    tex = find_tex("pdflatex")
    # MiKTeX would otherwise install missing packages on the fly; never change the TeX
    # installation from a build script (the preamble copes with missing optional packages).
    no_install = ["--disable-installer"] if "miktex" in tex.lower() else []
    pdflatex = [tex, *no_install, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    run(pdflatex)
    run([find_tex("bibtex"), "main"])
    run(pdflatex)
    run(pdflatex)
    print(f"built {PAPER / 'main.pdf'}")

    if args.preview:
        import pymupdf
        out = PAPER / "preview"
        out.mkdir(exist_ok=True)
        doc = pymupdf.open(PAPER / "main.pdf")
        for i, page in enumerate(doc, 1):
            page.get_pixmap(dpi=110).save(out / f"page-{i:02d}.png")
        print(f"{len(doc)} page previews in {out}")


if __name__ == "__main__":
    main()
