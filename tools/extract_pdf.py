"""Extract text and page images from a PDF.

Usage: python -I extract_pdf.py <pdf> <out_dir>

Writes:
  out_dir/text/page_XXX.txt   raw text (layout-preserving)
  out_dir/img/page_XXX.png    rendered page
"""
import sys
from pathlib import Path

import pymupdf


def main() -> None:
    pdf_path = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    print(f"pages={doc.page_count}")

    full_text = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        (out_dir / "text" / f"page_{i + 1:03d}.txt").write_text(text, encoding="utf-8")
        pix = page.get_pixmap(dpi=120)
        pix.save(out_dir / "img" / f"page_{i + 1:03d}.png")
        full_text.append(f"\n\n===== PAGE {i + 1} =====\n{text}")

    (out_dir / "full_text.txt").write_text("".join(full_text), encoding="utf-8")
    print(f"wrote {doc.page_count} pages to {out_dir}")


if __name__ == "__main__":
    main()
