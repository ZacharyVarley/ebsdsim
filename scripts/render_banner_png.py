"""One-off helper: rasterize docs/ebsdsim-banner.svg to PNG for PyPI README."""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "docs" / "ebsdsim-banner.svg"
OUT = ROOT / "docs" / "ebsdsim-banner.png"


def main() -> None:
    svg = SVG.read_text(encoding="utf-8")
    html = (
        "<!DOCTYPE html><html><body style='margin:0;background:#030914'>"
        f"{svg}"
        "</body></html>"
    )
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 400})
        page.set_content(html)
        page.screenshot(path=str(OUT), type="png")
        browser.close()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
