"""Extract LU WGSL from ebsdsim-web.js and copy ebsd WGSL sources."""

from __future__ import annotations

import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
JS = ROOT / "ebsdsim-web" / "dist" / "ebsdsim-web.js"
OUT = ROOT / "ebsdsim" / "wgsl"
SRC = ROOT / "ebsdsim-web" / "src" / "wgsl"

LU_NAMES = [
    "lu_factor_complex64",
    "lu_factor_lead_complex64",
    "lu_factor_upper_complex64",
    "lu_factor_trailing_complex64",
    "lu_solve_large_complex64",
    "lu_solve_shared_complex64",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    js = JS.read_text(encoding="utf-8")
    for name in LU_NAMES:
        m = re.search(rf'var {name}_default = "(.*?)";', js, re.DOTALL)
        if not m:
            raise SystemExit(f"missing {name}")
        wgsl = m.group(1).replace("\\n", "\n")
        path = OUT / f"{name}.wgsl"
        path.write_text(wgsl, encoding="utf-8")
        print("wrote", path.name, len(wgsl))
    for p in sorted(SRC.glob("ebsd-*.wgsl")):
        dst = OUT / (p.stem.replace("-", "_") + ".wgsl")
        shutil.copy2(p, dst)
        print("copied", dst.name)


if __name__ == "__main__":
    main()
