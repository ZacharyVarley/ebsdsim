"""Regenerate ebsdsim/_pg_ops_data.py from ebsdsim-web/src/pg-ops-data.ts."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "ebsdsim-web" / "src" / "pg-ops-data.ts"
OUT = ROOT / "ebsdsim" / "_pg_ops_data.py"


def _parse_float64_arrays(block: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for m in re.finditer(r"'([^']+)':\s*new Float64Array\(\[([^\]]+)\]\)", block):
        key = m.group(1)
        vals = [float(x.strip()) for x in m.group(2).split(",") if x.strip()]
        out[key] = vals
    return out


def main() -> None:
    text = TS.read_text(encoding="utf-8")
    sym_match = re.search(
        r"PG_NUM_TO_SYMBOL.*?=\s*\[(.*?)\]",
        text,
        re.DOTALL,
    )
    if not sym_match:
        raise RuntimeError("PG_NUM_TO_SYMBOL not found")
    symbols = re.findall(r"'([^']+)'", sym_match.group(1))

    ops_block = text.split("export const PG_OPERATORS")[1].split("export const FS_NORMALS")[0]
    fs_block = text.split("export const FS_NORMALS")[1]
    ops = _parse_float64_arrays(ops_block)
    fs = _parse_float64_arrays(fs_block)

    lines = [
        '"""AUTO-GENERATED from pg-ops-data.ts — do not edit by hand."""',
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "",
        f"PG_NUM_TO_SYMBOL = {symbols!r}",
        "PG_OPERATORS = {",
    ]
    for key, vals in ops.items():
        arr = ", ".join(f"{v:.16g}" for v in vals)
        lines.append(f"    {key!r}: np.array([{arr}], dtype=np.float64),")
    lines.append("}")
    lines.append("FS_NORMALS = {")
    for key, vals in fs.items():
        arr = ", ".join(f"{v:.16g}" for v in vals)
        lines.append(f"    {key!r}: np.array([{arr}], dtype=np.float64),")
    lines.append("}")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(ops)} PG ops, {len(fs)} FS normal sets)")


if __name__ == "__main__":
    main()
