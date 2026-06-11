"""CIF 1.1 tokenizer and crystal parser."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
DEFAULT_B_ISO_ANGSTROM_SQ = 0.5

ELEMENT_SYMBOLS: tuple[str, ...] = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
)

ATOMIC_NUMBERS: dict[str, int] = {sym: z for z, sym in enumerate(ELEMENT_SYMBOLS) if sym}
ATOMIC_NUMBERS["D"] = 1
ATOMIC_NUMBERS["T"] = 1


class _TK(enum.Enum):
    DATA_HEADER = enum.auto()
    SAVE_OPEN = enum.auto()
    SAVE_CLOSE = enum.auto()
    LOOP = enum.auto()
    TAG = enum.auto()
    VALUE = enum.auto()


@dataclass
class _Token:
    kind: _TK
    text: str


@dataclass
class CIFCellParameters:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group: int | None = None
    hm_symbol: str | None = None


@dataclass
class CIFAtomSite:
    label: str
    symbol: str
    atomic_number: int
    fract: tuple[float, float, float]
    occupancy: float
    b_iso: float


@dataclass
class CIFCrystal(CIFCellParameters):
    atom_sites: list[CIFAtomSite] = field(default_factory=list)
    sym_ops: list[str] = field(default_factory=lambda: ["x,y,z"])


@dataclass
class _CIFLoop:
    tags: list[str]
    rows: list[list[str]]


@dataclass
class _CIFBlock:
    name: str
    tags: dict[str, str]
    loops: list[_CIFLoop]


def _is_whitespace(c: str) -> bool:
    return c in " \t\r\n"


def _strip_bom(text: str) -> str:
    return text[1:] if text and ord(text[0]) == 0xFEFF else text


def _tokenize_cif(raw: str) -> list[_Token]:
    text = _strip_bom(raw)
    n = len(text)
    out: list[_Token] = []
    i = 0
    at_line_start = True

    while i < n:
        c = text[i]

        if c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue

        if c in " \t\r":
            i += 1
            continue
        if c == "\n":
            i += 1
            at_line_start = True
            continue

        if c == ";" and at_line_start:
            i += 1
            start = i
            end_body = -1
            while i < n:
                if text[i] == "\n":
                    j = i + 1
                    if j < n and text[j] == ";":
                        end_body = i
                        i = j + 1
                        break
                i += 1
            if end_body < 0:
                raise ValueError("CIF: unterminated semicolon-text block")
            body = text[start:end_body]
            if body.startswith("\r\n"):
                body = body[2:]
            elif body.startswith("\n"):
                body = body[1:]
            if body.endswith("\r\n"):
                body = body[:-2]
            elif body.endswith("\n"):
                body = body[:-1]
            out.append(_Token(_TK.VALUE, body))
            at_line_start = False
            continue

        if c in "'\"":
            quote = c
            i += 1
            start = i
            closed = False
            while i < n:
                if text[i] == quote:
                    nxt = "\n" if i + 1 >= n else text[i + 1]
                    if _is_whitespace(nxt) or i + 1 >= n:
                        out.append(_Token(_TK.VALUE, text[start:i]))
                        i += 1
                        closed = True
                        break
                if text[i] == "\n":
                    break
                i += 1
            if not closed:
                raise ValueError(f"CIF: unterminated quoted string starting at offset {start - 1}")
            at_line_start = False
            continue

        start = i
        while i < n and not _is_whitespace(text[i]):
            i += 1
        tok = text[start:i]
        lower = tok.lower()
        if lower.startswith("data_") and len(lower) > 5:
            out.append(_Token(_TK.DATA_HEADER, tok[5:]))
        elif lower == "loop_":
            out.append(_Token(_TK.LOOP, ""))
        elif lower == "save_":
            out.append(_Token(_TK.SAVE_CLOSE, ""))
        elif lower.startswith("save_"):
            out.append(_Token(_TK.SAVE_OPEN, tok[5:]))
        elif lower in ("global_", "stop_"):
            pass
        elif tok.startswith("_"):
            out.append(_Token(_TK.TAG, lower))
        else:
            out.append(_Token(_TK.VALUE, tok))
        at_line_start = False
    return out


def _build_blocks(tokens: list[_Token]) -> list[_CIFBlock]:
    blocks: list[_CIFBlock] = []
    block: _CIFBlock | None = None
    in_save_frame = False
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.kind == _TK.DATA_HEADER:
            block = _CIFBlock(name=t.text, tags={}, loops=[])
            blocks.append(block)
            in_save_frame = False
            i += 1
            continue
        if t.kind == _TK.SAVE_OPEN:
            in_save_frame = True
            i += 1
            continue
        if t.kind == _TK.SAVE_CLOSE:
            in_save_frame = False
            i += 1
            continue
        if block is None or in_save_frame:
            i += 1
            continue
        if t.kind == _TK.TAG:
            tag = t.text
            i += 1
            if i >= n or tokens[i].kind != _TK.VALUE:
                raise ValueError(f"CIF: tag {tag} has no value")
            block.tags[tag] = tokens[i].text
            i += 1
            continue
        if t.kind == _TK.LOOP:
            i += 1
            loop_tags: list[str] = []
            while i < n and tokens[i].kind == _TK.TAG:
                loop_tags.append(tokens[i].text)
                i += 1
            if not loop_tags:
                raise ValueError("CIF: loop_ has no tags")
            flat_values: list[str] = []
            while i < n and tokens[i].kind == _TK.VALUE:
                flat_values.append(tokens[i].text)
                i += 1
            stride = len(loop_tags)
            full_rows = len(flat_values) // stride
            rows = [
                flat_values[r * stride : (r + 1) * stride] for r in range(full_rows)
            ]
            block.loops.append(_CIFLoop(tags=loop_tags, rows=rows))
            continue
        i += 1
    return blocks


def _parse_file(text: str) -> _CIFBlock:
    blocks = _build_blocks(_tokenize_cif(text))
    if not blocks:
        raise ValueError("CIF has no data_ block")
    return blocks[0]


def _is_missing(raw: str | None) -> bool:
    return raw is None or raw in ("?", ".")


def _strip_uncertainty(s: str) -> str:
    return re.sub(r"\(\d+\)$", "", s)


def _number_value(raw: str | None, label: str) -> float:
    if _is_missing(raw):
        raise ValueError(f"CIF is missing {label}")
    cleaned = _strip_uncertainty(raw.strip())
    value = float(cleaned)
    if not np_isfinite(value):
        raise ValueError(f"CIF {label} is not numeric ({raw})")
    return value


def _number_value_or(raw: str | None, fallback: float) -> float:
    if _is_missing(raw):
        return fallback
    cleaned = _strip_uncertainty(raw.strip())
    try:
        value = float(cleaned)
    except ValueError:
        return fallback
    return value if np_isfinite(value) else fallback


def np_isfinite(x: float) -> bool:
    import math
    return math.isfinite(x)


def _clean_symbol(raw: str) -> str:
    stripped = re.sub(r"[0-9+\-.\(\)\[\]].*$", "", raw)
    m = re.match(r"^([A-Za-z]{1,2})", stripped)
    if not m:
        raise ValueError(f"Could not infer atom symbol from '{raw}'")
    candidate = m.group(1)[0].upper() + m.group(1)[1:].lower()
    if candidate in ATOMIC_NUMBERS:
        return candidate
    one_letter = candidate[0]
    if one_letter not in ATOMIC_NUMBERS:
        raise ValueError(f"Unsupported atom symbol '{candidate}' (from raw '{raw}')")
    return one_letter


def _cell_tags_from_block(block: _CIFBlock) -> CIFCellParameters:
    def tag(*names: str) -> str | None:
        for name in names:
            if name in block.tags:
                return block.tags[name]
        return None

    sg_raw = tag(
        "_space_group_it_number",
        "_symmetry_int_tables_number",
        "_space_group.it_number",
    )
    hm_raw = tag(
        "_space_group_name_h-m_alt",
        "_space_group.name_h-m_alt",
        "_symmetry_space_group_name_h-m",
        "_space_group_name_h-m",
    )
    return CIFCellParameters(
        a=_number_value(tag("_cell_length_a"), "_cell_length_a"),
        b=_number_value(tag("_cell_length_b"), "_cell_length_b"),
        c=_number_value(tag("_cell_length_c"), "_cell_length_c"),
        alpha=_number_value(tag("_cell_angle_alpha"), "_cell_angle_alpha"),
        beta=_number_value(tag("_cell_angle_beta"), "_cell_angle_beta"),
        gamma=_number_value(tag("_cell_angle_gamma"), "_cell_angle_gamma"),
        space_group=None if _is_missing(sg_raw) else int(_number_value(sg_raw, "space group number")),
        hm_symbol=None if _is_missing(hm_raw) else hm_raw.strip(),
    )


def parse_cif_cell_parameters(text: str) -> CIFCellParameters:
    return _cell_tags_from_block(_parse_file(text))


def parse_cif_crystal(text: str) -> CIFCrystal:
    block = _parse_file(text)
    cell = _cell_tags_from_block(block)

    atom_loop = next(
        (
            loop
            for loop in block.loops
            if any(t.startswith("_atom_site_") and t != "_atom_site_aniso_label" for t in loop.tags)
        ),
        None,
    )
    site_loop = next(
        (
            loop
            for loop in block.loops
            if "_atom_site_fract_x" in loop.tags
            and "_atom_site_fract_y" in loop.tags
            and "_atom_site_fract_z" in loop.tags
        ),
        atom_loop,
    )
    if site_loop is None:
        raise ValueError("CIF is missing an atom_site loop with fractional coordinates")

    def idx(*names: str) -> int:
        for name in names:
            if name in site_loop.tags:
                return site_loop.tags.index(name)
        return -1

    label_idx = idx("_atom_site_label")
    symbol_idx = idx("_atom_site_type_symbol")
    x_idx = idx("_atom_site_fract_x")
    y_idx = idx("_atom_site_fract_y")
    z_idx = idx("_atom_site_fract_z")
    occ_idx = idx("_atom_site_occupancy")
    b_idx = idx("_atom_site_b_iso_or_equiv")
    u_idx = idx("_atom_site_u_iso_or_equiv")
    if x_idx < 0 or y_idx < 0 or z_idx < 0:
        raise ValueError("CIF atom_site loop must include fract_x/fract_y/fract_z")

    atom_sites: list[CIFAtomSite] = []
    import math

    for i, row in enumerate(site_loop.rows):
        label_raw = row[label_idx] if label_idx >= 0 else f"site{i}"
        symbol_raw = row[symbol_idx] if symbol_idx >= 0 and not _is_missing(row[symbol_idx]) else label_raw
        symbol = _clean_symbol(symbol_raw)
        if b_idx >= 0 and not _is_missing(row[b_idx]):
            b_iso = _number_value(row[b_idx], "_atom_site_b_iso_or_equiv")
        elif u_idx >= 0 and not _is_missing(row[u_idx]):
            b_iso = _number_value(row[u_idx], "_atom_site_u_iso_or_equiv") * 8 * math.pi * math.pi
        else:
            b_iso = DEFAULT_B_ISO_ANGSTROM_SQ
        atom_sites.append(
            CIFAtomSite(
                label=label_raw,
                symbol=symbol,
                atomic_number=ATOMIC_NUMBERS[symbol],
                fract=(
                    _number_value(row[x_idx], "_atom_site_fract_x"),
                    _number_value(row[y_idx], "_atom_site_fract_y"),
                    _number_value(row[z_idx], "_atom_site_fract_z"),
                ),
                occupancy=_number_value_or(row[occ_idx], 1.0) if occ_idx >= 0 else 1.0,
                b_iso=b_iso,
            )
        )

    symop_tags = [
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    ]
    sym_loop = next((loop for loop in block.loops if any(t in symop_tags for t in loop.tags)), None)
    sym_ops = ["x,y,z"]
    if sym_loop is not None:
        sym_idx = next(i for i, t in enumerate(sym_loop.tags) if t in symop_tags)
        sym_ops = [row[sym_idx] for row in sym_loop.rows]

    return CIFCrystal(
        a=cell.a,
        b=cell.b,
        c=cell.c,
        alpha=cell.alpha,
        beta=cell.beta,
        gamma=cell.gamma,
        space_group=cell.space_group,
        hm_symbol=cell.hm_symbol,
        atom_sites=atom_sites,
        sym_ops=sym_ops,
    )
