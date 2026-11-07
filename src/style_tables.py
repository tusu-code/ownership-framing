"""Colour styling for the generated LaTeX tables (run after every table script).

  python src/style_tables.py   -> rewrites paper/tables/<name>.tex in place for every table the paper inputs

Only results are coloured: headers and panel rows stay white. In tables without registered estimates, body rows are
shaded in alternating bands, one band per group (a group starts
at a row whose first cell is non-empty and differs from the previous group's, so an estimate row and the interval
row under it share a band); in tables with registered estimates there are no bands. Cells holding a
registered estimate (set in bold by the table scripts) are highlighted, together with the interval cell below
them, and the "yes"/"no" verdict cells of the registered tests are marked. Spacer rows (\\addlinespace) are dropped
because the bands separate the groups. Colours are defined in the paper preamble (thead, tband, tsec, tkey, tyes,
tno) and follow the figures' teal-to-khaki ramp. The script is idempotent: a styled file is left unchanged. Row bodies of the long tables (*_rows) get no marker
line and are read with \\inputrows (si.tex), because \\input cannot start a coloured longtable row.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "paper" / "tables"
MARK = "% styled by src/style_tables.py"
VERDICT = {"mitig", "detach"}        # tables whose last column is the registered verdict
RULE = re.compile(r"^\\(toprule|midrule|bottomrule|cmidrule|begin\{tabular\}|end\{tabular\})")


def cells(row):
    """Split a row at top-level '&' (outside braces)."""
    out, depth, cur = [], 0, ""
    for ch in row:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def is_section(row):
    return row.lstrip().startswith(r"\multicolumn") and len(cells(row)) == 1


def segments(lines):
    """Index ranges of row lines between rules, with the rule that closes each."""
    segs, start = [], None
    for i, l in enumerate(lines):
        s = l.strip()
        if RULE.match(s) and not s.startswith(r"\cmidrule"):
            if start is not None:
                segs.append((start, i, s))
            start = i + 1
    return segs


def style(text, name, body_only=False):
    if MARK in text or r"\rowcolor" in text:
        return text
    lines = [l for l in text.rstrip("\n").split("\n") if not l.strip().startswith(r"\addlinespace")]
    header = set()
    if not body_only:
        first_mid = True
        for a, b, closer in segments(lines):
            if not closer.startswith(r"\midrule"):
                continue
            rows = [i for i in range(a, b) if lines[i].strip().endswith("\\\\")]
            if first_mid or (rows and is_section(lines[rows[0]])):
                header.update(i for i in rows if not is_section(lines[i]))
            first_mid = False
    has_key = r"\boldmath" in text     # tables with registered estimates: colour only the results
    out, band, group, key_cols = [], 1, None, set()
    for i, l in enumerate(lines):
        s = l.strip()
        if not s.endswith("\\\\") or RULE.match(s):
            out.append(l)
            continue
        if is_section(l):
            out.append(l)
            band, group = 1, None
            continue
        if i in header:
            out.append(l)
            continue
        c = cells(l[: l.rindex("\\\\")])
        first = c[0].strip()
        if first and not first.startswith(r"\quad") and first != group:
            band, group, key_cols = 1 - band, first, set()
        bold = {j for j, x in enumerate(c) if r"\boldmath" in x}
        if bold:
            key_cols = bold
        for j in key_cols:
            if j < len(c) and c[j].strip():
                c[j] = r"\cellcolor{tkey}" + c[j]
        if name in VERDICT and c[-1].strip() in ("yes", "no"):
            c[-1] = (r"\cellcolor{tyes}" if c[-1].strip() == "yes" else r"\cellcolor{tno}") + c[-1]
        row = " & ".join(x.strip() if j else x for j, x in enumerate(c)) + r" \\"
        out.append((r"\rowcolor{tband}" if band == 0 and not has_key else "") + row)
    return ("" if body_only else MARK + "\n") + "\n".join(out) + "\n"


def used():
    names = set()
    for f in ("main.tex", "si.tex"):
        p = ROOT / "paper" / f
        if p.exists():
            names |= set(re.findall(r"\\input\{tables/([A-Za-z_]+)", p.read_text()))
    return sorted(names)


if __name__ == "__main__":
    for n in used():
        p = TAB / f"{n}.tex"
        p.write_text(style(p.read_text(), n, body_only=n.endswith("_rows")))
        print("styled", p.name)
