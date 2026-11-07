"""Convert the downloaded SVG icons into the PDF (TikZ) and PNG (matplotlib) files the figures use.

  python src/make_icons.py   -> paper/icons/pdf/*.pdf, paper/icons/png/*.png

Sources (see paper/icons/LICENSES.md): model logos from Lobe Icons (@lobehub/icons-static-svg, MIT);
line icons from Lucide (lucide-static, ISC). Line icons are recoloured per ask so that the design
figure uses the same colour for each ask as the result figures.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "paper" / "icons"
TMP = ICONS / "tmp"

# Colour per ask, shared with src/paper_figs.py.
ASK = {"author": "#C8553D", "colleague": "#E09F3E", "venue": "#7A7A7A", "reviewer": "#2E6F9E"}
INK = "#2E3440"

LINE = {  # output name: (lucide icon, stroke colour)
    "paper": ("file-text", INK), "mask": ("venetian-mask", INK),
    "author": ("pen-line", ASK["author"]), "colleague": ("users-round", ASK["colleague"]),
    "venue": ("message-square", ASK["venue"]), "reviewer": ("clipboard-check", ASK["reviewer"]),
    "open": ("message-square", INK), "neutral": ("eye-off", INK), "judge": ("scale", INK),
    "form": ("clipboard-list", INK),
}
LOGO = {"claude": "claude-color", "gpt": "openai", "glm": "zhipu-color", "deepseek": "deepseek-color",
        "gemini": "gemini-color", "grok": "grok"}


def sized(svg):
    svg = re.sub(r'width="(1em|24)"', 'width="96"', svg, count=1)
    return re.sub(r'height="(1em|24)"', 'height="96"', svg, count=1)


def render(svg, name):
    TMP.mkdir(parents=True, exist_ok=True)
    src = TMP / f"{name}.svg"
    src.write_text(svg)
    subprocess.run(["rsvg-convert", "-f", "pdf", str(src), "-o", str(ICONS / "pdf" / f"{name}.pdf")], check=True)
    subprocess.run(["rsvg-convert", "-w", "512", "-h", "512", str(src), "-o", str(ICONS / "png" / f"{name}.png")], check=True)


def main():
    (ICONS / "pdf").mkdir(exist_ok=True)
    (ICONS / "png").mkdir(exist_ok=True)
    for name, (icon, colour) in LINE.items():
        svg = (ICONS / "svg" / f"lucide-{icon}.svg").read_text()
        render(sized(svg).replace('stroke="currentColor"', f'stroke="{colour}"'), f"ic_{name}")
    for model, icon in LOGO.items():
        svg = sized((ICONS / "svg" / f"{icon}.svg").read_text()).replace('fill="currentColor"', 'fill="#111111"')
        render(svg, f"logo_{model}")
    print("ok")


if __name__ == "__main__":
    main()
