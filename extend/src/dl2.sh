#!/bin/sh
# Extension study A: download PDFs in frame order, first N per field, fields interleaved (round robin).
# Usage: sh dl2.sh <root> <N> <part>   part 0/1: list positions with index % 2 == part; -1: all positions.
# Files over 10 MB are skipped (arXiv throttles large downloads); they count as failed downloads.
# A file counts only if curl finished without error, it starts with %PDF and ends with %%EOF (no truncated files).
cd "$1"
${PYTHON:-python3} - "$2" "$3" > extend/data/dl_list_$3.txt <<'PY'
import csv, sys
n, part = int(sys.argv[1]), int(sys.argv[2])
rows = list(csv.DictReader(open("extend/data/frame.csv")))
by = {f: sorted([r for r in rows if r["field"] == f], key=lambda x: int(x["order"])) for f in ("biology", "physics", "economics")}
seq = [(f, by[f][i]) for i in range(n) for f in by]
for k, (f, r) in enumerate(seq):
    if part < 0 or k % 2 == part:
        print(f, r["order"], r["arxiv_id"])
PY
complete() { [ -s "$1" ] && [ "$(head -c 4 "$1")" = "%PDF" ] && tail -c 2048 "$1" | grep -q '%%EOF'; }
while read f o a; do
  p=extend/data/pdfs/$a.pdf
  if complete "$p"; then continue; fi
  rm -f "$p"
  for t in 1 2; do
    if curl -s -L -m 300 --max-filesize 10000000 -A "ownership-framing-study/1.0 (mailto:anonymous@example.com)" -o "$p.part" "https://export.arxiv.org/pdf/${a}v1" && complete "$p.part"; then
      mv "$p.part" "$p"; break
    fi
    rm -f "$p.part"; sleep $((8 * t))
  done
  if [ -s "$p" ]; then echo "$f $o $a ok"; else echo "$f $o $a failed"; fi
  sleep 3
done < extend/data/dl_list_$3.txt
touch extend/data/dl_$3.done
echo DONE
