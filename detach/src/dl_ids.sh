#!/bin/sh
# Third registered study: download first-version arXiv PDFs for a list of ids (one per line).
# Usage: sh dl_ids.sh <root> <id list> <done flag>. export.arxiv.org mirror, 300 s cap, 2 attempts, 3 s pause.
# A file counts only if it starts with %PDF and ends with %%EOF (no truncated files).
cd "$1"
complete() { [ -s "$1" ] && [ "$(head -c 4 "$1")" = "%PDF" ] && tail -c 2048 "$1" | grep -q '%%EOF'; }
while read a; do
  [ -z "$a" ] && continue
  p=detach/data/pdfs/$a.pdf
  if complete "$p"; then continue; fi
  rm -f "$p"
  for t in 1 2; do
    if curl -s -L -m 300 -A "ownership-framing-study/1.0 (mailto:anonymous@example.com)" -o "$p.part" "https://export.arxiv.org/pdf/${a}v1" && complete "$p.part"; then
      mv "$p.part" "$p"; break
    fi
    rm -f "$p.part"; sleep $((8 * t))
  done
  if [ -s "$p" ]; then echo "$a ok"; else echo "$a failed"; fi
  sleep 3
done < "$2"
touch "$3"
echo DONE
