#!/usr/bin/env bash
#
# Build both versions of the paper from the single writeup.qmd.
#
#   writeup-anonymous.pdf   ARR submission. acl-mode: review, so acl.sty swaps the author
#                           block for "Anonymous ACL submission" and adds line numbers.
#                           The \publiclinks and \repolink footnotes stay undefined, so the
#                           HuggingFace and GitHub URLs do not appear.
#   writeup-arxiv.pdf       Named version, both link footnotes shown.
#   arxiv-submission.zip    What arXiv wants: the .tex, a pre-built .bbl (arXiv does not
#                           reliably run bibtex), the style files, and ONLY the figures the
#                           .tex actually includes.
#
# Usage:  bash render-both.sh
set -euo pipefail

QMD=writeup.qmd
RS="${RSCRIPT:-C:/Program Files/R/R-4.5.2/bin/x64/Rscript.exe}"
export QUARTO_R="$RS"
OUT=arxiv-submission

command -v quarto >/dev/null || { echo "quarto not on PATH" >&2; exit 1; }
[ -f "$QMD" ] || { echo "run me from the paper/ directory" >&2; exit 1; }

echo "== 1/4  anonymous build (ARR) =="
quarto render "$QMD"
cp writeup.pdf writeup-anonymous.pdf
echo "   -> writeup-anonymous.pdf"

echo "== 2/4  named build (arXiv) =="
# acl-mode:final shows the author block; the two flags enable the link footnotes.
quarto render "$QMD" -M acl-mode:final -M public-links:true -M repo-link:true
cp writeup.pdf writeup-arxiv.pdf
echo "   -> writeup-arxiv.pdf"

echo "== 3/4  bibliography (.bbl) =="
# Quarto cleans up after itself, so run the latex/bibtex cycle in a scratch dir to
# capture the .bbl that arXiv needs shipped alongside the source.
TMP=$(mktemp -d)
cp writeup.tex acl.sty acl_natbib.bst references.bib "$TMP"/
mkdir -p "$TMP"/writeup_files
cp -r writeup_files/* "$TMP"/writeup_files/ 2>/dev/null || true
( cd "$TMP" && pdflatex -interaction=nonstopmode writeup.tex >/dev/null 2>&1 || true
  bibtex writeup >/dev/null 2>&1 || true )
[ -s "$TMP/writeup.bbl" ] || { echo "   FAILED: no .bbl produced" >&2; exit 1; }
cp "$TMP/writeup.bbl" ./writeup.bbl
echo "   -> writeup.bbl ($(wc -l < writeup.bbl) lines)"

echo "== 4/4  arxiv-submission bundle =="
rm -rf "$OUT" arxiv-submission.zip
mkdir -p "$OUT/writeup_files/figure-pdf"
cp writeup.tex writeup.bbl acl.sty acl_natbib.bst references.bib "$OUT"/
# Copy only the figures the .tex includes, not the whole writeup_files tree.
n=0
for f in $(grep -oE "includegraphics[^{]*[{][^}]+[}]" writeup.tex | sed 's/.*[{]//; s/[}]//'); do
  for cand in "$f" "$f.pdf" "$f.png"; do
    if [ -f "$cand" ]; then
      mkdir -p "$OUT/$(dirname "$cand")"
      cp "$cand" "$OUT/$cand"; n=$((n+1)); break
    fi
  done
done
echo "   copied $n figure file(s)"
( cd "$OUT" && zip -qr ../arxiv-submission.zip . )
echo "   -> arxiv-submission.zip ($(du -h arxiv-submission.zip | cut -f1))"

# Leave the working tree on the anonymous build so a stray `quarto render` does not
# silently overwrite writeup.pdf with the named one.
quarto render "$QMD" >/dev/null
echo
echo "done:"
echo "   writeup-anonymous.pdf   -> ARR submission"
echo "   writeup-arxiv.pdf       -> preview of the named version"
echo "   arxiv-submission.zip    -> upload to arXiv"
