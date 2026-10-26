#!/usr/bin/env bash
# Build report/report.pdf from report/report.md (needs pandoc and xelatex). Run from the repository root.
set -e
cd "$(dirname "$0")"
pandoc report.md -o report.pdf --pdf-engine=xelatex --resource-path=.:.. -V linkcolor=blue
echo "wrote report/report.pdf"
