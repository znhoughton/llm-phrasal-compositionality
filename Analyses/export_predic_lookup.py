"""
Export predictability lookup CSV
=================================
Exports ftp_lookup.csv from olmo_corpus_stats.pkl for use by the R analysis.
The CSV has two columns: verb_up, ftp (raw P(up|V) probability).

The column is named 'ftp' for backward compatibility with the R join; the R
script renames it to 'predic' on load.

Usage:
    python export_predic_lookup.py [--in-pkl PATH] [--out-csv PATH]
"""

import argparse
import csv
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export predictability lookup CSV from olmo_corpus_stats.pkl."
    )
    parser.add_argument(
        "--in-pkl",
        default="../Data/olmo_corpus_stats.pkl",
        help="Path to olmo_corpus_stats.pkl. Default: ../Data/olmo_corpus_stats.pkl",
    )
    parser.add_argument(
        "--out-csv",
        default="../Data/ftp_lookup.csv",
        help="Output CSV path. Default: ../Data/ftp_lookup.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.in_pkl, "rb") as f:
        vup_freq, verb_freq, predic = pickle.load(f)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["verb_up", "ftp"])
        for vup_type, p in sorted(predic.items()):
            writer.writerow([vup_type, p])

    print(f"Wrote {len(predic)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
