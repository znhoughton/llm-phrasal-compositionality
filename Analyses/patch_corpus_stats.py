"""
Patch corpus statistics in existing result CSVs
=================================================
Replaces frequency and predictability values in pre-computed result CSVs with
values from the new olmo_corpus_stats.pkl (Dolma v1.7 via infini-gram), and
renames the legacy 'ftp' column to 'predic' where present.

Also exports ftp_lookup.csv for the Whisper R analysis.

Files patched:
    Data/olmo-3-7b/Data_up/all_layers_results.csv        frequency + ftp→predic
    Data/olmo-3-7b/Data_up/test.csv                      frequency + ftp→predic
    Data/olmo-3-7b/Data_upsubword/all_layers_results.csv frequency + ftp→predic
    Data/olmo-3-7b/Data_upsubword/test.csv               frequency + ftp→predic
    Data/whisper/encoder/all_layers_results.csv           frequency only
    Data/whisper/decoder/all_layers_results.csv           frequency only

Also writes:
    Data/ftp_lookup.csv    verb_up, ftp  (read by R; R renames ftp→predic on load)

Usage:
    python patch_corpus_stats.py [--stats-pkl PATH] [--data-root PATH]
"""

import argparse
import csv
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch frequency/predictability in result CSVs from olmo_corpus_stats.pkl."
    )
    parser.add_argument(
        "--stats-pkl",
        default="../Data/olmo_corpus_stats.pkl",
        help="Path to olmo_corpus_stats.pkl. Default: ../Data/olmo_corpus_stats.pkl",
    )
    parser.add_argument(
        "--data-root",
        default="../Data",
        help="Root Data directory. Default: ../Data",
    )
    return parser.parse_args()


def patch_csv(path: Path, vup_freq: dict, predic: dict, has_predic_col: bool):
    """
    Overwrite path in-place with updated frequency (and optionally predic) values.
    Renames legacy 'ftp' column to 'predic' if present.
    Rows whose verb_up is not in the new stats are left with their original values.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fields = reader.fieldnames or []

    # Rename ftp → predic in column list
    fields = ["predic" if c == "ftp" else c for c in original_fields]

    n_freq = n_predic = n_missing = 0
    for row in rows:
        vup = row.get("verb_up", "").strip()
        if vup in vup_freq:
            row["frequency"] = str(vup_freq[vup])
            n_freq += 1
        else:
            n_missing += 1

        if has_predic_col:
            src_col = "predic" if "predic" in row else "ftp"
            if vup in predic:
                row[src_col] = str(predic[vup])
                n_predic += 1
            # rename key if needed
            if src_col == "ftp" and "ftp" in row:
                row["predic"] = row.pop("ftp")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    msg = f"  {path.name}: {n_freq} freq updated"
    if has_predic_col:
        msg += f", {n_predic} predic updated"
    if n_missing:
        msg += f"  ({n_missing} verb_up not in new stats — left unchanged)"
    print(msg)


def export_ftp_lookup(predic: dict, out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["verb_up", "ftp"])
        for vup_type, p in sorted(predic.items()):
            writer.writerow([vup_type, p])
    print(f"  ftp_lookup.csv: {len(predic)} rows written")


def main():
    args = parse_args()
    data = Path(args.data_root)

    print(f"Loading {args.stats_pkl} ...")
    with open(args.stats_pkl, "rb") as f:
        vup_freq, verb_freq, predic = pickle.load(f)
    print(f"  {len(vup_freq)} V+up types, {len(predic)} with predictability")

    targets = [
        # (relative path,                                    has predic col)
        (data / "olmo-3-7b/Data_up/all_layers_results.csv",        True),
        (data / "olmo-3-7b/Data_up/test.csv",                      True),
        (data / "olmo-3-7b/Data_upsubword/all_layers_results.csv",  True),
        (data / "olmo-3-7b/Data_upsubword/test.csv",                True),
        (data / "whisper/encoder/all_layers_results.csv",           False),
        (data / "whisper/decoder/all_layers_results.csv",           False),
    ]

    print("\nPatching CSVs ...")
    for path, has_predic in targets:
        if path.exists():
            patch_csv(path, vup_freq, predic, has_predic)
        else:
            print(f"  SKIPPED (not found): {path}")

    print("\nExporting ftp_lookup.csv ...")
    export_ftp_lookup(predic, data / "ftp_lookup.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
