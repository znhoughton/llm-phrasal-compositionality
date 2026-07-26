"""
Validate reconstruct_up_audio_metadata.py's output against the real
Data/up-audio-metadata.csv.

Reports, in order of how diagnostic they are if something's wrong:
  1. Row counts (overall and by ds_source) -- gross coverage differences
     usually mean a GigaSpeech subset/split mismatch, not a logic bug.
  2. Unique sid overlap -- are we even looking at the same segments?
  3. Exact (sid, matched_phrase) match rate -- the real test of whether the
     matching LOGIC (not just corpus coverage) is right.
  4. For matched (sid, matched_phrase) pairs, whether begin_time/end_time/
     cleaned_text also agree exactly.
  5. Concrete mismatch examples (both directions) for debugging.

Usage:
    python validate_reconstruction.py --reconstructed Data/up-audio-metadata-reconstructed.csv
"""

import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default="Data/up-audio-metadata.csv")
    parser.add_argument("--reconstructed", required=True)
    parser.add_argument("--n-examples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    real = pd.read_csv(args.real)
    recon = pd.read_csv(args.reconstructed)

    print("=" * 70)
    print("1. ROW COUNTS")
    print("=" * 70)
    print(f"Real:          {len(real):,} rows")
    print(f"Reconstructed: {len(recon):,} rows")
    print()
    print("By ds_source (real):")
    print(real["ds_source"].value_counts().to_string())
    print()
    print("By ds_source (reconstructed):")
    print(recon["ds_source"].value_counts().to_string())

    print()
    print("=" * 70)
    print("2. UNIQUE SID OVERLAP")
    print("=" * 70)
    real_sids = set(real["sid"].astype(str))
    recon_sids = set(recon["sid"].astype(str))
    print(f"Real unique sids:          {len(real_sids):,}")
    print(f"Reconstructed unique sids: {len(recon_sids):,}")
    print(f"Overlap:                   {len(real_sids & recon_sids):,}")
    print(f"Real-only sids:            {len(real_sids - recon_sids):,}")
    print(f"Reconstructed-only sids:   {len(recon_sids - real_sids):,}")

    print()
    print("=" * 70)
    print("3. EXACT (sid, matched_phrase) MATCH RATE")
    print("=" * 70)
    real_pairs = set(zip(real["sid"].astype(str), real["matched_phrase"].astype(str)))
    recon_pairs = set(zip(recon["sid"].astype(str), recon["matched_phrase"].astype(str)))
    overlap = real_pairs & recon_pairs
    print(f"Real (sid, matched_phrase) pairs:          {len(real_pairs):,}")
    print(f"Reconstructed (sid, matched_phrase) pairs: {len(recon_pairs):,}")
    print(f"Exact overlap:                              {len(overlap):,}")
    if real_pairs:
        print(f"Recall (fraction of real pairs recovered): {len(overlap) / len(real_pairs):.4f}")
    if recon_pairs:
        print(f"Precision (fraction of recon pairs correct): {len(overlap) / len(recon_pairs):.4f}")

    print()
    print("=" * 70)
    print("4. FOR MATCHED PAIRS: do begin_time/end_time/cleaned_text also agree?")
    print("=" * 70)
    if overlap:
        real_idx = real.set_index(["sid", "matched_phrase"])
        recon_idx = recon.set_index(["sid", "matched_phrase"])
        real_idx.index = real_idx.index.map(lambda x: (str(x[0]), str(x[1])))
        recon_idx.index = recon_idx.index.map(lambda x: (str(x[0]), str(x[1])))

        n_checked = 0
        n_time_match = 0
        n_text_match = 0
        for key in list(overlap)[:5000]:  # cap for speed on huge files
            try:
                r = real_idx.loc[key]
                c = recon_idx.loc[key]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[0]
            except KeyError:
                continue
            n_checked += 1
            if abs(float(r["begin_time"]) - float(c["begin_time"])) < 0.01:
                n_time_match += 1
            if str(r["cleaned_text"]).strip() == str(c["cleaned_text"]).strip():
                n_text_match += 1
        print(f"Checked {n_checked} matched pairs (capped at 5000 for speed):")
        if n_checked:
            print(f"  begin_time agrees:   {n_time_match}/{n_checked} ({n_time_match/n_checked:.2%})")
            print(f"  cleaned_text agrees: {n_text_match}/{n_checked} ({n_text_match/n_checked:.2%})")
    else:
        print("No overlapping pairs to check.")

    print()
    print("=" * 70)
    print(f"5. EXAMPLES ({args.n_examples} each direction)")
    print("=" * 70)
    real_only = real_pairs - recon_pairs
    recon_only = recon_pairs - real_pairs
    print(f"\nIn REAL but not reconstructed ({len(real_only):,} total):")
    for sid, mp in list(real_only)[: args.n_examples]:
        row = real[(real["sid"].astype(str) == sid) & (real["matched_phrase"].astype(str) == mp)].iloc[0]
        print(f"  sid={sid}  matched_phrase='{mp}'  text={str(row['cleaned_text'])[:80]}")

    print(f"\nIn reconstructed but not REAL ({len(recon_only):,} total):")
    for sid, mp in list(recon_only)[: args.n_examples]:
        row = recon[(recon["sid"].astype(str) == sid) & (recon["matched_phrase"].astype(str) == mp)].iloc[0]
        print(f"  sid={sid}  matched_phrase='{mp}'  text={str(row['cleaned_text'])[:80]}")

    print()
    print("=" * 70)
    print("DONE. If recall/precision are high and begin_time/cleaned_text agree,")
    print("the reconstruction is faithful and can be extended to search for")
    print("up-CONTAINING WORDS (not just standalone 'up') for the Experiment 2")
    print("replication. If coverage (section 1/2) is off but matching logic")
    print("(section 3/4) is solid on the overlapping segments, the fix is a")
    print("GigaSpeech subset/split adjustment, not a logic change.")
    print("=" * 70)


if __name__ == "__main__":
    main()
