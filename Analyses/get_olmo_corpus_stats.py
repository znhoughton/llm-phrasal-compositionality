"""
OLMo Corpus Statistics (Dolma via infini-gram)
===============================================
Computes verb surface-form frequencies and predictability
for the OLMo analysis, using the infini-gram API on the Dolma corpus
(index: v4_dolma-v1_7_llama).  OLMo 2 was trained on OLMo-Mix-1124, a
Dolma-derived mix.

    predictability("pick up") = count("pick up") / count("pick")

V+up types are loaded from corpus_results.pkl (produced by create_dataset.py).
Bigram and unigram counts are fetched from the infini-gram API.

Output saved to:
    ../Data/olmo_corpus_stats.pkl
    (vup_freq, verb_freq, predic)

    vup_freq  : Counter  {"pick up": 5000, ...}  (counts from Dolma)
    verb_freq : Counter  {"pick": 50000, ...}     (counts from Dolma)
    predic    : dict     {"pick up": 0.10, ...}

Usage:
    python get_olmo_corpus_stats.py [--vup-pkl PATH] [--out-pkl PATH]
                                    [--index INDEX] [--min-freq N]
                                    [--delay SECONDS]

Requires:
    pip install requests tqdm
"""

import argparse
import collections
import logging
import pickle
import time

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_URL   = "https://api.infini-gram.io/"
# Dolma v1.7 indexed with the LLaMA tokenizer.
DEFAULT_INDEX = "v4_dolma-v1_7_llama"
MIN_FREQ      = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute V+up frequencies and predictability from Dolma via infini-gram."
    )
    parser.add_argument(
        "--vup-pkl",
        default="../Data/corpus_results.pkl",
        help="Path to corpus_results.pkl (from create_dataset.py). "
             "Default: ../Data/corpus_results.pkl",
    )
    parser.add_argument(
        "--out-pkl",
        default="../Data/olmo_corpus_stats.pkl",
        help="Output path. Default: ../Data/olmo_corpus_stats.pkl",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"infini-gram index to query. Default: {DEFAULT_INDEX}",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=MIN_FREQ,
        help=f"Minimum V+up count to include. Default: {MIN_FREQ}",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API requests (default: 0.5).",
    )
    return parser.parse_args()


RATE_LIMIT_COOLDOWN = 60  # seconds to pause after any 403 before retrying


def infigram_count(query: str, index: str, session: requests.Session) -> int:
    """Return the raw count of `query` in the given infini-gram index."""
    payload = {"index": index, "query_type": "count", "query": query}
    for attempt in range(5):
        try:
            r = session.post(API_URL, json=payload, timeout=30)
            if r.status_code == 403:
                log.warning("Rate limited on %r (attempt %d) — cooling down for %ds",
                            query, attempt + 1, RATE_LIMIT_COOLDOWN)
                time.sleep(RATE_LIMIT_COOLDOWN)
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                log.warning("API error for %r: %s", query, data["error"])
                return 0
            return data.get("count", 0)
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** (attempt + 2)  # 4, 8, 16, 32, 64s
            log.warning("Request failed for %r (attempt %d): %s — retrying in %ds",
                        query, attempt + 1, e, wait)
            time.sleep(wait)
    log.error("Giving up on %r after 5 attempts — cooling down for %ds before next query.",
              query, RATE_LIMIT_COOLDOWN)
    time.sleep(RATE_LIMIT_COOLDOWN)
    return 0


def main():
    args = parse_args()

    log.info("Loading V+up types from %s ...", args.vup_pkl)
    with open(args.vup_pkl, "rb") as f:
        _, vup_freq_pkl, _, _ = pickle.load(f)

    # Use the V+up types from the pkl as the candidate set (they were identified
    # from C4 sentence scans); counts will come from Dolma via infini-gram.
    candidate_types = [t for t, c in vup_freq_pkl.items() if c >= args.min_freq]
    target_verbs    = sorted({t.split()[0] for t in candidate_types})
    log.info("V+up types to query: %d  |  unique verbs: %d",
             len(candidate_types), len(target_verbs))

    session = requests.Session()

    # --- Bigram counts (V+up) ---
    log.info("Fetching bigram counts from infini-gram [index=%s] ...", args.index)
    vup_freq = collections.Counter()
    for vup_type in tqdm(candidate_types, desc="Bigram counts"):
        vup_freq[vup_type] = infigram_count(vup_type, args.index, session)
        time.sleep(args.delay)

    # --- Unigram counts (verbs) ---
    log.info("Fetching unigram counts ...")
    verb_freq = collections.Counter()
    for verb in tqdm(target_verbs, desc="Unigram counts"):
        verb_freq[verb] = infigram_count(verb, args.index, session)
        time.sleep(args.delay)

    # --- predictability ---
    predic = {}
    n_missing = 0
    for vup_type in candidate_types:
        verb  = vup_type.split()[0]
        denom = verb_freq.get(verb, 0)
        if denom > 0 and vup_freq[vup_type] > 0:
            predic[vup_type] = vup_freq[vup_type] / denom
        else:
            n_missing += 1

    log.info("Predictability computed for %d V+up types (%d skipped — zero count).",
             len(predic), n_missing)
    log.info("Top 10 V+up types by Dolma frequency:")
    for vup_type, cnt in vup_freq.most_common(10):
        predic_val = predic.get(vup_type, float("nan"))
        log.info("  %-25s  freq=%8d  predictability=%.4f", vup_type, cnt, predic_val)

    with open(args.out_pkl, "wb") as f:
        pickle.dump((vup_freq, verb_freq, predic), f)
    log.info("Saved to %s", args.out_pkl)


if __name__ == "__main__":
    main()
