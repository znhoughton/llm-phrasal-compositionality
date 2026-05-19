import pickle, csv

print("=== olmo_corpus_stats.pkl ===")
with open("../Data/olmo_corpus_stats.pkl", "rb") as f:
    vup_freq, verb_freq, predic = pickle.load(f)
print(f"  vup_freq: {len(vup_freq)} types, top 3: {list(vup_freq.most_common(3))}")
print(f"  predic:   {len(predic)} types")

print("\n=== babylm_corpus_stats.pkl ===")
with open("../Data/babylm_corpus_stats.pkl", "rb") as f:
    vup_freq_b, verb_freq_b, predic_b = pickle.load(f)
print(f"  vup_freq: {len(vup_freq_b)} types, top 3: {list(vup_freq_b.most_common(3))}")
print(f"  predic:   {len(predic_b)} types")

print("\n=== ftp_lookup.csv ===")
with open("../Data/ftp_lookup.csv") as f:
    rows = list(csv.reader(f))
print(f"  rows: {len(rows)-1}, header: {rows[0]}, first entry: {rows[1]}")