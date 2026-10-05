#!/usr/bin/env python
"""
Confirm with three independent taggers that each test item really is a verb followed by "up".

WHY. Reviewer mjt8 asked how reliable the parser is. The test set was built with spaCy, so
re-checking it with spaCy proves nothing about spaCy's blind spots. This tags the token before
"up" with three taggers that share no architecture, training corpus or codebase, and keeps only
the items all three independently call a verb.

    spacy_sm   spaCy en_core_web_sm        CNN,         OntoNotes 5      (built the test set)
    nltk_ap    NLTK averaged perceptron    perceptron,  Penn Treebank
    bert_pos   bert-english-...-pos        transformer, Universal Deps

All three see the SAME word tokenization, so their tags are directly comparable; without that,
disagreements would be tokenizer artefacts rather than real ones.

A token counts as a verb on a VERB/AUX coarse tag (spaCy, BERT) or a Penn tag starting with VB
(NLTK). Participles used adjectivally (*a revved up engine*) tag as ADJ/JJ and so surface as
disagreements rather than being silently kept or dropped.

Run with PRenv, the env that has spaCy and a CUDA torch:
    .../PRenv/python.exe Analyses/verify_verb_up.py --limit 2000      # smoke test
    .../PRenv/python.exe Analyses/verify_verb_up.py                   # full set
"""
import argparse
import os
import re
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(REPO, "Data", "olmo-3-7b", "Data_up", "all_layers_results_filtered.csv")
WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+|[^\sA-Za-z\d]")

t0 = time.time()
def log(msg):
    print("[%6.0fs] %s" % (time.time() - t0, msg), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_IN)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch", type=int, default=64, help="BERT batch size")
    p.add_argument("--out", default=os.path.join(REPO, "Analyses", "tagger_agreement.csv"))
    return p.parse_args()


def verb_before_up(words, isverb, verb):
    """Is the token immediately before a matching 'up' tagged as a verb?

    Returns None when "<verb> up" cannot be located at all, which is reported separately rather
    than being lumped in with "not a verb".
    """
    v = verb.lower()
    for i in range(1, len(words)):
        if words[i].lower() == "up" and words[i - 1].lower() == v:
            return bool(isverb[i - 1])
    return None


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    df = df[df.layer == df.layer.min()][["verb_up", "frequency", "sentence"]].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    df["verb"] = df.verb_up.astype(str).str.split().str[0]
    toks = [WORD.findall(str(s)) for s in df.sentence]
    log("tokenized %d items" % len(df))

    # ---- 1. spaCy, on the shared tokenization ----------------------------------------------
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "parser"])
    out = []
    for doc, ws, v in zip(nlp.pipe((Doc(nlp.vocab, words=w) for w in toks), batch_size=256),
                          toks, df.verb):
        out.append(verb_before_up(ws, [t.pos_ in ("VERB", "AUX") for t in doc], v))
    df["spacy_sm"] = out
    log("spacy_sm done")

    # ---- 2. NLTK averaged perceptron -------------------------------------------------------
    import nltk
    df["nltk_ap"] = [verb_before_up(ws, [p.startswith("VB") for _, p in nltk.pos_tag(ws)], v)
                     for ws, v in zip(toks, df.verb)]
    log("nltk_ap done")

    # ---- 3. BERT POS, aligned back to the shared words -------------------------------------
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    name = "vblagoje/bert-english-uncased-finetuned-pos"
    tk = AutoTokenizer.from_pretrained(name)
    md = AutoModelForTokenClassification.from_pretrained(name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    md = md.to(dev).eval()
    log("bert loaded on %s" % dev)
    id2label = md.config.id2label
    res = []
    for s in range(0, len(toks), args.batch):
        chunk = [w[:400] for w in toks[s:s + args.batch]]          # BERT's 512-token ceiling
        enc = tk(chunk, is_split_into_words=True, truncation=True, max_length=512,
                 padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            pred = md(**enc).logits.argmax(-1).cpu()
        for b in range(len(chunk)):
            wid = enc.word_ids(batch_index=b)
            isv = [False] * len(chunk[b])
            for pos, w in enumerate(wid):                           # first subword wins
                if w is not None and not isv[w]:
                    isv[w] = id2label[int(pred[b][pos])] in ("VERB", "AUX")
            res.append(isv)
        if s and s % (args.batch * 100) == 0:
            log("  bert %d/%d" % (s, len(toks)))
    df["bert_pos"] = [verb_before_up(w[:400], iv, v) for w, iv, v in zip(toks, res, df.verb)]
    log("bert_pos done")

    cols = ["spacy_sm", "nltk_ap", "bert_pos"]
    df["n_verb"] = sum((df[c] == True).astype(int) for c in cols)
    df["n_located"] = sum(df[c].notna().astype(int) for c in cols)
    df["all_agree_verb"] = df.n_verb == 3
    df.to_csv(args.out, index=False, encoding="utf-8")

    n = len(df)
    print()
    for c in cols:
        print("  %-9s calls it a verb : %6d  (%5.2f%%)" % (c, (df[c] == True).sum(),
                                                           100 * (df[c] == True).mean()))
    print("  ---")
    print("  all three agree       : %6d  (%5.2f%%)" % (df.all_agree_verb.sum(),
                                                        100 * df.all_agree_verb.mean()))
    print("  two of three          : %6d" % (df.n_verb == 2).sum())
    print("  one of three          : %6d" % (df.n_verb == 1).sum())
    print("  none                  : %6d" % (df.n_verb == 0).sum())
    print("  'V up' not locatable  : %6d" % (df.n_located < 3).sum())
    print("  wrote %s" % args.out)


if __name__ == "__main__":
    main()
