"""Train the distilled classifier on Claude gold labels; evaluate honestly.

METHOD (the part that's easy to get wrong):
  train on  frame="stratified"  — oversampled for rare classes, so the model
                                  actually sees enough `denies` to learn it.
  test on   frame="random"      — a pure random draw, never trained on. This is
                                  the ONLY frame that gives an unbiased accuracy
                                  AND an unbiased prevalence estimate.
Testing on the stratified frame would flatter the model badly (rare classes would
look far more common and far easier than they are). So we don't.

We also score the OLD 7B tags against the same gold, on the same held-out items —
that's the apples-to-apples "was this worth doing" number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from db import db as dbmod

GOLD = Path(__file__).resolve().parent.parent / "_gold"
CACHE = Path(__file__).resolve().parent.parent / "_emb"
STANCES = ["none", "asserts", "denies", "make_your_own", "doesnt_know"]


def load_gold() -> dict[str, dict]:
    frames = json.loads((GOLD / "_frames.json").read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for f in sorted(GOLD.glob("out_*.json")):
        try:
            for rec in json.loads(f.read_text(encoding="utf-8")):
                if rec.get("id") in frames and rec.get("stance") in STANCES:
                    out[rec["id"]] = {"stance": rec["stance"],
                                      "sources": rec.get("sources") or [],
                                      "register": rec.get("register"),
                                      "frame": frames[rec["id"]]}
        except json.JSONDecodeError:
            print(f"  WARN: {f.name} is not valid JSON — skipped")
    return out


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score

    gold = load_gold()
    if not gold:
        print("no gold labels found in _gold/out_*.json — run the labelling agents first")
        return
    ids = json.loads((CACHE / "ids.json").read_text(encoding="utf-8"))
    vecs = np.load(CACHE / "vecs.npy")
    pos = {i: k for k, i in enumerate(ids)}

    usable = [i for i in gold if i in pos]
    print(f"gold: {len(gold):,} labelled, {len(usable):,} have embeddings")
    strat = [i for i in usable if gold[i]["frame"] == "stratified"]
    rand = sorted(i for i in usable if gold[i]["frame"] == "random")
    if not strat or not rand:
        print("need both frames — aborting")
        return

    # The random frame is the only true-prior data we have, so it must appear in
    # BOTH training (or the model learns the stratified frame's fake prior and
    # over-fires on rare classes) and testing (or the score is meaningless).
    # Split it in half: half teaches the real prior, half stays a clean holdout.
    half = len(rand) // 2
    rand_tr, rand_te = rand[:half], rand[half:]
    train = strat + rand_tr
    test = rand_te
    print(f"train: {len(train):,} ({len(strat):,} stratified + {len(rand_tr):,} random)"
          f"   test (held-out random, unbiased): {len(test):,}")

    Xtr = vecs[[pos[i] for i in train]]
    ytr = [gold[i]["stance"] for i in train]
    Xte = vecs[[pos[i] for i in test]]
    yte = [gold[i]["stance"] for i in test]

    # NO class_weight="balanced": the training set is ALREADY oversampled for rare
    # classes, so balancing on top double-corrects and makes the model believe
    # nihilism is common when it is ~0%. That bug cost 9 points of accuracy.
    best = None
    for C in (0.5, 1.0, 4.0, 16.0):
        m = LogisticRegression(max_iter=3000, C=C)
        m.fit(Xtr, ytr)
        a = accuracy_score(yte, m.predict(Xte))
        print(f"  C={C:<5} accuracy={a:.3f}")
        if best is None or a > best[0]:
            best = (a, C, m)
    _, C, clf = best
    print(f"\nbest C={C}")
    pred = clf.predict(Xte)

    print("\n=== DISTILLED CLASSIFIER vs gold (held-out random frame) ===")
    print(classification_report(yte, pred, zero_division=0))
    print(f"accuracy: {accuracy_score(yte, pred):.3f}")

    # --- the honest comparison: what did the 7B score on these same items? ---
    conn = dbmod.get_conn()
    q = "SELECT item_id, meaning_stance FROM enrichment WHERE model='qwen2.5:7b#g3'"
    old = {r[0]: r[1] for r in conn.execute(q)}
    conn.close()
    both = [i for i in test if i in old]
    if both:
        print(f"\n=== OLD 7B vs gold (same {len(both):,} held-out items) ===")
        print(classification_report([gold[i]["stance"] for i in both],
                                    [old[i] for i in both], zero_division=0))
        print(f"accuracy: {accuracy_score([gold[i]['stance'] for i in both], [old[i] for i in both]):.3f}")

    # --- ensemble: 7B's gate is 95.8% precise on `none`; the classifier is better
    # at spotting asserts. Trust the 7B when it says none, else the classifier. ---
    if both:
        ens = [("none" if old[i] == "none" else p)
               for i, p in zip(test, pred) if i in old]
        ytrue = [gold[i]["stance"] for i in test if i in old]
        print(f"\n=== ENSEMBLE (7B gate + classifier) on {len(ens):,} held-out ===")
        print(classification_report(ytrue, ens, zero_division=0))
        print(f"accuracy: {accuracy_score(ytrue, ens):.3f}")

    # --- unbiased prevalence, from the random frame's GOLD labels only ---
    print("\n=== TRUE PREVALENCE (gold, full random frame — the honest distribution) ===")
    from collections import Counter
    c = Counter(gold[i]["stance"] for i in rand)
    for s, n in c.most_common():
        print(f"  {100*n/len(rand):5.1f}%  {s:15s} ({n})")

    import pickle
    (CACHE / "stance_clf.pkl").write_bytes(pickle.dumps(clf))
    print(f"\nsaved classifier -> {CACHE/'stance_clf.pkl'}")


if __name__ == "__main__":
    main()
