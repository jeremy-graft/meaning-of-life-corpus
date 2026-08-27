"""Fine-tune a small multilingual encoder on the Claude gold labels.

WHY THIS AND NOT distill/classify.py: frozen sentence-embeddings + logistic
regression FAILED (asserts F1 0.26 vs the 7B's 0.64). Root cause: those
embeddings encode what a text is ABOUT, and stance is a SPEECH ACT — a hospice
comment asserting faith and one merely grieving are topically near-identical, so
they land in the same neighbourhood and no linear boundary can split them.

Fine-tuning updates the encoder itself, so it can learn "is this asserting?" as
a feature rather than inheriting a topical one. Same gold, same split, same
metric — the only thing that changes is that the features are learned.

THE METRIC THAT MATTERS is asserts-F1, NOT accuracy: `none` is 85.8% of the data,
so "always predict none" scores 85.8% while being completely useless. Target to
beat: the 7B's asserts F1 = 0.64.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from db import db as dbmod
from distill.classify import load_gold

BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT = Path(__file__).resolve().parent.parent / "_emb" / "stance_ft"
MAX_LEN = 128
EPOCHS = 5
LR = 3e-5
BATCH = 16


class DS(Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.labels[i])
        return d


def get_texts(ids: list[str]) -> dict[str, str]:
    conn = dbmod.get_conn()
    out: dict[str, str] = {}
    for k in range(0, len(ids), 500):
        chunk = ids[k:k + 500]
        q = f"SELECT id, text FROM items WHERE id IN ({','.join('?' * len(chunk))})"
        for i, t in conn.execute(q, chunk):
            out[i] = (t or "")[:500]
    conn.close()
    return out


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import classification_report, f1_score

    gold = load_gold()
    strat = sorted(i for i in gold if gold[i]["frame"] == "stratified")
    rand = sorted(i for i in gold if gold[i]["frame"] == "random")
    half = len(rand) // 2
    # identical split to classify.py so the comparison is apples-to-apples
    train_ids = strat + rand[:half]
    test_ids = rand[half:]

    texts = get_texts(train_ids + test_ids)
    train_ids = [i for i in train_ids if texts.get(i)]
    test_ids = [i for i in test_ids if texts.get(i)]

    labels = sorted({gold[i]["stance"] for i in train_ids} | {gold[i]["stance"] for i in test_ids})
    l2i = {l: k for k, l in enumerate(labels)}
    print(f"train {len(train_ids):,}  test {len(test_ids):,}  classes {labels}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(labels)).to(dev)

    def enc(ids):
        return tok([texts[i] for i in ids], truncation=True, padding="max_length",
                   max_length=MAX_LEN, return_tensors="pt")

    tr = DS(enc(train_ids), [l2i[gold[i]["stance"]] for i in train_ids])
    te_enc, te_y = enc(test_ids), [l2i[gold[i]["stance"]] for i in test_ids]

    dl = DataLoader(tr, batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            out = model(**b)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += out.loss.item()
        # eval each epoch — small data overfits fast, so watch asserts-F1 not loss
        model.eval()
        with torch.no_grad():
            logits = []
            for k in range(0, len(te_y), 64):
                bb = {kk: vv[k:k + 64].to(dev) for kk, vv in te_enc.items()}
                logits.append(model(**bb).logits.cpu())
            pred = torch.cat(logits).argmax(-1).numpy()
        af = f1_score(te_y, pred, labels=[l2i["asserts"]], average="macro", zero_division=0)
        print(f"  epoch {ep+1}: loss={tot/len(dl):.4f}  asserts_F1={af:.3f}")

    print("\n=== FINE-TUNED vs gold (held-out random frame) ===")
    # pass labels= explicitly: rare classes (denies ~0%) may be absent from the
    # test split entirely, and target_names alone then mismatches the class count
    print(classification_report(te_y, pred, labels=list(range(len(labels))),
                                target_names=labels, zero_division=0))
    print(f"\nasserts F1 = {af:.3f}    (7B baseline = 0.64, frozen-embed+LR = 0.26)")
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    (OUT / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
