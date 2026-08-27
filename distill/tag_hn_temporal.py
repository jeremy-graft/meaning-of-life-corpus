"""Semantic temporal pass: run the fixed XLM-R stance tagger over HN by year.
Upgrades the lexical drift (keyword 'meaningless') to real stance (denies/asserts).
CAVEAT: tagger trained on YouTube-register; dumps samples to spot-check HN transfer."""
import sys, json
from collections import defaultdict
from pathlib import Path
import torch
from db import db as dbmod
OUT=Path("_emb/stance_ft_v2"); MAXLEN=128; BATCH=64
def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    labels=json.loads((OUT/"labels.json").read_text()); l2i={l:i for i,l in enumerate(labels)}
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(OUT); model=AutoModelForSequenceClassification.from_pretrained(OUT).to(dev).eval()
    c=dbmod.get_conn()
    rows=c.execute("SELECT substr(published_at,1,4) y, substr(text,1,600) t FROM items WHERE source_id='hackernews' AND published_at IS NOT NULL").fetchall()
    print(f"tagging {len(rows):,} HN items on {dev}")
    tot=defaultdict(int); st=defaultdict(lambda: defaultdict(int)); samples=[]
    for k in range(0,len(rows),BATCH):
        chunk=rows[k:k+BATCH]; texts=[r[1] for r in chunk]
        enc=tok(texts,truncation=True,padding=True,max_length=MAXLEN,return_tensors="pt").to(dev)
        with torch.no_grad(): pred=model(**enc).logits.argmax(-1).cpu().tolist()
        for (y,t),p in zip(chunk,pred):
            if not(y and y.isdigit()): continue
            y=int(y); tot[y]+=1; st[y][labels[p]]+=1
            if labels[p] in ("asserts","denies") and len(samples)<40: samples.append((labels[p],t[:180]))
        if k%25600==0: print(f"  {k:,}/{len(rows):,}")
    json.dump({"tot":dict(tot),"st":{y:dict(st[y]) for y in st},"samples":samples},open("_hn_temporal.json","w"))
    print("done")
if __name__=="__main__": main()
