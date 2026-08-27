"""Emergent taxonomy: cluster gold ASSERTS by meaning (multilingual embeddings),
independent of our 15 hand-drawn sources. Do the data's own groupings match ours,
or reveal categories we didn't predefine?"""
import sys, json, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter
from pathlib import Path
import numpy as np
from db import db as dbmod
ROOT=Path(".")
def load_asserts():
    recs={}
    for d in ["_gold","_goldml","_gold3","_goldact"]:
        p=ROOT/d
        if not p.exists(): continue
        lm=json.loads((p/"_langs.json").read_text(encoding="utf-8")) if (p/"_langs.json").exists() else {}
        for f in sorted(p.glob("out_*.json")):
            try:
                for r in json.loads(f.read_text(encoding="utf-8")):
                    if r.get("stance")=="asserts" and r.get("id"):
                        recs[r["id"]]={"src":r.get("sources") or [],"lang":lm.get(r["id"],"en")}
            except json.JSONDecodeError: pass
    return recs
def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    A=load_asserts(); ids=list(A)
    conn=dbmod.get_conn(); txt={}
    for k in range(0,len(ids),400):
        ch=ids[k:k+400]
        for i,t in conn.execute(f"SELECT id,substr(text,1,400) FROM items WHERE id IN ({','.join('?'*len(ch))})",ch):
            txt[i]=t
    ids=[i for i in ids if txt.get(i)]
    print(f"asserts with text: {len(ids):,}")
    m=SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    X=m.encode([txt[i] for i in ids],batch_size=256,show_progress_bar=False,normalize_embeddings=True)
    K=14
    km=KMeans(n_clusters=K,n_init=8,random_state=0).fit(X)
    lab=km.labels_
    print(f"\n=== {K} EMERGENT CLUSTERS (data's own grouping of {len(ids)} asserts) ===\n")
    for cl in range(K):
        idx=[j for j in range(len(ids)) if lab[j]==cl]
        srcs=Counter(); langs=Counter()
        for j in idx:
            srcs.update(A[ids[j]]["src"]); langs[A[ids[j]]["lang"]]+=1
        dom=", ".join(f"{s}:{n}" for s,n in srcs.most_common(3))
        lg=", ".join(f"{l}:{n}" for l,n in langs.most_common(4))
        # centroid-nearest examples
        c=km.cluster_centers_[cl]; d=[(float(np.dot(X[j],c)),j) for j in idx]; d.sort(reverse=True)
        exs=[re.sub(r"\s+"," ",txt[ids[j]]).strip()[:90] for _,j in d[:2]]
        print(f"CLUSTER {cl}  (n={len(idx)})  hand-labels: {dom or '-'}")
        print(f"   langs: {lg}")
        for e in exs: print(f"   • {e}")
        print()
if __name__=="__main__": main()
