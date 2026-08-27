"""Give the local 7B a FAIR test: one binary question instead of a 7-field JSON object.

The 7B was judged on structured extraction (7 fields, enums, arrays) and collapsed.
Small models are far better at single yes/no decisions. If asking one plain question
lifts assert-detection to ~0.8, all 1.28M items can be tagged locally and free.
Scored against the same gold, on the unbiased random frame only.
"""
import sys, json, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from db import db as dbmod

MODEL="qwen2.5:7b"; N=500; WORKERS=4
Q=("Here is a comment someone left on a video.\n\n"
   "---\n{t}\n---\n\n"
   "Question: is this person saying what makes life meaningful or worth living?\n\n"
   "Answer YES only if they state something that gives life meaning or purpose, such as "
   "family, God, freedom, their work, helping others, or becoming themselves.\n"
   "Answer NO for grief, condolences, jokes, arguments, greetings, praise of the video, "
   "or a prayer or blessing used as a polite phrase.\n\n"
   "Reply with exactly one word: YES or NO.")

def load_gold():
    recs={}
    for d in ["_gold","_goldml","_gold3"]:
        p=Path(d)
        if not p.exists(): continue
        fr=json.loads((p/"_frames.json").read_text(encoding="utf-8")) if (p/"_frames.json").exists() else {}
        lm=json.loads((p/"_langs.json").read_text(encoding="utf-8")) if (p/"_langs.json").exists() else {}
        for f in sorted(p.glob("out_*.json")):
            try:
                for r in json.loads(f.read_text(encoding="utf-8")):
                    i=r.get("id")
                    if i and fr.get(i,"random")=="random":
                        recs[i]={"y":r.get("stance")=="asserts","lang":lm.get(i,"en")}
            except json.JSONDecodeError: pass
    return recs

def main():
    import ollama, random
    G=load_gold(); ids=sorted(G); random.Random(5).shuffle(ids); ids=ids[:N]
    conn=dbmod.get_conn(); txt={}
    for k in range(0,len(ids),400):
        ch=ids[k:k+400]
        for i,t in conn.execute(f"SELECT id,substr(text,1,700) FROM items WHERE id IN ({','.join('?'*len(ch))})",ch):
            txt[i]=t
    ids=[i for i in ids if txt.get(i)]
    print(f"testing {len(ids)} gold items (random frame) against {MODEL}\n",flush=True)
    def ask(i):
        try:
            r=ollama.chat(model=MODEL,messages=[{"role":"user","content":Q.format(t=txt[i])}],
                options={"temperature":0,"num_ctx":2048,"num_predict":4})
            a=r["message"]["content"].strip().upper()
            return i,("YES" in a[:6])
        except Exception:
            return i,None
    tp=fp=fn=tn=0; bad=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for n,(i,p) in enumerate(ex.map(ask,ids),1):
            if p is None: bad+=1; continue
            y=G[i]["y"]
            if p and y: tp+=1
            elif p and not y: fp+=1
            elif not p and y: fn+=1
            else: tn+=1
            if n%100==0: print(f"  {n}/{len(ids)}",flush=True)
    pr=tp/max(1,tp+fp); rc=tp/max(1,tp+fn); f1=2*pr*rc/max(1e-9,pr+rc)
    print(f"\nBINARY GATE, local {MODEL}")
    print(f"  asserts precision {100*pr:.0f}%   recall {100*rc:.0f}%   F1 {f1:.3f}")
    print(f"  (tp {tp}  fp {fp}  fn {fn}  tn {tn}  failed {bad})")
    print(f"\n  compare: XLM-R fine-tune F1 0.537 | 7B on 7-field JSON F1 0.64 (English only)")
    print(f"  target for tagging everything locally: about 0.80")
if __name__=="__main__": main()
