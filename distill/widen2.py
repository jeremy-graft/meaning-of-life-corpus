"""Re-run the map extension with alphabet-aware religion patterns.
Reports the raw keyword figure AND the figure corrected by the +7pp overcount
measured against gold, so the number is honest about its own bias."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import torch
from db import db as dbmod
from distill.relwords import pattern, LANGS
OUT=Path("_emb/stance_ft_v2"); MAXLEN=128; BATCH=96; BIAS=7.0
GOLD={"ar":70,"id":53,"es":49,"ko":47,"pt":45,"en":44,"hi":41,"tr":34,"ja":11}
def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    labels=json.loads((OUT/"labels.json").read_text()); ai=labels.index("asserts")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(OUT)
    m=AutoModelForSequenceClassification.from_pretrained(OUT).to(dev).eval()
    conn=dbmod.get_conn()
    print(f"{'lang':5s}{'comments':>10s}{'asserts':>9s}{'raw':>7s}{'corrected':>11s}{'gold':>7s}{'gap':>6s}")
    for lg in LANGS:
        rows=conn.execute("SELECT substr(text,1,600) FROM items WHERE lang=? AND kind='comment' "
                          "AND length(text)>40",(lg,)).fetchall()
        if len(rows)<800: continue
        rx=pattern(lg); a=0; r=0; n=0
        for k in range(0,len(rows),BATCH):
            ch=[x[0] for x in rows[k:k+BATCH]]
            enc=tok(ch,truncation=True,padding=True,max_length=MAXLEN,return_tensors="pt").to(dev)
            with torch.no_grad(): pred=m(**enc).logits.argmax(-1).cpu().tolist()
            for t,p in zip(ch,pred):
                n+=1
                if p==ai:
                    a+=1
                    if rx.search(t): r+=1
        raw=100*r/max(1,a); corr=max(0,raw-BIAS)
        g=GOLD.get(lg); gs=f"{g}%" if g else "  -"
        gap=f"{corr-g:+.0f}" if g else "   -"
        print(f"{lg:5s}{n:10,}{a:9,}{raw:6.0f}%{corr:10.0f}%{gs:>7s}{gap:>6s}",flush=True)
if __name__=="__main__": main()
