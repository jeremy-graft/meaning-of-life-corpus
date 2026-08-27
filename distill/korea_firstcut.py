"""Free first-cut Korea-vs-Japan: tagger finds asserts, lexical religion detects
religious language within them. Rough/directional (not gold), same method for both."""
import sys, json, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import torch
from db import db as dbmod
OUT=Path("_emb/stance_ft_v2"); MAXLEN=128; BATCH=64

# religion keywords per language (God / pray / heaven / faith / Buddha / temple ...)
REL = {
 "ko": re.compile(r"하나님|하느님|하늘|예수|기도|천국|신앙|믿음|교회|성경|부처|불교|극락|영혼|천당|신께|주님"),
 "ja": re.compile(r"神様|神さま|\b神\b|神を|仏|仏様|お祈り|祈り|天国|信仰|お寺|極楽|魂|あの世|神仏|ご先祖"),
}
def run(lang):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    labels=json.loads((OUT/"labels.json").read_text())
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(OUT); m=AutoModelForSequenceClassification.from_pretrained(OUT).to(dev).eval()
    c=dbmod.get_conn()
    rows=c.execute("SELECT substr(text,1,600) t FROM items WHERE lang=? AND kind='comment' AND length(text)>40",(lang,)).fetchall()
    ai=labels.index("asserts"); rx=REL[lang]
    asserts=0; rel_in_asserts=0; n=0
    for k in range(0,len(rows),BATCH):
        ch=[r[0] for r in rows[k:k+BATCH]]
        enc=tok(ch,truncation=True,padding=True,max_length=MAXLEN,return_tensors="pt").to(dev)
        with torch.no_grad(): pred=m(**enc).logits.argmax(-1).cpu().tolist()
        for t,p in zip(ch,pred):
            n+=1
            if p==ai:
                asserts+=1
                if rx.search(t): rel_in_asserts+=1
    print(f"{lang}: {n:,} comments | tagger-asserts {asserts:,} ({100*asserts/n:.1f}%) | "
          f"religious language within asserts: {rel_in_asserts:,} ({100*rel_in_asserts/max(1,asserts):.0f}%)")
for lg in ("ko","ja"): run(lg)
