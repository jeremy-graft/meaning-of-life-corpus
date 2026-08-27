"""Seam-composition confound check, SAMPLED so it finishes in minutes and reports
per language as it goes. Question: is the cross-cultural religion ranking really
about culture, or about which seams YouTube happened to serve each language?"""
import sys, json, re, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict, Counter
from pathlib import Path
import torch
from db import db as dbmod
OUT=Path("_emb/stance_ft_v2"); MAXLEN=128; BATCH=96; PER_LANG=6000; SEED=11
REL={
"ar":r"الله|ربنا|الجنة|الصلاة|الدعاء|القرآن|الاخرة|الآخرة|إيمان|النبي|رب|روح",
"id":r"allah|tuhan|doa|surga|iman|ibadah|akhirat|sholat|shalat|agama|rohani|berkah",
"es":r"\bdios|jesús|cristo|\bfe\b|oración|orar|cielo|\balma\b|iglesia|biblia|divin|etern|sagrad|señor",
"ko":r"하나님|하느님|하늘|예수|기도|천국|신앙|믿음|교회|성경|부처|불교|극락|영혼|주님",
"pt":r"\bdeus|jesus|cristo|\bfé\b|oração|orar|\bcéu\b|\balma\b|igreja|bíblia|divin|etern|sagrad|senhor",
"hi":r"भगवान|ईश्वर|प्रभु|प्रार्थना|स्वर्ग|आत्मा|धर्म|पूजा|मोक्ष|गुरु|अल्लाह",
"tr":r"allah|tanrı|dua|cennet|\biman|ibadet|ahiret|namaz|\bdin\b|ruh|peygamber|rabbim",
"ja":r"神様|神さま|\b神\b|神を|仏|仏様|祈り|お祈り|天国|信仰|お寺|極楽|魂|ご先祖",
}
KEY=["last_words_hospice","near_death_experience","terminal_diagnosis","widow_grief",
     "faith_crisis","new_parent_first_child","sobriety_milestone","immigrant_why_i_left"]
def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    labels=json.loads((OUT/"labels.json").read_text()); ai=labels.index("asserts")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(OUT)
    m=AutoModelForSequenceClassification.from_pretrained(OUT).to(dev).eval()
    conn=dbmod.get_conn(); rng=random.Random(SEED)
    mix=defaultdict(Counter); cell=defaultdict(lambda:[0,0])
    print("sampling up to %d comments per language\n"%PER_LANG,flush=True)
    for lg in REL:
        rows=conn.execute("SELECT genre_seed, substr(text,1,500) FROM items WHERE lang=? "
            "AND kind='comment' AND length(text)>40",(lg,)).fetchall()
        if not rows: print(f"  {lg}: none",flush=True); continue
        if len(rows)>PER_LANG: rows=rng.sample(rows,PER_LANG)
        rx=re.compile(REL[lg])
        for k in range(0,len(rows),BATCH):
            ch=rows[k:k+BATCH]
            enc=tok([r[1] for r in ch],truncation=True,padding=True,
                    max_length=MAXLEN,return_tensors="pt").to(dev)
            with torch.no_grad(): pred=m(**enc).logits.argmax(-1).cpu().tolist()
            for (sm,t),p in zip(ch,pred):
                mix[lg][sm]+=1
                if p==ai:
                    cell[(lg,sm)][0]+=1
                    if rx.search(t): cell[(lg,sm)][1]+=1
        A=sum(cell[(lg,s)][0] for s in mix[lg]); R=sum(cell[(lg,s)][1] for s in mix[lg])
        print(f"  {lg}: n={len(rows):,}  asserts={A:,}  religion={100*R/max(1,A):.0f}%",flush=True)
    json.dump({"mix":{l:dict(mix[l]) for l in mix},
               "cell":{f"{l}|{s}":v for (l,s),v in cell.items()}},
              open("_seamcheck.json","w"))
    print("\n=== 1. SEAM MIX  (% of each language's sampled comments) ===",flush=True)
    print(f"{'lang':5s}"+"".join(f"{s[:10]:>12s}" for s in KEY),flush=True)
    for lg in mix:
        tot=sum(mix[lg].values())
        print(f"{lg:5s}"+"".join(f"{100*mix[lg].get(s,0)/tot:11.0f}%" for s in KEY),flush=True)
    print("\n=== 2. RELIGION SHARE WITHIN SEAM (like-for-like; · = under 25 asserts) ===",flush=True)
    print(f"{'lang':5s}"+"".join(f"{s[:10]:>12s}" for s in KEY)+f"{'ALL':>8s}",flush=True)
    for lg in mix:
        out=[]
        for s in KEY:
            a,r=cell.get((lg,s),[0,0])
            out.append(f"{100*r/a:10.0f}%" if a>=25 else f"{'·':>11s}")
        A=sum(cell[(lg,s)][0] for s in mix[lg]); R=sum(cell[(lg,s)][1] for s in mix[lg])
        print(f"{lg:5s}"+"".join(out)+f"{100*R/max(1,A):7.0f}%",flush=True)
if __name__=="__main__": main()
