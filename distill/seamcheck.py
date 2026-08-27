"""Is the cross-cultural map confounded by SEAM COMPOSITION?
(1) each language's seam mix; (2) religion share WITHIN the same seam (like-for-like)."""
import sys, json, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict, Counter
from pathlib import Path
import torch
from db import db as dbmod
OUT=Path("_emb/stance_ft_v2"); MAXLEN=128; BATCH=96
REL={
"ar":r"الله|ربنا|الجنة|الصلاة|الدعاء|القرآن|الاخرة|الآخرة|الرزق|إيمان|روح|النبي|رب",
"fa":r"خدا|الله|نماز|\bدعا|بهشت|\bروح\b|ایمان|قرآن|پیامبر|\bدین\b|معنوی|خداوند",
"id":r"allah|tuhan|doa|surga|iman|ibadah|akhirat|sholat|shalat|agama|rohani|berkah",
"es":r"\bdios|jesús|cristo|\bfe\b|oración|orar|cielo|\balma\b|iglesia|biblia|divin|etern|sagrad|señor",
"ko":r"하나님|하느님|하늘|예수|기도|천국|신앙|믿음|교회|성경|부처|불교|극락|영혼|주님",
"pt":r"\bdeus|jesus|cristo|\bfé\b|oração|orar|\bcéu\b|\balma\b|igreja|bíblia|divin|etern|sagrad|senhor",
"hi":r"भगवान|ईश्वर|प्रभु|प्रार्थना|स्वर्ग|आत्मा|धर्म|पूजा|मोक्ष|ईश|गुरु|अल्लाह",
"tr":r"allah|tanrı|dua|cennet|\biman|ibadet|ahiret|namaz|\bdin\b|ruh|peygamber|rabbim",
"fr":r"\bdieu|jésus|christ|\bfoi\b|prière|prier|\bciel\b|\bâme\b|église|bible|divin|éternel|sacré|seigneur",
"ru":r"\bбог|госпо|молитв|\bвера\b|верую|душа|небеса|церк|\bрай\b|христ|аллах|свят",
"de":r"\bgott|jesus|christ|glaub|gebet|beten|himmel|seele|kirche|bibel|göttlich|\bewig|heilig|herr",
"pl":r"\bbóg|boga|bogu|jezus|chrystus|wiar|modli|niebo|dusza|kościół|bibli|święt|wieczn|duchow|pan bóg",
"tl":r"diyos|panginoon|jesus|kristo|pananampalataya|panalangin|langit|kaluluwa|simbahan|banal|espirit|dasal",
"th":r"พระเจ้า|พระพุทธ|พระธรรม|สวรรค์|วิญญาณ|ศรัทธา|บุญ|นิพพาน|ภาวนา|เทพ|ทำบุญ|พระ",
"vi":r"chúa|thượng đế|đức phật|\bphật|cầu nguyện|thiên đường|linh hồn|niết bàn|đức tin|tâm linh|thần|cầu",
"ja":r"神様|神さま|\b神\b|神を|仏|仏様|祈り|お祈り|天国|信仰|お寺|極楽|魂|ご先祖",
}
def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    labels=json.loads((OUT/"labels.json").read_text()); ai=labels.index("asserts")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(OUT); m=AutoModelForSequenceClassification.from_pretrained(OUT).to(dev).eval()
    conn=dbmod.get_conn()
    langs=list(REL)
    mix=defaultdict(Counter)          # lang -> seam -> n comments
    cell=defaultdict(lambda:[0,0])    # (lang,seam) -> [asserts, religious asserts]
    for lg in langs:
        rows=conn.execute("SELECT genre_seed, substr(text,1,600) FROM items WHERE lang=? AND kind='comment' AND length(text)>40",(lg,)).fetchall()
        if not rows: continue
        rx=re.compile(REL[lg])
        for k in range(0,len(rows),BATCH):
            ch=rows[k:k+BATCH]
            enc=tok([r[1] for r in ch],truncation=True,padding=True,max_length=MAXLEN,return_tensors="pt").to(dev)
            with torch.no_grad(): pred=m(**enc).logits.argmax(-1).cpu().tolist()
            for (sm,t),p in zip(ch,pred):
                mix[lg][sm]+=1
                if p==ai:
                    cell[(lg,sm)][0]+=1
                    if rx.search(t): cell[(lg,sm)][1]+=1
    json.dump({"mix":{l:dict(mix[l]) for l in mix},
               "cell":{f"{l}|{s}":v for (l,s),v in cell.items()}},open("_seamcheck.json","w"))
    # ---- report 1: seam mix skew
    print("=== 1. SEAM MIX per language (% of that language's comments) ===")
    seams=sorted({s for l in mix for s in mix[l]})
    key=["last_words_hospice","near_death_experience","terminal_diagnosis","widow_grief",
         "faith_crisis","new_parent_first_child","sobriety_milestone","immigrant_why_i_left"]
    print(f"{'lang':5s}"+"".join(f"{s[:11]:>13s}" for s in key))
    for lg in langs:
        if not mix[lg]: continue
        tot=sum(mix[lg].values())
        print(f"{lg:5s}"+"".join(f"{100*mix[lg].get(s,0)/tot:12.0f}%" for s in key))
    # ---- report 2: religion share WITHIN each seam
    print("\n=== 2. RELIGION SHARE OF ASSERTS, WITHIN SEAM (like-for-like) ===")
    print(f"{'lang':5s}"+"".join(f"{s[:11]:>13s}" for s in key)+f"{'ALL':>8s}")
    for lg in langs:
        if not mix[lg]: continue
        cells=[]
        for s in key:
            a,r=cell.get((lg,s),[0,0])
            cells.append(f"{100*r/a:11.0f}%" if a>=25 else f"{'·':>12s}")
        A=sum(cell[(lg,s)][0] for s in mix[lg] if (lg,s) in cell)
        R=sum(cell[(lg,s)][1] for s in mix[lg] if (lg,s) in cell)
        print(f"{lg:5s}"+"".join(cells)+f"{100*R/max(1,A):7.0f}%")
if __name__=="__main__": main()
