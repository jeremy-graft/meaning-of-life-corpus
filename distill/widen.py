"""Free first-cut extension of the cross-cultural map to the new languages:
tagger finds asserts, language-specific religion keywords detect faith within them.
ROUGH & zero-shot for never-trained langs — directional only."""
import sys, json, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import torch
from db import db as dbmod
OUT=Path("_emb/stance_ft_v2"); MAXLEN=128; BATCH=64
REL={
"ko":r"하나님|하느님|하늘|예수|기도|천국|신앙|믿음|교회|성경|부처|불교|극락|영혼|주님",
"ja":r"神様|神さま|\b神\b|神を|仏|仏様|祈り|お祈り|天国|信仰|お寺|極楽|魂|ご先祖",
"ru":r"\bбог|госпо|молитв|\bвера\b|верую|душа|небеса|церк|\bрай\b|христ|аллах|\bдух|свят",
"de":r"\bgott|jesus|christ|glaub|gebet|beten|himmel|seele|kirche|bibel|göttlich|\bewig|heilig",
"fr":r"\bdieu|jésus|christ|\bfoi\b|prière|prier|\bciel\b|\bâme\b|église|bible|divin|éternel|sacré",
"vi":r"chúa|thượng đế|đức phật|\bphật|cầu nguyện|thiên đường|linh hồn|niết bàn|đức tin|tâm linh|thần",
"th":r"พระเจ้า|พระพุทธ|พระธรรม|สวรรค์|วิญญาณ|ศรัทธา|บุญ|นิพพาน|ภาวนา|เทพ|ทำบุญ",
"fa":r"خدا|الله|نماز|\bدعا|بهشت|\bروح\b|ایمان|قرآن|پیامبر|\bدین\b|معنوی|خداوند",
"pl":r"\bbóg|boga|jezus|chrystus|wiar|modli|niebo|dusza|kości|bibli|święt|wieczn|duchow",
"tl":r"diyos|panginoon|jesus|kristo|pananampalataya|panalangin|langit|kaluluwa|simbahan|banal|espirit",
}
def run(lang,rx,ai,tok,m,dev):
    c=dbmod.get_conn()
    rows=c.execute("SELECT substr(text,1,600) t FROM items WHERE lang=? AND kind='comment' AND length(text)>40",(lang,)).fetchall()
    if not rows: print(f"{lang}: no data"); return
    asr=0; rel=0; n=0
    for k in range(0,len(rows),BATCH):
        ch=[r[0] for r in rows[k:k+BATCH]]
        enc=tok(ch,truncation=True,padding=True,max_length=MAXLEN,return_tensors="pt").to(dev)
        with torch.no_grad(): pred=m(**enc).logits.argmax(-1).cpu().tolist()
        for t,p in zip(ch,pred):
            n+=1
            if p==ai:
                asr+=1
                if rx.search(t): rel+=1
    print(f"  {lang}: {n:>6,} comments | asserts {100*asr/n:4.1f}% | RELIGION share of asserts {100*rel/max(1,asr):4.0f}%  (n_asr={asr})")
def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    labels=json.loads((OUT/"labels.json").read_text()); ai=labels.index("asserts")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(OUT); m=AutoModelForSequenceClassification.from_pretrained(OUT).to(dev).eval()
    print("RELIGION SHARE OF ASSERTS — free first-cut extension (rough, zero-shot for new langs)\n")
    for lg,pat in REL.items(): run(lg,re.compile(pat),ai,tok,m,dev)
if __name__=="__main__": main()
