"""Sample the CLEAN hopeless corpus for coping-strategy gold labelling.

Random within language (prevalence-safe, so base rates are estimable), stratified
across the languages with enough volume. Excludes Hacker News and Stack Exchange,
which contaminated the first count by 29 percent.
"""
import sys, json, random, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from db import db as dbmod
OUT=Path("_hope"); CHUNK=400; SEED=31; PER_LANG=300
LANGS=["en","es","pt","ar","id","ja","tr","hi"]
SEAMS=["terminal_diagnosis","last_words_hospice","widow_grief","child_loss","suicide_survivor",
 "caregiver_dying_parent","cancer_survivor","eulogy_grief","miscarriage","prison_release"]
def main():
    rng=random.Random(SEED); conn=dbmod.get_conn(); picked=[]
    ph=",".join("?"*len(SEAMS))
    for lg in LANGS:
        cond="(lang IS NULL OR lang='en')" if lg=="en" else "lang=?"
        args=SEAMS+([] if lg=="en" else [lg])
        rows=conn.execute(
            f"SELECT id, genre_seed, text FROM items WHERE genre_seed IN ({ph}) "
            f"AND source_id!='hackernews' AND source_id NOT LIKE 'se_%' AND {cond} "
            f"AND length(text) BETWEEN 70 AND 600 ORDER BY id",args).fetchall()
        if not rows: print(f"  {lg}: none"); continue
        take=rng.sample(rows,min(PER_LANG,len(rows)))
        for r in take:
            picked.append({"id":r[0],"lang":lg,"seam":r[1],
                           "text":re.sub(r"\s+"," ",r[2]).strip()[:500]})
        print(f"  {lg}: pool {len(rows):,} -> {len(take)}")
    rng.shuffle(picked)
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.json*"): f.unlink()
    n=0
    for k in range(0,len(picked),CHUNK):
        lines=[json.dumps({"id":p["id"],"text":p["text"]},ensure_ascii=False)
               for p in picked[k:k+CHUNK]]
        (OUT/f"chunk_{k//CHUNK:02d}.jsonl").write_text("\n".join(lines),encoding="utf-8")
        n+=1
    (OUT/"_meta.json").write_text(json.dumps(
        {p["id"]:{"lang":p["lang"],"seam":p["seam"]} for p in picked},ensure_ascii=False),
        encoding="utf-8")
    print(f"\nhope gold: {len(picked):,} items -> {n} chunks of {CHUNK}")
if __name__=="__main__": main()
