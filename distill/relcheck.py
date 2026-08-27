"""Validate the religion patterns AGAINST GOLD, on the very same fragments.
If keyword-detection agrees with the careful reader on items we have both for,
the pattern is calibrated and can be trusted on languages we have no gold for."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict
from pathlib import Path
from db import db as dbmod
from distill.relwords import pattern

G={}
for d in ["_gold","_goldml","_gold3","_goldact"]:
    p=Path(d)
    if not p.exists(): continue
    lm=json.loads((p/"_langs.json").read_text(encoding="utf-8")) if (p/"_langs.json").exists() else {}
    for f in sorted(p.glob("out_*.json")):
        try:
            for r in json.loads(f.read_text(encoding="utf-8")):
                if r.get("stance")=="asserts" and r.get("sources"):
                    G[r["id"]]={"rel":"religion_transcendence" in r["sources"],
                                "lang":lm.get(r["id"],"en")}
        except json.JSONDecodeError: pass
ids=list(G); conn=dbmod.get_conn(); txt={}
for k in range(0,len(ids),400):
    ch=ids[k:k+400]
    for i,t in conn.execute(f"SELECT id,text FROM items WHERE id IN ({','.join('?'*len(ch))})",ch):
        txt[i]=t or ""
per=defaultdict(lambda:{"n":0,"gold":0,"kw":0,"tp":0,"fp":0,"fn":0})
for i,g in G.items():
    if i not in txt: continue
    rx=pattern(g["lang"]); hit=bool(rx.search(txt[i])); d=per[g["lang"]]
    d["n"]+=1; d["gold"]+=g["rel"]; d["kw"]+=hit
    if hit and g["rel"]: d["tp"]+=1
    elif hit and not g["rel"]: d["fp"]+=1
    elif not hit and g["rel"]: d["fn"]+=1
print("KEYWORD vs CAREFUL READER, on the same fragments (religion share of asserts)\n")
print(f"{'lang':5s}{'asserts':>8s}{'gold':>7s}{'keyword':>9s}{'error':>8s}{'precis':>8s}{'recall':>8s}")
tot=defaultdict(int)
for lg in sorted(per,key=lambda l:-per[l]["n"]):
    d=per[lg]
    if d["n"]<40: continue
    gs=100*d["gold"]/d["n"]; ks=100*d["kw"]/d["n"]
    pr=d["tp"]/max(1,d["tp"]+d["fp"]); rc=d["tp"]/max(1,d["tp"]+d["fn"])
    print(f"{lg:5s}{d['n']:8d}{gs:6.0f}%{ks:8.0f}%{ks-gs:+7.0f}{100*pr:7.0f}%{100*rc:7.0f}%")
    for k in ("n","gold","kw","tp","fp","fn"): tot[k]+=d[k]
gs=100*tot["gold"]/tot["n"]; ks=100*tot["kw"]/tot["n"]
pr=tot["tp"]/max(1,tot["tp"]+tot["fp"]); rc=tot["tp"]/max(1,tot["tp"]+tot["fn"])
print(f"\n{'ALL':5s}{tot['n']:8d}{gs:6.0f}%{ks:8.0f}%{ks-gs:+7.0f}{100*pr:7.0f}%{100*rc:7.0f}%")
print(f"\nF1 = {2*pr*rc/max(1e-9,pr+rc):.3f}")
