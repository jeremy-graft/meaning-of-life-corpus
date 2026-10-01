"""What strangers do with someone else's reason for living.

Reads the coded replies and answers three things:

  1. What is the distribution of moves, and does it differ between the two
     PROMPT GROUPS? Chunks 00-05 were coded under an instruction whose first
     distinction was redirect-versus-affiliate; chunks 06-10 under one whose
     first distinction was redirect-versus-contradict, with an explicit note
     that warm correction is a redirect. If those two groups disagree, the
     result is an artifact of my wording and not a fact about people. This is
     checked first, before anything is reported.

  2. Does the response depend on WHAT was claimed? Every parent here was already
     judged, so this asks directly: when a person says God, does something
     different come back than when they say family?

  3. Do alignment and affiliation come apart, and where? That is the whole
     reason for using Stivers' two axes rather than a single agree/disagree
     scale.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEGO = ROOT / "_nego"
EARLY = {f"out_0{i}.json" for i in range(6)}          # original wording
LATE = {"out_06.json", "out_07.json", "out_08.json",
        "out_09.json", "out_10.json"}                 # revised wording


def load():
    meta = json.loads((NEGO / "_meta.json").read_text(encoding="utf-8"))
    rows = []
    for f in sorted(NEGO.glob("out_*.json")):
        try:
            recs = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ! unreadable: {f.name}")
            continue
        group = "early" if f.name in EARLY else "late"
        for r in recs:
            m = meta.get(r.get("id"))
            if not m:
                continue
            rows.append({
                "move": r.get("move"),
                "alignment": r.get("alignment"),
                "affiliation": r.get("affiliation"),
                "lang": m["lang"],
                "sources": m["sources"],
                "group": group,
                "file": f.name,
            })
    return rows


def pct(c: Counter, n: int):
    return {k: 100 * v / max(1, n) for k, v in c.items()}


def two_prop_z(a, na, b, nb):
    """Is move-rate a/na different from b/nb?"""
    if na == 0 or nb == 0:
        return 0.0
    p1, p2 = a / na, b / nb
    p = (a + b) / (na + nb)
    se = sqrt(p * (1 - p) * (1 / na + 1 / nb))
    return (p1 - p2) / se if se else 0.0


def main() -> None:
    rows = load()
    print(f"coded replies: {len(rows):,}\n")

    # ---- 1. the prompt-sensitivity check, before anything else ----
    early = [r for r in rows if r["group"] == "early"]
    late = [r for r in rows if r["group"] == "late"]
    print("=== 1. DID MY WORDING MAKE THE RESULT? ===")
    print(f"  early wording (chunks 00-05): {len(early):,} replies")
    print(f"  late  wording (chunks 06-10): {len(late):,} replies")
    ce, cl = Counter(r["move"] for r in early), Counter(r["move"] for r in late)
    print(f"\n  {'move':14s}{'early':>8s}{'late':>8s}{'z':>7s}")
    worst = 0.0
    for mv in sorted(set(ce) | set(cl), key=lambda m: -(ce[m] + cl[m])):
        z = two_prop_z(ce[mv], len(early), cl[mv], len(late))
        worst = max(worst, abs(z))
        flag = "  <-- differs" if abs(z) > 2.6 else ""
        print(f"  {str(mv):14s}{100*ce[mv]/max(1,len(early)):7.1f}%"
              f"{100*cl[mv]/max(1,len(late)):7.1f}%{z:7.1f}{flag}")
    print(f"\n  largest divergence: z = {worst:.1f}")
    print("  -> " + ("WORDING MATTERS. Do not pool these; recode the early group."
                     if worst > 2.6 else
                     "The two wordings agree. Pooling is defensible."))

    # ---- 2. the distribution ----
    print("\n=== 2. WHAT PEOPLE DO WITH SOMEONE ELSE'S MEANING ===")
    cm = Counter(r["move"] for r in rows)
    for mv, n in cm.most_common():
        print(f"  {100*n/len(rows):5.1f}%  {str(mv):14s}{n:>5}")

    # ---- 3. the two axes ----
    print("\n=== 3. ALIGNMENT vs AFFILIATION ===")
    grid = Counter((r["alignment"], r["affiliation"]) for r in rows)
    aff_order = ["affiliates", "neutral", "disaffiliates"]
    ali_order = ["aligns", "neutral", "disaligns"]
    print(f"  {'':12s}" + "".join(f"{a:>15s}" for a in aff_order))
    for al in ali_order:
        line = f"  {al:12s}"
        for af in aff_order:
            line += f"{100*grid[(al,af)]/len(rows):14.1f}%"
        print(line)
    split = sum(v for (al, af), v in grid.items()
                if (al == "aligns" and af == "disaffiliates")
                or (al == "disaligns" and af == "affiliates"))
    print(f"\n  the axes come apart in {100*split/len(rows):.1f}% of replies")
    print("  (warm but off-topic, or engaged but declining). A single")
    print("  agree/disagree scale would score every one of those wrong.")

    # ---- 4. does the response depend on the claim? ----
    print("\n=== 4. DOES WHAT THEY SAID CHANGE WHAT COMES BACK? ===")
    by_src = defaultdict(Counter)
    tot_src = Counter()
    for r in rows:
        for s in r["sources"]:
            by_src[s][r["move"]] += 1
            tot_src[s] += 1
    interesting = [s for s, n in tot_src.most_common() if n >= 80]
    moves = [m for m, _ in cm.most_common(7)]
    print(f"  {'claimed':24s}{'n':>6s}" + "".join(f"{str(m)[:9]:>10s}" for m in moves))
    for s in interesting:
        line = f"  {s:24s}{tot_src[s]:>6}"
        for m in moves:
            line += f"{100*by_src[s][m]/tot_src[s]:9.0f}%"
        print(line)

    # religion vs the rest, on the moves that matter
    rel = by_src.get("religion_transcendence", Counter())
    nrel = Counter()
    nn = 0
    for s, c in by_src.items():
        if s == "religion_transcendence":
            continue
        nrel.update(c)
        nn += tot_src[s]
    nr = tot_src.get("religion_transcendence", 0)
    print(f"\n  said God (n={nr}) vs said anything else (n={nn}):")
    for m in ["echo", "bless", "contradict", "redirect", "witness", "confess", "amplify"]:
        z = two_prop_z(rel[m], nr, nrel[m], nn)
        mark = "  <-- real" if abs(z) > 2.6 else ""
        print(f"    {m:12s}{100*rel[m]/max(1,nr):6.1f}% vs {100*nrel[m]/max(1,nn):5.1f}%   z={z:5.1f}{mark}")

    # ---- 5. by language ----
    print("\n=== 5. BY LANGUAGE (share disaffiliating) ===")
    bl = defaultdict(lambda: [0, 0])
    for r in rows:
        bl[r["lang"]][0] += 1
        if r["affiliation"] == "disaffiliates":
            bl[r["lang"]][1] += 1
    for lg in sorted(bl, key=lambda l: -bl[l][1] / max(1, bl[l][0])):
        n, d = bl[lg]
        print(f"  {lg:4s} n={n:>4}  {100*d/max(1,n):5.1f}% decline the claim")


if __name__ == "__main__":
    main()
