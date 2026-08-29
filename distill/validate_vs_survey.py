"""Test the claim this project keeps making.

Five times now the write-up has said the language ordering "reproduces real-world
religiosity". That has never been measured. This measures it.

Reference: the Gallup 2009 global poll, "Is religion important in your daily
life?", percent yes, by country. Fetched from the Wikipedia table that reproduces
it, so the number is checkable rather than remembered.

Ours: the share of stated meanings that are religious, per language, from two
independent instruments built at different times with different taxonomies:
  MEANING  what people say gives life meaning
  COPING   what people reach for in irreversible situations

A language is not a country, so each is mapped to the country holding the largest
share of its speakers on YouTube. That mapping is a judgement and is printed in
full so it can be argued with.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from math import sqrt

UA = {"User-Agent": "meaning-of-life-corpus/0.1 (research; contact jerrrr92@gmail.com)"}

# our two instruments, religion as a share of stated meanings
MEANING = {"ar": 70, "id": 53, "es": 49, "ko": 47, "pt": 45, "en": 44, "hi": 41, "tr": 34, "ja": 11}
COPING = {"ar": 52, "tr": 39, "id": 36, "pt": 27, "hi": 20, "es": 18, "en": 9, "ja": 2}

# language -> country, by largest speaker population in the collected seams
LANG_COUNTRY = {
    "ar": ("EGY", "Egypt"),
    "id": ("IDN", "Indonesia"),
    "es": ("MEX", "Mexico"),
    "ko": ("KOR", "South Korea"),
    "pt": ("BRA", "Brazil"),
    "en": ("USA", "United States"),
    "hi": ("IND", "India"),
    "tr": ("TUR", "Turkey"),
    "ja": ("JPN", "Japan"),
}
NAME_TO_ISO = {
    "Egypt": "EGY", "Indonesia": "IDN", "Mexico": "MEX", "South Korea": "KOR",
    "Brazil": "BRA", "United States": "USA", "India": "IND", "Turkey": "TUR",
    "Japan": "JPN",
}


def gallup() -> dict[str, int]:
    u = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "parse", "page": "Importance of religion by country",
        "prop": "wikitext", "redirects": 1, "format": "json"})
    wt = json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers=UA), timeout=30).read().decode("utf-8")
    )["parse"]["wikitext"]["*"]

    out: dict[str, int] = {}
    for line in wt.split("\n"):
        if "||" not in line:
            continue
        m = re.search(r"\{\{(?:flag\|)?([A-Za-z ]{3,30})\}\}\s*\|\|\s*(\d+)%", line)
        if not m:
            continue
        token, pct = m.group(1).strip(), int(m.group(2))
        iso = token.upper() if len(token) == 3 else NAME_TO_ISO.get(token)
        if iso:
            out[iso] = pct
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    return pearson(rank(xs), rank(ys))


def main() -> None:
    g = gallup()
    print(f"Gallup 2009 table parsed: {len(g)} countries\n")

    for label, ours in (("MEANING", MEANING), ("COPING", COPING)):
        pairs = []
        for lang, val in ours.items():
            iso, country = LANG_COUNTRY[lang]
            if iso in g:
                pairs.append((lang, country, val, g[iso]))
        if len(pairs) < 4:
            print(f"{label}: not enough matched countries ({len(pairs)})")
            continue

        print(f"=== {label}: ours vs Gallup ===")
        print(f"{'lang':5s}{'country':16s}{'ours':>7s}{'Gallup':>9s}")
        for lang, country, val, gv in sorted(pairs, key=lambda p: -p[2]):
            print(f"{lang:5s}{country:16s}{val:6d}%{gv:8d}%")
        xs = [p[2] for p in pairs]
        ys = [p[3] for p in pairs]
        r = pearson(xs, ys)
        rho = spearman(xs, ys)
        n = len(pairs)
        # significance of r, two-tailed, via t with n-2 df
        t = r * sqrt((n - 2) / max(1e-9, 1 - r * r)) if abs(r) < 1 else float("inf")
        print(f"\n  n = {n}")
        print(f"  Pearson  r   = {r:+.3f}   (t = {t:.2f}, df = {n-2})")
        print(f"  Spearman rho = {rho:+.3f}")
        verdict = ("STRONG, the claim holds" if r >= .8 else
                   "MODERATE, directionally right" if r >= .5 else
                   "WEAK, the claim does not hold")
        print(f"  -> {verdict}\n")


if __name__ == "__main__":
    main()
