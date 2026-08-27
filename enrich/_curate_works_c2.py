"""Curated works pass (c2) — hand-authored, on the tidied taxonomy.

Supersedes c1. Changes: the reject axis now uses the false_idols vocabulary
(so a work names what it actually attacks — consumerism, status, conformity,
escapism…), the new positive tags are used where apt (purpose_cause,
achievement_ambition, beauty_awe), and the old stance-tags (absurdism_rejection,
uncertainty) are dropped from the axes — they live in the `stance` field now.

Provenance stamped `claude-in-chat#c2`. Run once:
    uv run python -m enrich._curate_works_c2
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from db import db as dbmod
from enrich import extract
from enrich.works_extract import ensure_table

MODEL = "claude-in-chat#c2"

# axis vocab guard — fail loudly if a hand-typed tag drifts off the taxonomy.
_TAX = extract.load_taxonomy()
_POS = set(_TAX["meaning_sources"])
_IDOL = set(_TAX["false_idols"])

C2 = {
    # ---------------- films ----------------
    "2001: A Space Odyssey": dict(
        stance="ambiguous",
        affirms=["growth_self", "religion_transcendence", "beauty_awe"], rejects=[], unresolved=[],
        essence="Humanity's meaning lies in transcending its current form toward something vast and incomprehensible; the film withholds any tidy answer.",
        pull_theme="Human meaning lies in evolutionary transcendence beyond our comprehension."),
    "American Beauty (1999 film)": dict(
        stance="critical",
        affirms=["beauty_awe", "pleasure_experience", "growth_self", "freedom_autonomy"],
        rejects=["consumerism_materialism", "vanity_image", "conformity"], unresolved=[],
        essence="Beauty and meaning hide inside ordinary life, buried under suburban performance, image, and consumer emptiness.",
        pull_theme="Look closer: beauty hides beneath suburban conformity and consumer emptiness."),
    "Blade Runner": dict(
        stance="ambiguous",
        affirms=["pleasure_experience", "connection_belonging", "beauty_awe"], rejects=[], unresolved=[],
        essence="Humanity is defined not by origin but by empathy and the poignancy of mortal, fleeting experience — the replicant feels it most.",
        pull_theme="To be human is to feel time slipping and value it."),
    "Everything Everywhere All at Once": dict(
        stance="affirming",
        affirms=["family_children", "relationships", "connection_belonging"], rejects=[], unresolved=[],
        essence="In a meaningless multiverse (the everything-bagel of nihilism), love and kindness toward family are the meaning we deliberately choose.",
        pull_theme="Nothing matters, so be kind — love is the meaning we choose."),
    "Fight Club": dict(
        stance="critical",
        affirms=[], rejects=["consumerism_materialism", "power_domination", "conformity"],
        unresolved=["freedom_autonomy"],
        essence="A savage double critique: of consumer emptiness, and of the violent, nihilistic masculinity offered as its cure — Tyler's 'liberation' is exposed as fascism.",
        pull_theme="Consumer emptiness and its violent 'cure' are both indicted."),
    "Groundhog Day (film)": dict(
        stance="affirming",
        affirms=["service_others", "growth_self", "romantic_love"], rejects=["hedonism_numbing"], unresolved=[],
        essence="Escaping the self-serving loop requires becoming good; mastery, kindness, and love finally make a day worth repeating.",
        pull_theme="A life becomes worth living once lived for others."),
    "Ikiru": dict(
        stance="affirming",
        affirms=["legacy", "service_others", "purpose_cause"],
        rejects=["conformity", "hedonism_numbing"], unresolved=[],
        essence="A dying bureaucrat, his decades of paperwork exposed as death-in-life, finds meaning by building one small good thing for others.",
        pull_theme="One selfless act redeems a life of empty routine."),
    "Manchester by the Sea (film)": dict(
        stance="tragic",
        affirms=[], rejects=[], unresolved=["family_children", "connection_belonging"],
        essence="Some grief cannot be healed; the film honestly refuses the redemption arc — 'I can't beat it.'",
        pull_theme="Some grief has no recovery; 'I can't beat it.'"),
    "Synecdoche, New York": dict(
        stance="tragic",
        affirms=[], rejects=["legacy", "control_rationalism"], unresolved=[],
        essence="The attempt to master and immortalize life through ever-grander art consumes the life itself; you spend your years representing what you forgot to live.",
        pull_theme="Trying to capture life whole, you miss living it."),
    "The Fountain": dict(
        stance="affirming",
        affirms=["romantic_love", "religion_transcendence", "growth_self"], rejects=["legacy"], unresolved=[],
        essence="Meaning comes from accepting mortality with love, not from conquering death; the quest for eternal life is the trap to release.",
        pull_theme="Stop fighting death — accept it, and love."),
    "The Seventh Seal": dict(
        stance="ambiguous",
        affirms=["service_others", "relationships"], rejects=[], unresolved=["religion_transcendence"],
        essence="Facing death and God's silence, the knight gets no cosmic answer; meaning is found only in one small act of human kindness.",
        pull_theme="God is silent; one kind act is the only answer."),
    "The Tree of Life (film)": dict(
        stance="affirming",
        affirms=["family_children", "religion_transcendence", "nature"],
        rejects=["achievement_striving"], unresolved=[],
        essence="Life is a tension between the way of nature (self-assertion, worldly ambition) and the way of grace (love); the film leans toward grace.",
        pull_theme="Choose grace over nature: love, not self-assertion."),
    "The Truman Show": dict(
        stance="affirming",
        affirms=["freedom_autonomy", "growth_self"],
        rejects=["escapism_distraction", "conformity"], unresolved=[],
        essence="An authentic, uncertain, frightening freedom is worth more than a safe, manufactured paradise built for someone else's consumption.",
        pull_theme="Better a real, frightening world than a comfortable lie."),
    # ---------------- novels ----------------
    "Crime and Punishment": dict(
        stance="critical",
        affirms=["religion_transcendence", "service_others", "romantic_love"],
        rejects=["power_domination", "control_rationalism"], unresolved=[],
        essence="The 'extraordinary man' who places himself above morality commits spiritual suicide; redemption comes through suffering, love, and faith.",
        pull_theme="The 'extraordinary man' above morality is a murderous lie."),
    "Fahrenheit 451": dict(
        stance="critical",
        affirms=["growth_self", "freedom_autonomy"],
        rejects=["escapism_distraction", "hedonism_numbing", "conformity"], unresolved=[],
        essence="A society anesthetized by mindless entertainment loses thought and freedom; meaning lives in books, memory, and the inner life.",
        pull_theme="Mindless entertainment is the anesthetic that kills free thought."),
    "Life of Pi": dict(
        stance="affirming",
        affirms=["religion_transcendence"], rejects=["control_rationalism"], unresolved=[],
        essence="When the truth is unknowable, choosing the story that contains God and wonder is the braver, better choice than dry factuality.",
        pull_theme="Given two stories, choose the one with God."),
    "Nausea (novel)": dict(
        stance="absurdist",
        affirms=["freedom_autonomy", "craft_work"], rejects=["conformity"], unresolved=[],
        essence="Existence is superfluous and contingent — the nausea of things just being there; the only exit is self-made freedom, and perhaps art.",
        pull_theme="Existence is contingent and absurd; meaning must be self-made."),
    "Notes from Underground": dict(
        stance="ironic",
        affirms=["freedom_autonomy"], rejects=["control_rationalism"], unresolved=[],
        essence="Against every rationalist utopia (the Crystal Palace), the underground man insists man will choose suffering over a calculated paradise — to prove his irrational free will.",
        pull_theme="Man will choose pain over a rational paradise, to stay free."),
    "Siddhartha (novel)": dict(
        stance="affirming",
        affirms=["growth_self", "religion_transcendence", "pleasure_experience"],
        rejects=["consumerism_materialism"], unresolved=[],
        essence="Wisdom cannot be taught, only lived; meaning comes through the full spectrum of experience toward unity — though the merchant's greed is a trap along the way.",
        pull_theme="Wisdom is lived through, not taught: seek the whole."),
    "Slaughterhouse-Five": dict(
        stance="ironic",
        affirms=[], rejects=["power_domination"], unresolved=["freedom_autonomy"],
        essence="In a violent, seemingly deterministic universe, the Tralfamadorian 'so it goes' is left ambiguous — trauma anesthetic or hard-won peace?",
        pull_theme="'So it goes' — is fatalism wisdom, or trauma talking?"),
    "Steppenwolf (novel)": dict(
        stance="affirming",
        affirms=["growth_self", "pleasure_experience", "religion_transcendence"],
        rejects=["conformity"], unresolved=[],
        essence="The self-divided intellectual must embrace life's sensual chaos and learn to laugh to become whole; the intellect alone starves the soul.",
        pull_theme="Learn to laugh and live; the intellect alone starves you."),
    "The Brothers Karamazov": dict(
        stance="ambiguous",
        affirms=["religion_transcendence", "service_others", "relationships"], rejects=[], unresolved=[],
        essence="Faith lived as active love is Dostoevsky's answer — one that answers, but never silences, the strongest case against God: the suffering of children.",
        pull_theme="Active love answers, but never silences, the problem of suffering."),
    "The Death of Ivan Ilyich": dict(
        stance="critical",
        affirms=["relationships", "religion_transcendence"],
        rejects=["status_prestige", "conformity", "achievement_striving"], unresolved=[],
        essence="A 'proper,' respectable, socially correct life is exposed by death as a hollow lie; only simple, authentic compassion redeems it.",
        pull_theme="'What if my whole life was wrong?' — respectability is a lie."),
    "The Metamorphosis": dict(
        stance="tragic",
        affirms=[], rejects=["achievement_striving", "family_children"], unresolved=[],
        essence="A man whose entire worth was his usefulness as provider becomes, when he can no longer provide, a burden his own family discards.",
        pull_theme="Reduced to a burden, the provider is discarded by his own."),
    "The Myth of Sisyphus": dict(
        stance="absurdist",
        affirms=["freedom_autonomy", "pleasure_experience"],
        rejects=["religion_transcendence", "escapism_distraction"], unresolved=[],
        essence="Life is absurd and without given meaning; refusing both suicide and the 'leap of faith,' one lives fully through lucid revolt — imagine Sisyphus happy.",
        pull_theme="No hope, no God — revolt, and imagine Sisyphus happy."),
    "The Old Man and the Sea": dict(
        stance="affirming",
        affirms=["craft_work", "growth_self", "nature"], rejects=[], unresolved=[],
        essence="Dignity lies in skill, endurance, and the struggle itself, held with deep respect for a worthy adversary — destroyed, perhaps, but not defeated.",
        pull_theme="A man can be destroyed but not defeated."),
    "The Stranger (Camus novel)": dict(
        stance="absurdist",
        affirms=["freedom_autonomy", "pleasure_experience"],
        rejects=["conformity", "religion_transcendence"], unresolved=[],
        essence="Society condemns Meursault less for murder than for refusing to perform grief; freedom is accepting the benign indifference of the universe.",
        pull_theme="Condemned less for murder than for not performing grief."),
    "The Unbearable Lightness of Being": dict(
        stance="ambiguous",
        affirms=["romantic_love"], rejects=[], unresolved=["freedom_autonomy"],
        essence="Is a meaningful life found in weightless freedom or in the heavy burdens of love and commitment? The novel deliberately refuses to decide.",
        pull_theme="Lightness or weight — which makes a life? No answer given."),
}


def main() -> None:
    # validate every tag against the tidied taxonomy before touching the DB
    for title, j in C2.items():
        for t in j["affirms"] + j["unresolved"]:
            assert t in _POS, f"{title}: '{t}' not a positive source"
        for t in j["rejects"]:
            assert t in _POS or t in _IDOL, f"{title}: '{t}' not a source or false idol"
    conn = dbmod.get_conn()
    dbmod.init_db(conn)
    ensure_table(conn)
    have = {r[0] for r in conn.execute("SELECT id FROM works")}
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for title, j in C2.items():
        if title not in have:
            print(f"  WARN: no work row for {title!r} — skipped")
            continue
        conn.execute(
            "INSERT OR REPLACE INTO work_enrichment(work_id, states_meaning, stance,"
            " affirms_json, rejects_json, unresolved_json, essence, pull_theme, model, enriched_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (title, 1, j["stance"], json.dumps(j["affirms"]), json.dumps(j["rejects"]),
             json.dumps(j["unresolved"]), j["essence"], j["pull_theme"], MODEL, now),
        )
        n += 1
    conn.commit()
    conn.close()
    print(f"curated {n} works as {MODEL}")


if __name__ == "__main__":
    main()
