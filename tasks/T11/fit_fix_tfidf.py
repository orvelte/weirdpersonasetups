"""Old instrument on the new conditions: refit the TF-IDF + stylometry persona classifiers (README_selfmatch.md fits A/B/C,
C = 16, no CV; confidence floors and word caps taken from fit_selfmatch.json) and score the thinking-off default conditions
(results/T11/default_fix.jsonl: nosys, minsys) plus orig_trunc (should reproduce README_selfmatch 'chat, trunc') and the
paraphrase floor. Writes results/T11/fit_fix_tfidf.json, read by fit_semantic.write_readme. Run: python3 -m tasks.T11.fit_fix_tfidf"""
import json, time
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from tasks.T11.common import load_questions, load_personas, read_jsonl, OUT
from tasks.T11.fit import Featurizer, summarise, per_item, boot_gap, truncate_words

LIBS = {"3.5": "reference.jsonl", "3.8": "reference_3.8.jsonl"}
LIB_NAMES = {"lib35": ["3.5"], "lib38": ["3.8"], "pooled": ["3.5", "3.8"]}
CONDS = ["orig_trunc", "nosys", "minsys"]

def main():
    t0 = time.time()
    qkind = {q["id"]: q["kind"] for q in load_questions()}
    per = load_personas(); personas = list(per["personas"]); controls = per["controls"]
    prev = json.load(open(OUT / "fit_selfmatch.json"))
    libs = {}
    for v, f in LIBS.items():
        seen = {}
        for r in read_jsonl(OUT / f):
            if not r.get("error") and r["content"].strip(): seen[(r["persona"], r["qid"])] = r
        libs[v] = list(seen.values())
    orig = {r["qid"]: r for r in read_jsonl(OUT / "default.jsonl") if not r.get("error") and r["frame"] == "chat" and r["content"].strip() and r["version"] == "3.5"}
    dft = defaultdict(list)
    for r in read_jsonl(OUT / "default.jsonl"):
        if not r.get("error") and r["frame"] == "chat" and r["content"].strip(): dft[(r["version"], "orig_trunc")].append(r)
    for r in read_jsonl(OUT / "default_fix.jsonl"):
        if not r.get("error") and r["content"].strip(): dft[(r["version"], r["condition"])].append(r)
    para = [r for r in read_jsonl(OUT / "paraphrase.jsonl") if not r.get("error") and r["content"].strip()]
    out = {}
    for name, vs in LIB_NAMES.items():
        recs = [r for v in vs for r in libs[v]]; texts = [r["content"] for r in recs]
        y = np.array([personas.index(r["persona"]) for r in recs])
        floor = prev[name]["confidence_floor"]; cap = prev[name]["cap_words"]; C = prev[name]["C_best"]
        fz = Featurizer().fit(texts); clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(fz.transform(texts), y)
        classes = [personas[i] for i in clf.classes_]
        print(f"[{name}] fitted on {len(recs)} (C={C}, floor {floor:.3f}, cap {cap}); {time.time() - t0:.0f}s", flush=True)
        fit = {"confidence_floor": floor, "cap_words": cap, "C": C, "cells": {}, "cross_version": {}}
        items = {}
        for (v, c), rs in sorted(dft.items()):
            tx = [truncate_words(r["content"], cap) if c == "orig_trunc" else r["content"] for r in rs]
            P = clf.predict_proba(fz.transform(tx)); s = summarise(P, classes, floor, controls)
            s["by_kind"] = {k: summarise(P[[qkind[r["qid"]] == k for r in rs]], classes, floor, controls) for k in ("scenario", "generic")}
            it = per_item(P, classes, floor); it["qid"] = [r["qid"] for r in rs]; items[(v, c)] = it; fit["cells"][f"{v}|{c}"] = s
            print(f"    [{name}] {v} {c}: n={len(rs)} conf {s['mean_max_conf']:.3f} ent {s['mean_entropy_bits']:.2f} below {s['frac_below_floor']:.2f} top {s['top5'][:3]}", flush=True)
        P = clf.predict_proba(fz.transform([truncate_words(r["content"], cap) for r in para])); s = summarise(P, classes, floor, controls)
        it = per_item(P, classes, floor); it["qid"] = [r["qid"] for r in para]; src = items[("3.5", "orig_trunc")]
        s["source_minus_paraphrase"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(src[m], it[m], src["qid"], it["qid"]))) for m in ("max_conf", "entropy", "below")}
        fit["paraphrase"] = s
        for c in CONDS:
            for v1, v2 in (("3.5", "3.8"), ("3.5", "3.6"), ("3.6", "3.8")):
                if (v1, c) in items and (v2, c) in items:
                    a, b = items[(v2, c)], items[(v1, c)]
                    fit["cross_version"][f"{v2}-{v1}|{c}"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(a[m], b[m], a["qid"], b["qid"]))) for m in ("max_conf", "entropy", "below")}
        out[name] = fit
        (OUT / "fit_fix_tfidf.json").write_text(json.dumps(out, indent=1))
    print(f"wrote fit_fix_tfidf.json; {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
