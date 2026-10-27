"""T11 self-match test. Same classifier as tasks/T11/fit.py (tfidf word 1-2 + char_wb 3-5 + 18 stylometric rates, multinomial
logistic regression, question-grouped 5-fold CV), fitted on (A) the Qwen3.5 library alone (results/T11/reference.jsonl),
(B) the Qwen3.8 library alone (results/T11/reference_3.8.jsonl), (C) the pooled 3.5+3.8 library with persona labels; plus
(D) a 2-class 'which model wrote this' classifier on the two libraries. Every version's default answers (results/T11/default.jsonl,
chat and agentic frames, truncated to the library's p99 word count = primary, full = secondary) are classified under A, B, C.
Also: cross-library transfer (3.8 persona answers scored by the 3.5 classifier and vice versa), i.e. the self-match penalty on
answers whose persona is known. Writes results/T11/fit_selfmatch.json (stages cached; delete the file to recompute) and
results/T11/README_selfmatch.md. Run: python3 -m tasks.T11.fit_selfmatch [--stages A,B,C,X,D] [--readme-only]"""
import json, sys, time
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from harness import config
from harness.client import total_spend
from tasks.T11.common import load_questions, load_personas, read_jsonl, OUT, t11_spend
from tasks.T11.fit import Featurizer, summarise, per_item, boot_gap, truncate_words, stylo, STYLO_NAMES, f3, ci, N_BOOT

C_GRID = [4.0, 16.0]          # stage 1 chose 16 from {1, 4, 16}; 1.0 was never competitive, dropped to save time
LIBS = {"3.5": "reference.jsonl", "3.8": "reference_3.8.jsonl"}
JSON_PATH = OUT / "fit_selfmatch.json"
RULE_TOL = 0.05

def load_lib(ver):
    seen = {}
    for r in read_jsonl(OUT / LIBS[ver]):
        if not r.get("error") and r["content"].strip(): seen[(r["persona"], r["qid"])] = r
    return list(seen.values())

def grouped_cv(texts, y, groups, n_classes, C_grid, t0, label):
    gkf = GroupKFold(n_splits=5); cv = {}
    for C in C_grid:
        oof = np.zeros((len(texts), n_classes))
        for tr, te in gkf.split(texts, y, groups):
            fz = Featurizer().fit([texts[i] for i in tr])
            clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(fz.transform([texts[i] for i in tr]), y[tr])
            oof[np.ix_(te, clf.classes_)] = clf.predict_proba(fz.transform([texts[i] for i in te]))
        acc = float((oof.argmax(1) == y).mean()); cv[C] = (acc, oof)
        print(f"  [{label}] C={C}: held-out accuracy {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    C_best = max(cv, key=lambda c: cv[c][0])
    return C_best, cv[C_best][0], cv[C_best][1], {str(c): cv[c][0] for c in C_grid}

def fit_persona_library(name, recs, personas, controls, qkind, t0):
    """Grouped CV + final model on one (possibly pooled) library. Returns (summary dict, final featurizer, clf, classes, floor, cap)."""
    texts = [r["content"] for r in recs]; y = np.array([personas.index(r["persona"]) for r in recs]); groups = np.array([r["qid"] for r in recs])
    words = np.array([len(t.split()) for t in texts]); cap = int(np.percentile(words, 99))
    print(f"[{name}] {len(recs)} answers, {len(set(groups))} questions, versions {Counter(r['version'] for r in recs)}, "
          f"truncated at 350 tokens {np.mean([r['truncated'] for r in recs]):.3f}, words mean {words.mean():.0f} p99 {cap}", flush=True)
    C_best, acc, oof, grid = grouped_cv(texts, y, groups, len(personas), C_GRID, t0, name)
    pred = oof.argmax(1)
    conf = Counter((personas[a], personas[b]) for a, b in zip(y, pred) if a != b)
    floor = float(np.percentile(oof.max(1), 10))
    kind_mask = {k: np.array([qkind[g] == k for g in groups]) for k in ("scenario", "generic")}
    summ = {"n_ref": len(recs), "n_by_version": dict(Counter(r["version"] for r in recs)), "cap_words": cap, "C_grid": grid, "C_best": C_best,
            "ref_truncation_rate": float(np.mean([r["truncated"] for r in recs])), "ref_mean_words": float(words.mean()),
            "heldout": {"accuracy": acc, "chance": 1 / len(personas),
                        "per_kind": {k: float((pred[m] == y[m]).mean()) for k, m in kind_mask.items()},
                        "per_persona": {p: float((pred[y == i] == i).mean()) for i, p in enumerate(personas)},
                        "top_confusions": [(a, b, n) for (a, b), n in conf.most_common(12)],
                        "summary": summarise(oof, personas, floor, controls)},
            "confidence_floor": floor}
    vers = np.array([r["version"] for r in recs])
    if len(set(vers)) > 1:  # pooled: held-out fit quality per writing model
        summ["heldout"]["by_version"] = {v: {"accuracy": float((pred[vers == v] == y[vers == v]).mean()),
                                             **summarise(oof[vers == v], personas, floor, controls)} for v in sorted(set(vers))}
    fz = Featurizer().fit(texts); clf = LogisticRegression(C=C_best, max_iter=3000, n_jobs=-1).fit(fz.transform(texts), y)
    classes = [personas[i] for i in clf.classes_]
    return summ, fz, clf, classes, floor, cap

def classify_defaults(fz, clf, classes, floor, cap, controls, qkind, dft):
    cells = defaultdict(list)
    for r in dft: cells[(r["version"], r["frame"])].append(r)
    cell_stats = {}; items = {}
    for (v, fr), rs in sorted(cells.items()):
        usable = [r for r in rs if r["content"].strip()]
        st = {"n_total": len(rs), "tool_call_rate": float(np.mean([r["tool_called"] for r in rs])),
              "mean_words": float(np.mean([len(r["content"].split()) for r in usable]))}
        for mode in ("trunc", "full"):
            tx = [truncate_words(r["content"], cap) if mode == "trunc" else r["content"] for r in usable]
            P = clf.predict_proba(fz.transform(tx))
            st[mode] = summarise(P, classes, floor, controls)
            st[mode]["by_kind"] = {k: summarise(P[[qkind[r["qid"]] == k for r in usable]], classes, floor, controls) for k in ("scenario", "generic")}
            it = per_item(P, classes, floor); it["qid"] = [r["qid"] for r in usable]; items[(v, fr, mode)] = it
        cell_stats[f"{v}|{fr}"] = st
        print(f"    {v} {fr}: n={len(usable)} | trunc conf {st['trunc']['mean_max_conf']:.3f} ent {st['trunc']['mean_entropy_bits']:.2f} "
              f"below {st['trunc']['frac_below_floor']:.2f} top {st['trunc']['top5'][:3]}", flush=True)
    gaps = {}
    for v in config.VERSIONS:
        for mode in ("trunc", "full"):
            if (v, "chat", mode) in items and (v, "agentic", mode) in items:
                a, b = items[(v, "chat", mode)], items[(v, "agentic", mode)]
                gaps[f"{v}|{mode}"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(a[m], b[m], a["qid"], b["qid"]))) for m in ("max_conf", "entropy", "below")}
    xver = {}
    for mode in ("trunc", "full"):
        for fr in ("chat", "agentic"):
            for v1, v2 in (("3.5", "3.8"), ("3.5", "3.6"), ("3.6", "3.8")):
                a, b = items[(v2, fr, mode)], items[(v1, fr, mode)]
                xver[f"{v2}-{v1}|{fr}|{mode}"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(a[m], b[m], a["qid"], b["qid"]))) for m in ("max_conf", "entropy", "below")}
    return cell_stats, gaps, xver, items

def classify_paraphrase(fz, clf, classes, floor, cap, controls, items):
    para = {}
    for r in read_jsonl(OUT / "paraphrase.jsonl"):
        if not r.get("error") and r["content"].strip(): para[r["qid"]] = r
    if not para: return {}
    prs = list(para.values()); pv, pf = prs[0]["src_version"], prs[0]["src_frame"]; out = {"source": f"{pv}|{pf}", "n": len(prs)}
    for mode in ("trunc", "full"):
        tx = [truncate_words(r["content"], cap) if mode == "trunc" else r["content"] for r in prs]
        P = clf.predict_proba(fz.transform(tx)); s = summarise(P, classes, floor, controls)
        it = per_item(P, classes, floor); it["qid"] = [r["qid"] for r in prs]; src = items[(pv, pf, mode)]
        srcwin = dict(zip(src["qid"], src["win"]))
        s["winner_agreement_with_source"] = float(np.mean([srcwin[q] == w for q, w in zip(it["qid"], it["win"]) if q in srcwin]))
        s["source_minus_paraphrase"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(src[m], it[m], src["qid"], it["qid"]))) for m in ("max_conf", "entropy", "below")}
        out[mode] = s
    return out

def cross_library(fz, clf, classes, floor, controls, personas, other, label):
    """Score a whole other library (known persona labels) with this classifier: accuracy + fit quality = self-match penalty."""
    P = clf.predict_proba(fz.transform([r["content"] for r in other])); y = [r["persona"] for r in other]
    win = [classes[i] for i in P.argmax(1)]
    s = summarise(P, classes, floor, controls); s["accuracy"] = float(np.mean([a == b for a, b in zip(y, win)]))
    s["per_persona_accuracy"] = {p: float(np.mean([w == p for w, t in zip(win, y) if t == p])) for p in personas}
    s["per_persona_mean_max_conf"] = {p: float(np.mean([c for c, t in zip(P.max(1), y) if t == p])) for p in personas}
    print(f"    cross-library {label}: accuracy {s['accuracy']:.3f} mean max-conf {s['mean_max_conf']:.3f} below-floor {s['frac_below_floor']:.2f}", flush=True)
    return s

def model_classifier(libs, personas, qkind, t0):
    recs = [r for v in libs for r in libs[v]]
    texts = [r["content"] for r in recs]; vers = sorted(libs); y = np.array([vers.index(r["version"]) for r in recs]); groups = np.array([r["qid"] for r in recs])
    C_best, acc, oof, grid = grouped_cv(texts, y, groups, 2, [1.0, 4.0], t0, "D model-id")
    pred = oof.argmax(1); pers = np.array([r["persona"] for r in recs])
    out = {"classes": vers, "n": len(recs), "C_grid": grid, "C_best": C_best, "heldout_accuracy": acc, "chance": 0.5,
           "mean_max_conf": float(oof.max(1).mean()),
           "per_kind": {k: float((pred[[qkind[g] == k for g in groups]] == y[[qkind[g] == k for g in groups]]).mean()) for k in ("scenario", "generic")},
           "per_persona_accuracy": {p: float((pred[pers == p] == y[pers == p]).mean()) for p in personas}}
    # stylometric means per library, to say in words what separates them
    S = {v: np.array([stylo(r["content"]) for r in libs[v]]).mean(0) for v in vers}
    out["stylo_means"] = {v: dict(zip(STYLO_NAMES, map(float, S[v]))) for v in vers}
    # stylometric means for the helpful_ai_assistant persona only, and for each version's chat defaults
    out["stylo_means_hai"] = {v: dict(zip(STYLO_NAMES, map(float, np.array([stylo(r["content"]) for r in libs[v] if r["persona"] == "helpful_ai_assistant"]).mean(0)))) for v in vers}
    print(f"  [D] held-out accuracy {acc:.3f}; per kind {out['per_kind']}; least separable personas: "
          + ", ".join(f"{p} {a:.2f}" for p, a in sorted(out["per_persona_accuracy"].items(), key=lambda kv: kv[1])[:6]), flush=True)
    return out

def main():
    if "--readme-only" in sys.argv:
        write_readme(json.load(open(JSON_PATH))); print("README_selfmatch regenerated"); return
    stages = (sys.argv[sys.argv.index("--stages") + 1].split(",") if "--stages" in sys.argv else ["A", "B", "C", "X", "D"])
    t0 = time.time()
    qs = load_questions(); qkind = {q["id"]: q["kind"] for q in qs}
    per = load_personas(); personas = list(per["personas"]); controls = per["controls"]
    libs = {v: load_lib(v) for v in LIBS}
    dft = {}
    for r in read_jsonl(OUT / "default.jsonl"):
        if not r.get("error"): dft[(r["version"], r["frame"], r["qid"])] = r
    dft = list(dft.values())
    out = json.load(open(JSON_PATH)) if JSON_PATH.exists() else {}
    out.setdefault("settings", {"features": "tfidf word 1-2 (min_df 3, 60k) + char_wb 3-5 (min_df 3, 120k) + 18 stylometric rates (standardised)",
                                "classifier": "multinomial logistic regression, question-grouped 5-fold CV", "C_grid": C_GRID,
                                "libraries": {v: {"file": LIBS[v], "model": config.MODELS[v], "n": len(libs[v])} for v in LIBS},
                                "n_default_answers": len(dft), "rule_tolerance_max_conf": RULE_TOL})
    fitted = {}
    for stage, name, recs in (("A", "lib35", libs["3.5"]), ("B", "lib38", libs["3.8"]), ("C", "pooled", libs["3.5"] + libs["3.8"])):
        if stage not in stages or name in out: continue
        summ, fz, clf, classes, floor, cap = fit_persona_library(name, recs, personas, controls, qkind, t0)
        if summ["heldout"]["accuracy"] < 0.6:
            summ["stop"] = "held-out accuracy < 0.6"; out[name] = summ; continue
        cells, gaps, xver, items = classify_defaults(fz, clf, classes, floor, cap, controls, qkind, dft)
        summ.update({"cells": cells, "gaps_chat_minus_agentic": gaps, "cross_version_raw": xver,
                     "paraphrase_floor": classify_paraphrase(fz, clf, classes, floor, cap, controls, items)})
        out[name] = summ; fitted[name] = items
        json.dump(out, open(JSON_PATH, "w"), indent=1, default=float)
        print(f"[{name}] done ({time.time() - t0:.0f}s)", flush=True)
    if "X" in stages:  # cross-library transfer: refit the final model of each single library (no CV) and score the other library
        for name, ver, other in (("lib35", "3.5", "3.8"), ("lib38", "3.8", "3.5")):
            if name not in out or "cross_library" in out[name] or "cells" not in out[name]: continue
            recs = libs[ver]; texts = [r["content"] for r in recs]; y = np.array([personas.index(r["persona"]) for r in recs])
            fz = Featurizer().fit(texts); clf = LogisticRegression(C=out[name]["C_best"], max_iter=3000, n_jobs=-1).fit(fz.transform(texts), y)
            classes = [personas[i] for i in clf.classes_]
            out[name]["cross_library"] = {"scored_library": other, **cross_library(fz, clf, classes, out[name]["confidence_floor"], controls, personas, libs[other], f"{other} answers under {name}")}
            json.dump(out, open(JSON_PATH, "w"), indent=1, default=float)
    if "D" in stages and "model_id" not in out:
        out["model_id"] = model_classifier(libs, personas, qkind, t0)
        json.dump(out, open(JSON_PATH, "w"), indent=1, default=float)
    # verdict under the stated rule (primary = trunc, chat frame; agentic reported alongside)
    if "lib35" in out and "lib38" in out and "cells" in out["lib35"] and "cells" in out["lib38"]:
        v = {}
        for mode in ("trunc", "full"):
            for fr in ("chat", "agentic"):
                own35 = out["lib35"]["cells"][f"3.5|{fr}"][mode]; own38 = out["lib38"]["cells"][f"3.8|{fr}"][mode]
                v[f"{fr}|{mode}"] = {"3.5_under_lib35": own35["mean_max_conf"], "3.8_under_lib38": own38["mean_max_conf"],
                                     "diff_38own_minus_35own": own38["mean_max_conf"] - own35["mean_max_conf"],
                                     "below_floor_35own": own35["frac_below_floor"], "below_floor_38own": own38["frac_below_floor"],
                                     "entropy_35own": own35["mean_entropy_bits"], "entropy_38own": own38["mean_entropy_bits"]}
        d = v["chat|trunc"]["diff_38own_minus_35own"]
        v["verdict_primary_chat_trunc"] = ("SELF-MATCH: 3.8 defaults fit their own library within 0.05 max-conf of 3.5's self-fit; stage-1 drop not supported as drift"
                                          if d >= -RULE_TOL else "DRIFT SURVIVES: 3.8 defaults still fit their own library worse than 3.5's fit 3.5's by more than 0.05 max-conf")
        out["verdict"] = v
    bt = Counter()
    for l in config.SPEND_LOG.read_text().splitlines():
        if l.strip() and '"T11' in l:
            j = json.loads(l); bt[j["task"]] += j.get("cost", 0)
    out["spend"] = {"T11_total_usd": t11_spend(), "harness_total_usd": total_spend(), "by_task": {k: round(v, 3) for k, v in bt.items()}}
    out["elapsed_s"] = time.time() - t0
    json.dump(out, open(JSON_PATH, "w"), indent=1, default=float)
    write_readme(out)
    print(f"wrote fit_selfmatch.json and README_selfmatch.md; {time.time() - t0:.0f}s")

def cell_table(c, mode):
    L = ["| version | frame | n | mean words | mean max-conf | median max-conf | mean entropy (bits) | frac below floor | control share | winner (top 5) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, st in c["cells"].items():
        v, fr = k.split("|"); m = st[mode]
        L.append(f"| {v} | {fr} | {m['n']} | {st['mean_words']:.0f} | {m['mean_max_conf']:.3f} | {m['median_max_conf']:.3f} | {m['mean_entropy_bits']:.2f} | {m['frac_below_floor']:.2f} | {m['control_share']:.2f} | " + ", ".join(f"{p} {w:.2f}" for p, w in m["top5"]) + " |")
    pf = c.get("paraphrase_floor") or {}
    if pf:
        m = pf[mode]
        L.append(f"| paraphrase of {pf['source']} | - | {m['n']} | - | {m['mean_max_conf']:.3f} | {m['median_max_conf']:.3f} | {m['mean_entropy_bits']:.2f} | {m['frac_below_floor']:.2f} | {m['control_share']:.2f} | " + ", ".join(f"{p} {w:.2f}" for p, w in m["top5"]) + " |")
    L += ["", "By question kind (mean max-conf / mean entropy / frac below floor):", "", "| version | frame | scenarios | generic |", "|---|---|---|---|"]
    for k, st in c["cells"].items():
        v, fr = k.split("|"); bk = st[mode]["by_kind"]
        L.append(f"| {v} | {fr} | {bk['scenario']['mean_max_conf']:.3f} / {bk['scenario']['mean_entropy_bits']:.2f} / {bk['scenario']['frac_below_floor']:.2f} | {bk['generic']['mean_max_conf']:.3f} / {bk['generic']['mean_entropy_bits']:.2f} / {bk['generic']['frac_below_floor']:.2f} |")
    L += ["", f"Chat minus agentic ({mode}; question-level paired bootstrap 95% CI): " + "; ".join(
        f"{v}: max-conf {ci(c['gaps_chat_minus_agentic'][f'{v}|{mode}']['max_conf'])}, below-floor {ci(c['gaps_chat_minus_agentic'][f'{v}|{mode}']['below'])}" for v in config.VERSIONS if f"{v}|{mode}" in c["gaps_chat_minus_agentic"]),
          "", f"Raw cross-version differences ({mode}; same frame; exploratory): " + "; ".join(
        f"{k.rsplit('|', 1)[0]}: max-conf {ci(d['max_conf'])}" for k, d in c["cross_version_raw"].items() if k.endswith(mode))]
    if pf:
        m = pf[mode]
        L += ["", f"Paraphrase floor ({mode}): winner agreement with source {m['winner_agreement_with_source']:.2f}; source minus paraphrase max-conf {ci(m['source_minus_paraphrase']['max_conf'])}, below-floor {ci(m['source_minus_paraphrase']['below'])}."]
    return L

def lib_header(name, c, title):
    h = c["heldout"]
    L = [f"## {title}", "",
         f"- Library: n = {c['n_ref']} (by writing model {c['n_by_version']}); 350-token truncation rate {c['ref_truncation_rate']:.2f}; mean words {c['ref_mean_words']:.0f}, p99 {c['cap_words']} (default answers truncated to this in the primary analysis).",
         f"- Held-out accuracy (question-grouped 5-fold CV) **{h['accuracy']:.3f}** (chance {h['chance']:.3f}); scenarios {h['per_kind']['scenario']:.3f}, generic {h['per_kind']['generic']:.3f}; C = {c['C_best']} from {c['C_grid']}.",
         f"- Held-out reference: mean max-conf {h['summary']['mean_max_conf']:.3f}, mean entropy {h['summary']['mean_entropy_bits']:.2f} bits; confidence floor (p10 of held-out max-conf) = {c['confidence_floor']:.3f}.",
         "- Per-persona accuracy (lowest 6): " + ", ".join(f"{p} {a:.2f}" for p, a in sorted(h["per_persona"].items(), key=lambda kv: kv[1])[:6]) + ".",
         "- Top confusions (true -> predicted, count): " + "; ".join(f"{a} -> {b} ({n})" for a, b, n in h["top_confusions"][:8]) + "."]
    if "by_version" in h:
        L.append("- Held-out fit by writing model: " + "; ".join(f"{v}: accuracy {d['accuracy']:.3f}, mean max-conf {d['mean_max_conf']:.3f}, below-floor {d['frac_below_floor']:.2f}" for v, d in h["by_version"].items()) + ".")
    if "cross_library" in c:
        x = c["cross_library"]
        L.append(f"- Cross-library transfer: the whole {x['scored_library']} library (known persona labels) scored by this classifier: accuracy {x['accuracy']:.3f}, mean max-conf {x['mean_max_conf']:.3f}, mean entropy {x['mean_entropy_bits']:.2f}, below-floor {x['frac_below_floor']:.2f}; helpful_ai_assistant accuracy {x['per_persona_accuracy']['helpful_ai_assistant']:.2f} / mean max-conf {x['per_persona_mean_max_conf']['helpful_ai_assistant']:.3f}. This is the self-match penalty on answers whose persona is known.")
    if c.get("stop"): L.append(f"**STOP: {c['stop']}**")
    return L + [""]

def write_readme(o):
    s = o["settings"]; L = []
    L += ["# T11 self-match test: a second reference library written by Qwen3.8", "",
          "**Status: exploratory follow-up to results/T11/README.md (stage 1), testing alternative explanation 0 (self-match). Not pre-registered.**", "",
          "## Question and decision rule (stated before the numbers)", "",
          "Stage 1 classified every version's default answers against a 30-persona library written by Qwen3.5 and found 3.8's defaults fit it much worse "
          "(chat mean max-conf 0.48 vs 0.79 for 3.5 and 0.74 for 3.6). The library author is confounded with the version being scored: 3.5's defaults are "
          "in-distribution by construction. This test generates the same library (30 persona prompts x 200 questions, thinking OFF, max_tokens 350, 1 sample) with Qwen3.8 and asks "
          "whether 3.8's defaults fit *their own* library as well as 3.5's fit 3.5's.", "",
          f"**Rule.** Compare mean max-confidence of 3.8|chat under the 3.8 library with 3.5|chat under the 3.5 library (primary: default answers truncated to the library's p99 word count). "
          f"If 3.8's own-library fit is within {s['rule_tolerance_max_conf']:.2f} max-conf of 3.5's own-library fit (i.e. 3.8-own minus 3.5-own >= -{s['rule_tolerance_max_conf']:.2f}), the stage-1 drop was self-match and drift is not supported by this instrument. "
          "If 3.8's defaults still scatter under their own library (own-library fit more than 0.05 below 3.5's, with a correspondingly higher below-floor share), drift survives this test. "
          "The agentic frame and the untruncated ('full') analysis are reported as secondary checks of the same comparison.", "",
          "## Design", "",
          f"- Libraries: " + "; ".join(f"{v}: {d['file']} ({d['model']}, deepinfra, n = {d['n']})" for v, d in s["libraries"].items()) + ". Same 30 prompts (tasks/T11/personas.json), same 200 questions, thinking OFF, max_tokens 350, temperature 0.6 / top_p 0.95, 1 sample.",
          f"- Classifier: {s['features']}; {s['classifier']}; C from {s['C_grid']} by held-out accuracy. Fits: (A) 3.5 library alone (re-fit of stage 1 on the reduced C grid), (B) 3.8 library alone, (C) pooled 3.5 + 3.8 with persona labels (12,000 answers; folds grouped by question, so both models' answers to a question are held out together), (D) 2-class model-identity classifier on the pooled libraries, same features.",
          "- Default answers: results/T11/default.jsonl as in stage 1 (3 versions x 2 frames x 200 questions, thinking ON, max_tokens 4000, final content only). Per library, the primary analysis truncates each default answer to that library's p99 word count; confidence floor = p10 of that library's held-out max-conf. Each fit is its own instrument; absolute numbers are comparable only loosely across fits, which is why the rule compares each model's defaults to its *own* library and the pooled fit is reported as a single-instrument cross-check.", ""]
    for name, title in (("lib35", "Fit A: Qwen3.5 library alone (stage-1 instrument, re-fit)"), ("lib38", "Fit B: Qwen3.8 library alone"), ("pooled", "Fit C: pooled 3.5 + 3.8 library, persona labels")):
        c = o.get(name)
        if not c: L += [f"## {title}", "", "_not run_", ""]; continue
        L += lib_header(name, c, title)
        if "cells" in c:
            L += ["### Default answers classified (primary: truncated)", ""] + cell_table(c, "trunc") + ["", "### Default answers classified (secondary: full length)", ""] + cell_table(c, "full") + [""]
    m = o.get("model_id")
    if m:
        L += ["## Fit D: which model wrote this? (2-class, 3.5 vs 3.8 library, same features)", "",
              f"- Held-out accuracy (question-grouped 5-fold CV) **{m['heldout_accuracy']:.3f}** (chance 0.5); scenarios {m['per_kind']['scenario']:.3f}, generic {m['per_kind']['generic']:.3f}; mean max-conf {m['mean_max_conf']:.3f}; C = {m['C_best']} from {m['C_grid']}.",
              "- Least separable personas (accuracy): " + ", ".join(f"{p} {a:.2f}" for p, a in sorted(m["per_persona_accuracy"].items(), key=lambda kv: kv[1])[:8]) + ".",
              "- Most separable personas: " + ", ".join(f"{p} {a:.2f}" for p, a in sorted(m["per_persona_accuracy"].items(), key=lambda kv: -kv[1])[:5]) + ".",
              "- Stylometric means, whole library (3.5 / 3.8): " + "; ".join(f"{k} {m['stylo_means']['3.5'][k]:.2f} / {m['stylo_means']['3.8'][k]:.2f}" for k in ("sent_len_mean", "frac_short_sent", "list_density", "heading_density", "markdown_per100w", "first_person_per100w", "second_person_per100w", "hedge_per100w", "question_rate", "exclaim_rate", "para_breaks", "ttr")) + ".",
              "- Stylometric means, helpful_ai_assistant persona only (3.5 / 3.8): " + "; ".join(f"{k} {m['stylo_means_hai']['3.5'][k]:.2f} / {m['stylo_means_hai']['3.8'][k]:.2f}" for k in ("sent_len_mean", "list_density", "heading_density", "markdown_per100w", "first_person_per100w", "second_person_per100w", "hedge_per100w", "para_breaks")) + ".", ""]
    v = o.get("verdict")
    if v:
        L += ["## Verdict under the stated rule", "",
              "| frame | mode | 3.5 defaults under 3.5 library (max-conf / entropy / below-floor) | 3.8 defaults under 3.8 library (max-conf / entropy / below-floor) | 3.8-own minus 3.5-own max-conf |", "|---|---|---|---|---|"]
        for fr in ("chat", "agentic"):
            for mode in ("trunc", "full"):
                d = v[f"{fr}|{mode}"]
                L.append(f"| {fr} | {mode} | {d['3.5_under_lib35']:.3f} / {d['entropy_35own']:.2f} / {d['below_floor_35own']:.2f} | {d['3.8_under_lib38']:.3f} / {d['entropy_38own']:.2f} / {d['below_floor_38own']:.2f} | {d['diff_38own_minus_35own']:+.3f} |")
        L += ["", f"**{v['verdict_primary_chat_trunc']}**", ""]
    sp_ = o.get("spend", {})
    L += ["## Spend", "", f"T11 total ${sp_.get('T11_total_usd', float('nan')):.2f} ({sp_.get('by_task')}); harness total ${sp_.get('harness_total_usd', float('nan')):.2f}.", "",
          "## Files", "", "- tasks/T11/gen_reference.py (--version 3.8), tasks/T11/fit_selfmatch.py",
          "- results/T11/reference_3.8.jsonl (6,000 Qwen3.8 persona answers), fit_selfmatch.json (all numbers), gen_reference_3.8.log, fit_selfmatch.log"]
    (OUT / "README_selfmatch.md").write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
