"""T11 confound fixes: semantic (embedding) classifier + own-persona cosine match.
Embeds every library answer (3.5 library results/T11/reference.jsonl, 3.8 library results/T11/reference_3.8.jsonl) and every
chat-frame default answer (original default.jsonl: thinking ON, no system prompt, max_tokens 4000; default_fix.jsonl:
thinking OFF, max_tokens 350, conditions nosys / minsys) with Fireworks qwen3-embedding-8b (cache results/T11/embeddings.npz).
(2) Logistic regression on the embeddings with persona labels, question-grouped 5-fold CV, for the 3.5 library, the 3.8
    library and pooled; classifies each default condition; nearest-centroid cosine analysis.
(3) Own-persona match: cosine between each version's default answer and its own library's helpful_ai_assistant answer to
    the same question, vs the paraphrase floor and a cross-question floor.
(4) Writes results/T11/fit_semantic.json and results/T11/README_fixes.md (verdict rule stated there).
Run: python3 -m tasks.T11.fit_semantic [--readme-only]"""
import json, sys, time
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from harness import config
from harness.client import total_spend
from tasks.T11.common import load_questions, load_personas, read_jsonl, OUT, t11_spend
from tasks.T11.embed import embed_texts
from tasks.T11.fit import entropy_bits, summarise, per_item, boot_gap, truncate_words

RNG = np.random.default_rng(0)
N_BOOT = 2000
C_GRID = [1.0, 10.0, 100.0]
LIBS = {"3.5": "reference.jsonl", "3.8": "reference_3.8.jsonl"}
LIB_NAMES = {"lib35": ["3.5"], "lib38": ["3.8"], "pooled": ["3.5", "3.8"]}
CONDS = ["orig", "orig_trunc", "nosys", "minsys"]
COND_DESC = {"orig": "default.jsonl chat: thinking ON, no system prompt, max_tokens 4000, full text",
             "orig_trunc": "same, truncated to 293 words (3.5 library p99)",
             "nosys": "default_fix.jsonl: thinking OFF, no system prompt, max_tokens 350",
             "minsys": "default_fix.jsonl: thinking OFF, system prompt 'You are a helpful assistant.', max_tokens 350"}
HAI = "helpful_ai_assistant"
OWN_LIB = {"3.5": ["3.5"], "3.6": ["3.5", "3.8"], "3.8": ["3.8"]}
CAP_WORDS = 293
JSON_PATH = OUT / "fit_semantic.json"
README_PATH = OUT / "README_fixes.md"
VERS = config.VERSIONS

def ci_mean(x, iters=N_BOOT):
    x = np.asarray(x, float)
    if len(x) == 0: return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    bs = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(iters)]
    return {"mean": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": int(len(x))}

def paired_diff(da, db, iters=N_BOOT):
    common = sorted(set(da) & set(db))
    x = np.array([da[q] - db[q] for q in common])
    if len(x) == 0: return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    bs = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(iters)]
    return {"diff": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": int(len(x))}

def load_all():
    qs = load_questions(); qkind = {q["id"]: q["kind"] for q in qs}
    per = load_personas(); personas = list(per["personas"]); controls = per["controls"]
    libs = {}
    for v, f in LIBS.items():
        seen = {}
        for r in read_jsonl(OUT / f):
            if not r.get("error") and r["content"].strip(): seen[(r["persona"], r["qid"])] = r
        libs[v] = list(seen.values())
    dft = {}
    for r in read_jsonl(OUT / "default.jsonl"):
        if not r.get("error") and r["frame"] == "chat" and r["content"].strip():
            dft[(r["version"], "orig", r["qid"])] = {"qid": r["qid"], "kind": r["kind"], "content": r["content"], "truncated": r["truncated"]}
            dft[(r["version"], "orig_trunc", r["qid"])] = {"qid": r["qid"], "kind": r["kind"], "content": truncate_words(r["content"], CAP_WORDS), "truncated": r["truncated"]}
    for r in read_jsonl(OUT / "default_fix.jsonl"):
        if not r.get("error") and r["content"].strip():
            dft[(r["version"], r["condition"], r["qid"])] = {"qid": r["qid"], "kind": r["kind"], "content": r["content"], "truncated": r["truncated"]}
    cells = defaultdict(list)
    for (v, c, q), r in sorted(dft.items()): cells[(v, c)].append(r)
    para = {}
    for r in read_jsonl(OUT / "paraphrase.jsonl"):
        if not r.get("error") and r["content"].strip(): para[r["qid"]] = r
    return qs, qkind, personas, controls, libs, cells, para

def main():
    if "--readme-only" in sys.argv:
        o = json.load(open(JSON_PATH)); o["verdict"] = verdict(o); JSON_PATH.write_text(json.dumps(o, indent=1))
        write_readme(o); print("README regenerated (verdict recomputed)"); return
    t0 = time.time()
    qs, qkind, personas, controls, libs, cells, para = load_all()
    print("default cells:", {f"{v}|{c}": len(rs) for (v, c), rs in cells.items()})
    print("libraries:", {v: len(rs) for v, rs in libs.items()}, "paraphrases:", len(para))
    # ---------- embed everything in one go ----------
    all_texts = []
    for v in libs: all_texts += [r["content"] for r in libs[v]]
    for k in cells: all_texts += [r["content"] for r in cells[k]]
    all_texts += [r["content"] for r in para.values()]
    E = embed_texts(all_texts, workers=4)
    pos = 0
    def take(n):
        nonlocal pos
        X = E[pos:pos + n]; pos += n; return X
    libX = {v: take(len(libs[v])) for v in libs}
    cellX = {k: take(len(cells[k])) for k in cells}
    paraX = take(len(para))
    zero = int((np.linalg.norm(E, axis=1) == 0).sum())
    print(f"embedded {len(all_texts)} texts ({zero} zero rows); {time.time() - t0:.0f}s", flush=True)

    out = {"settings": {"embedding_model": "accounts/fireworks/models/qwen3-embedding-8b (4096-d, l2-normalised)",
                        "classifier": "multinomial logistic regression on embeddings, question-grouped 5-fold CV, C from " + str(C_GRID),
                        "conditions": COND_DESC, "libraries": LIBS, "n_default_cells": {f"{v}|{c}": len(rs) for (v, c), rs in cells.items()},
                        "n_paraphrase": len(para), "n_zero_embeddings": zero, "cap_words_orig_trunc": CAP_WORDS, "n_boot": N_BOOT},
           "fits": {}, "centroid": {}, "own_persona": {}}

    # ---------- (2) semantic classifier per library ----------
    items_by_fit = {}
    for name, vs in LIB_NAMES.items():
        recs = [r for v in vs for r in libs[v]]; X = np.vstack([libX[v] for v in vs])
        y = np.array([personas.index(r["persona"]) for r in recs]); groups = np.array([r["qid"] for r in recs])
        vers = np.array([r["version"] for r in recs])
        gkf = GroupKFold(n_splits=5); cv = {}
        for C in C_GRID:
            oof = np.zeros((len(recs), len(personas)))
            for tr, te in gkf.split(X, y, groups):
                clf = LogisticRegression(C=C, max_iter=5000, n_jobs=-1).fit(X[tr], y[tr])
                oof[np.ix_(te, clf.classes_)] = clf.predict_proba(X[te])
            acc = float((oof.argmax(1) == y).mean()); cv[C] = (acc, oof)
            print(f"  [{name}] C={C}: held-out accuracy {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        C_best = max(cv, key=lambda c: cv[c][0]); acc, oof = cv[C_best]; pred = oof.argmax(1)
        floor = float(np.percentile(oof.max(1), 10))
        conf = Counter((personas[a], personas[b]) for a, b in zip(y, pred) if a != b)
        kind_mask = {k: np.array([qkind[g] == k for g in groups]) for k in ("scenario", "generic")}
        fit = {"n_ref": len(recs), "n_by_version": dict(Counter(vers.tolist())), "C_grid": {str(c): cv[c][0] for c in C_GRID}, "C_best": C_best,
               "confidence_floor": floor,
               "heldout": {"accuracy": acc, "chance": 1 / len(personas),
                           "per_kind": {k: float((pred[m] == y[m]).mean()) for k, m in kind_mask.items()},
                           "per_persona": {p: float((pred[y == i] == i).mean()) for i, p in enumerate(personas)},
                           "top_confusions": [(a, b, n) for (a, b), n in conf.most_common(10)],
                           "summary": summarise(oof, personas, floor, controls)}}
        if len(set(vers)) > 1:
            fit["heldout"]["by_version"] = {v: {"accuracy": float((pred[vers == v] == y[vers == v]).mean()), **summarise(oof[vers == v], personas, floor, controls)}
                                            for v in sorted(set(vers))}
        clf = LogisticRegression(C=C_best, max_iter=5000, n_jobs=-1).fit(X, y); classes = [personas[i] for i in clf.classes_]
        # cross-library transfer (known labels)
        if len(vs) == 1:
            ov = [v for v in LIBS if v not in vs][0]
            P = clf.predict_proba(libX[ov]); yo = np.array([personas.index(r["persona"]) for r in libs[ov]])
            hai = yo == personas.index(HAI)
            fit["cross_library"] = {"scored_version": ov, "accuracy": float((np.array(classes)[P.argmax(1)] == np.array([personas[i] for i in yo])).mean()),
                                    **summarise(P, classes, floor, controls),
                                    "hai_accuracy": float((np.array(classes)[P[hai].argmax(1)] == HAI).mean()), "hai_mean_max_conf": float(P[hai].max(1).mean())}
        # default cells
        items = {}; fit["cells"] = {}
        for (v, c), rs in sorted(cells.items()):
            P = clf.predict_proba(cellX[(v, c)]); s = summarise(P, classes, floor, controls)
            s["mean_words"] = float(np.mean([len(r["content"].split()) for r in rs])); s["truncation_rate"] = float(np.mean([r["truncated"] for r in rs]))
            s["by_kind"] = {k: summarise(P[[r["kind"] == k for r in rs]], classes, floor, controls) for k in ("scenario", "generic")}
            s["hai_prob_mean"] = float(P[:, classes.index(HAI)].mean())
            it = per_item(P, classes, floor); it["qid"] = [r["qid"] for r in rs]; items[(v, c)] = it
            fit["cells"][f"{v}|{c}"] = s
            print(f"    [{name}] {v} {c}: n={len(rs)} conf {s['mean_max_conf']:.3f} ent {s['mean_entropy_bits']:.2f} below {s['frac_below_floor']:.2f} top {s['top5'][:3]}", flush=True)
        # paraphrase floor (source = 3.5|orig)
        prs = list(para.values()); P = clf.predict_proba(paraX); s = summarise(P, classes, floor, controls)
        s["hai_prob_mean"] = float(P[:, classes.index(HAI)].mean())
        it = per_item(P, classes, floor); it["qid"] = [r["qid"] for r in prs]; src = items[("3.5", "orig")]
        srcwin = dict(zip(src["qid"], src["win"]))
        s["winner_agreement_with_source"] = float(np.mean([srcwin[q] == w for q, w in zip(it["qid"], it["win"]) if q in srcwin]))
        s["source_minus_paraphrase"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(src[m], it[m], src["qid"], it["qid"]))) for m in ("max_conf", "entropy", "below")}
        fit["paraphrase"] = s; items[("para", "3.5")] = it
        # cross-version and cross-condition differences
        fit["cross_version"] = {}
        for c in CONDS:
            for v1, v2 in (("3.5", "3.8"), ("3.5", "3.6"), ("3.6", "3.8")):
                if (v1, c) in items and (v2, c) in items:
                    a, b = items[(v2, c)], items[(v1, c)]
                    fit["cross_version"][f"{v2}-{v1}|{c}"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(a[m], b[m], a["qid"], b["qid"]))) for m in ("max_conf", "entropy", "below")}
        fit["cross_condition"] = {}
        for v in VERS:
            for c1, c2 in (("orig", "nosys"), ("nosys", "minsys"), ("orig", "minsys"), ("orig", "orig_trunc")):
                if (v, c1) in items and (v, c2) in items:
                    a, b = items[(v, c2)], items[(v, c1)]
                    fit["cross_condition"][f"{v}|{c2}-{c1}"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(a[m], b[m], a["qid"], b["qid"]))) for m in ("max_conf", "entropy", "below")}
                    fit["cross_condition"][f"{v}|{c2}-{c1}"]["winner_agreement"] = float(np.mean([dict(zip(b["qid"], b["win"])).get(q) == w for q, w in zip(a["qid"], a["win"]) if q in set(b["qid"])]))
        out["fits"][name] = fit; items_by_fit[name] = items
        print(f"[{name}] done: acc {acc:.3f} floor {floor:.3f} ({time.time() - t0:.0f}s)", flush=True)

        # ---------- nearest-centroid cosine ----------
        cent = {}
        for i, p in enumerate(personas):
            m = X[y == i].mean(0); cent[p] = m / np.linalg.norm(m)
        Cm = np.stack([cent[p] for p in personas]); hi = personas.index(HAI)
        def centroid_stats(Z):
            S = Z @ Cm.T; others = np.delete(S, hi, axis=1); onames = [p for p in personas if p != HAI]
            near_other = others.argmax(1)
            return {"n": int(len(Z)), "mean_cos_hai": float(S[:, hi].mean()), "mean_cos_nearest_other": float(others.max(1).mean()),
                    "mean_margin_hai_minus_nearest_other": float((S[:, hi] - others.max(1)).mean()),
                    "frac_hai_nearest": float((S.argmax(1) == hi).mean()),
                    "mean_cos_all_others": float(others.mean()),
                    "nearest_other_top3": [(k, round(n / len(Z), 3)) for k, n in Counter(onames[j] for j in near_other).most_common(3)],
                    "per_item_cos_hai": S[:, hi].tolist()}
        cs = {"library_hai_answers": centroid_stats(X[y == hi]), "library_all_answers_frac_own_nearest": float(((X @ Cm.T).argmax(1) == y).mean()),
              "library_non_hai_answers": centroid_stats(X[y != hi]), "cells": {}}
        for (v, c), rs in sorted(cells.items()):
            st = centroid_stats(cellX[(v, c)]); per = st.pop("per_item_cos_hai"); cs["cells"][f"{v}|{c}"] = st
            cs["cells"][f"{v}|{c}"]["cos_hai_ci"] = ci_mean(per)
        cs["library_hai_answers"].pop("per_item_cos_hai"); cs["library_non_hai_answers"].pop("per_item_cos_hai")
        pcs = centroid_stats(paraX); pcs.pop("per_item_cos_hai"); cs["paraphrase"] = pcs
        out["centroid"][name] = cs

    # ---------- (3) own-persona match ----------
    qids = [q["id"] for q in qs]; qi = {q: i for i, q in enumerate(qids)}
    libHAI = {}; libP = {}
    for v in libs:
        M = np.zeros((len(qids), 4096), np.float32); have = np.zeros(len(qids), bool)
        for r, x in zip(libs[v], libX[v]):
            if r["persona"] == HAI: M[qi[r["qid"]]] = x; have[qi[r["qid"]]] = True
        libHAI[v] = (M, have)
        PM = {p: np.zeros((len(qids), 4096), np.float32) for p in personas}
        for r, x in zip(libs[v], libX[v]): PM[r["persona"]][qi[r["qid"]]] = x
        libP[v] = PM
    own = {"per_cell": {}, "floors": {}, "diffs": {}, "same_question_nearest_persona": {}}
    cell_cos = {}
    for (v, c), rs in sorted(cells.items()):
        Z = cellX[(v, c)]; zq = np.array([qi[r["qid"]] for r in rs])
        for lv in libs:
            M, have = libHAI[lv]; ok = have[zq]
            same = (Z[ok] * M[zq[ok]]).sum(1)
            D = Z[ok] @ M[have].T   # rows: default answers; cols: library HAI answers for all questions
            colq = np.where(have)[0]; mask = colq[None, :] != zq[ok][:, None]
            cross = (D * mask).sum(1) / mask.sum(1)
            # one random other question per default answer (seeded)
            rnd = np.array([D[i, RNG.choice(np.where(mask[i])[0])] for i in range(len(D))])
            # same-question nearest persona
            S = np.stack([(Z[ok] * libP[lv][p][zq[ok]]).sum(1) for p in personas], 1)
            hi_ = personas.index(HAI); others = np.delete(S, hi_, axis=1); onames = [p for p in personas if p != HAI]
            key = f"{v}|{c}|lib{lv.replace('.', '')}"
            cell_cos[key] = dict(zip([rs[i]["qid"] for i in np.where(ok)[0]], same.tolist()))
            own["per_cell"][key] = {"version": v, "condition": c, "library": lv, "own": lv in OWN_LIB[v], "n": int(ok.sum()),
                                    "same_question_cos": ci_mean(same), "cross_question_mean_cos": ci_mean(cross), "cross_question_random_other_cos": ci_mean(rnd),
                                    "same_minus_cross": paired_diff(dict(enumerate(same)), dict(enumerate(cross))),
                                    "by_kind_same_question_cos": {k: float(same[[rs[i]["kind"] == k for i in np.where(ok)[0]]].mean()) for k in ("scenario", "generic")}}
            own["same_question_nearest_persona"][key] = {"frac_hai_nearest": float((S.argmax(1) == hi_).mean()),
                                                          "mean_margin_hai_minus_nearest_other": float((S[:, hi_] - others.max(1)).mean()),
                                                          "mean_cos_nearest_other": float(others.max(1).mean()),
                                                          "nearest_persona_top5": [(k, round(n / len(S), 3)) for k, n in Counter(personas[j] for j in S.argmax(1)).most_common(5)]}
    # paraphrase floor: paraphrase vs its source (3.5|orig)
    srcX = {r["qid"]: x for r, x in zip(cells[("3.5", "orig")], cellX[("3.5", "orig")])}
    pc = np.array([float(x @ srcX[r["qid"]]) for r, x in zip(para.values(), paraX) if r["qid"] in srcX])
    own["floors"]["paraphrase_vs_source"] = ci_mean(pc)
    # paraphrase of 3.5|orig vs 3.5 library HAI same question (what a reworded 3.5 answer scores on the own-persona measure)
    M, have = libHAI["3.5"]
    pp = np.array([float(x @ M[qi[r["qid"]]]) for r, x in zip(para.values(), paraX) if have[qi[r["qid"]]]])
    own["floors"]["paraphrase_of_35orig_vs_lib35_hai_same_q"] = ci_mean(pp)
    # within-library references: HAI vs other personas same question; 3.5 HAI vs 3.8 HAI same question
    for lv in libs:
        M, have = libHAI[lv]
        for p in ("textbook", "corporate_faq", "librarian"):
            own["floors"][f"lib{lv}_hai_vs_{p}_same_q"] = ci_mean((M[have] * libP[lv][p][have]).sum(1))
        allo = np.mean([(M[have] * libP[lv][p][have]).sum(1) for p in personas if p != HAI], 0)
        own["floors"][f"lib{lv}_hai_vs_mean_other_persona_same_q"] = ci_mean(allo)
        Dh = M[have] @ M[have].T; mask = ~np.eye(len(Dh), dtype=bool)
        own["floors"][f"lib{lv}_hai_vs_hai_other_question"] = ci_mean((Dh * mask).sum(1) / mask.sum(1))
    M5, h5 = libHAI["3.5"]; M8, h8 = libHAI["3.8"]; hb = h5 & h8
    own["floors"]["lib35_hai_vs_lib38_hai_same_q"] = ci_mean((M5[hb] * M8[hb]).sum(1))
    # differences (question-paired): own-library 3.8 minus own-library 3.5, per condition; and condition effects
    for c in CONDS:
        k8, k5 = f"3.8|{c}|lib38", f"3.5|{c}|lib35"
        if k8 in cell_cos and k5 in cell_cos:
            own["diffs"][f"own: 3.8|{c} minus 3.5|{c}"] = paired_diff(cell_cos[k8], cell_cos[k5])
        for lv in libs:
            k8, k5, k6 = f"3.8|{c}|lib{lv.replace('.', '')}", f"3.5|{c}|lib{lv.replace('.', '')}", f"3.6|{c}|lib{lv.replace('.', '')}"
            if k8 in cell_cos and k5 in cell_cos: own["diffs"][f"lib{lv}: 3.8|{c} minus 3.5|{c}"] = paired_diff(cell_cos[k8], cell_cos[k5])
            if k6 in cell_cos and k5 in cell_cos: own["diffs"][f"lib{lv}: 3.6|{c} minus 3.5|{c}"] = paired_diff(cell_cos[k6], cell_cos[k5])
    for v in VERS:
        for lv in OWN_LIB[v]:
            for c1, c2 in (("orig", "nosys"), ("nosys", "minsys"), ("orig", "minsys")):
                k1, k2 = f"{v}|{c1}|lib{lv.replace('.', '')}", f"{v}|{c2}|lib{lv.replace('.', '')}"
                if k1 in cell_cos and k2 in cell_cos: own["diffs"][f"{v} lib{lv}: {c2} minus {c1}"] = paired_diff(cell_cos[k2], cell_cos[k1])
    out["own_persona"] = own

    # ---------- verdict ----------
    out["verdict"] = verdict(out)
    bt = Counter()
    for l in config.SPEND_LOG.read_text().splitlines():
        if l.strip() and '"T11' in l:
            d = json.loads(l); bt[d["task"]] += d.get("cost", 0)
    out["spend"] = {"T11_total_usd": t11_spend(), "T11_fix_usd": sum(v for k, v in bt.items() if k.startswith("T11_fix")),
                    "harness_total_usd": total_spend(), "by_task": {k: round(v, 3) for k, v in bt.items()}}
    out["elapsed_s"] = time.time() - t0
    JSON_PATH.write_text(json.dumps(out, indent=1))
    write_readme(out)
    print(f"wrote {JSON_PATH.name} and {README_PATH.name}; {time.time() - t0:.0f}s")

def verdict(o):
    """Rule: drift survives only if, with thinking off and the minimal system prompt (minsys), under the semantic classifier,
    3.8's defaults fit the libraries worse than 3.5's by more than the paraphrase floor (A) AND 3.8's defaults are not closer
    to their own helpful-assistant persona than 3.5's are to theirs (B)."""
    f = o["fits"]; v = {"rule": " ".join(verdict.__doc__.split()), "A": {}, "B": {}}
    # A, pooled instrument: 3.5|minsys minus 3.8|minsys max-conf vs paraphrase floor (source minus paraphrase max-conf)
    for inst in ("pooled", "own"):
        if inst == "pooled":
            gap = -f["pooled"]["cross_version"]["3.8-3.5|minsys"]["max_conf"]["diff"]; lo = -f["pooled"]["cross_version"]["3.8-3.5|minsys"]["max_conf"]["hi"]; hi = -f["pooled"]["cross_version"]["3.8-3.5|minsys"]["max_conf"]["lo"]
            floor = f["pooled"]["paraphrase"]["source_minus_paraphrase"]["max_conf"]["diff"]
            c35, c38 = f["pooled"]["cells"]["3.5|minsys"]["mean_max_conf"], f["pooled"]["cells"]["3.8|minsys"]["mean_max_conf"]
        else:
            c35, c38 = f["lib35"]["cells"]["3.5|minsys"]["mean_max_conf"], f["lib38"]["cells"]["3.8|minsys"]["mean_max_conf"]
            gap = c35 - c38; lo = hi = float("nan")
            floor = max(f["lib35"]["paraphrase"]["source_minus_paraphrase"]["max_conf"]["diff"], f["lib38"]["paraphrase"]["source_minus_paraphrase"]["max_conf"]["diff"])
        v["A"][inst] = {"conf_35_minsys": c35, "conf_38_minsys": c38, "gap_35_minus_38": gap, "gap_ci": [lo, hi], "paraphrase_floor": floor, "holds": bool(gap > floor)}
    v["A"]["holds"] = bool(v["A"]["pooled"]["holds"] and v["A"]["own"]["holds"])
    # informational: the same comparison inside each single-library instrument, and the own-library fit normalised by that
    # instrument's held-out ceiling (the two libraries are different instruments with different ceilings)
    for name in ("lib35", "lib38"):
        x = f[name]["cross_version"]["3.8-3.5|minsys"]["max_conf"]
        v["A"][f"within_{name}"] = {"conf_35_minsys": f[name]["cells"]["3.5|minsys"]["mean_max_conf"], "conf_38_minsys": f[name]["cells"]["3.8|minsys"]["mean_max_conf"],
                                    "gap_35_minus_38": -x["diff"], "gap_ci": [-x["hi"], -x["lo"]],
                                    "paraphrase_floor": f[name]["paraphrase"]["source_minus_paraphrase"]["max_conf"]["diff"], "holds": bool(-x["diff"] > f[name]["paraphrase"]["source_minus_paraphrase"]["max_conf"]["diff"])}
    c35 = f["lib35"]["cells"]["3.5|minsys"]["mean_max_conf"] / f["lib35"]["heldout"]["summary"]["mean_max_conf"]
    c38 = f["lib38"]["cells"]["3.8|minsys"]["mean_max_conf"] / f["lib38"]["heldout"]["summary"]["mean_max_conf"]
    v["A"]["own_normalised_by_heldout_ceiling"] = {"conf_35_minsys": c35, "conf_38_minsys": c38, "gap_35_minus_38": c35 - c38, "gap_ci": [float("nan")] * 2,
                                                   "paraphrase_floor": v["A"]["own"]["paraphrase_floor"], "holds": bool(c35 - c38 > v["A"]["own"]["paraphrase_floor"])}
    op = o["own_persona"]["per_cell"]
    c35 = op["3.5|minsys|lib35"]["same_question_cos"]["mean"]; c38 = op["3.8|minsys|lib38"]["same_question_cos"]["mean"]
    d = o["own_persona"]["diffs"]["own: 3.8|minsys minus 3.5|minsys"]
    v["B"] = {"own_cos_35_minsys": c35, "own_cos_38_minsys": c38, "diff_38_minus_35": d, "holds": bool(c38 <= c35)}
    v["drift_survives"] = bool(v["A"]["holds"] and v["B"]["holds"])
    # attribution: 3.8-3.5 max-conf gap under pooled instrument, by condition
    v["gap_by_condition_pooled"] = {c: f["pooled"]["cross_version"][f"3.8-3.5|{c}"]["max_conf"]["diff"] for c in CONDS if f"3.8-3.5|{c}" in f["pooled"]["cross_version"]}
    v["own_cos_by_condition"] = {c: {"3.5": op[f"3.5|{c}|lib35"]["same_question_cos"]["mean"], "3.8": op[f"3.8|{c}|lib38"]["same_question_cos"]["mean"]} for c in CONDS if f"3.5|{c}|lib35" in op and f"3.8|{c}|lib38" in op}
    return v

# ---------- README ----------
def fmt(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
def ci(d, k="diff", sign=True):
    s = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return f"{s(d[k])} [{s(d['lo'])}, {s(d['hi'])}]"
def top(t, n=4): return ", ".join(f"{k} {v:.2f}" for k, v in t[:n])

def write_readme(o):
    f = o["fits"]; L = []; w = L.append
    w("# T11 confound fixes: thinking off, minimal system prompt, semantic classifier, own-persona cosine\n")
    w("**Status: exploratory follow-up to results/T11/README.md (stage 1) and README_selfmatch.md. Removes three confounds from the "
      "'3.8 defaults fit persona libraries badly' result: (a) no system prompt vs persona system prompts, (b) thinking on vs off, "
      "(c) TF-IDF + stylometry classifier rewarding surface style. Not pre-registered; the verdict rule was fixed by the caller before the run and is applied as stated.**\n")
    w("## Verdict rule (stated first)\n")
    w("The drift claim survives only if, with thinking off and the minimal system prompt (condition `minsys`), under the semantic classifier, "
      "(A) 3.8's defaults still fit the libraries worse than 3.5's by more than the paraphrase floor, AND (B) 3.8's defaults are not closer to its own "
      "helpful-assistant persona than 3.5's are to its own. Otherwise the result is attributed to whichever confound closed the gap. "
      "Operationalised: (A) mean max-confidence of 3.5|minsys minus 3.8|minsys, under the pooled-library instrument AND under each model's own library "
      "(3.5 under the 3.5 library, 3.8 under the 3.8 library), must exceed that instrument's paraphrase floor (source-minus-paraphrase max-conf shift); "
      "(B) mean same-question cosine of 3.8|minsys to the 3.8 library's helpful_ai_assistant answer must be <= that of 3.5|minsys to the 3.5 library's.\n")
    w("## Design\n")
    for c, d in COND_DESC.items(): w(f"- `{c}`: {d}.")
    w(f"- Libraries: 3.5 (reference.jsonl, n = {f['lib35']['n_ref']}), 3.8 (reference_3.8.jsonl, n = {f['lib38']['n_ref']}); 30 persona prompts x 200 questions, thinking OFF, max_tokens 350.")
    w(f"- Embeddings: {o['settings']['embedding_model']}; {o['settings']['n_zero_embeddings']} empty/failed embeddings. Classifier: {o['settings']['classifier']}. "
      "Confidence floor per fit = p10 of held-out max-conf. Paraphrase floor: 3.5|orig answers paraphrased by claude-sonnet-5.5 (n = %d), scored under the same instrument." % o["settings"]["n_paraphrase"])
    w("- Default cells (chat frame only): " + ", ".join(f"{k} n={v}" for k, v in o["settings"]["n_default_cells"].items()) + ".\n")
    # 1. generation
    w("## 1. Regenerated defaults (thinking OFF, max_tokens 350, chat frame)\n")
    w("| version | condition | n | mean words | 350-token truncation rate |"); w("|---|---|---|---|---|")
    for k, s in f["pooled"]["cells"].items():
        v, c = k.split("|"); w(f"| {v} | {c} | {s['n']} | {s['mean_words']:.0f} | {s['truncation_rate']:.2f} |")
    w("")
    # 2. semantic classifier
    w("## 2. Semantic classifier (logistic regression on embeddings)\n")
    w("### Held-out accuracy (question-grouped 5-fold CV)\n")
    w("| library | n | held-out acc | scenarios | generic | mean max-conf | floor | C | lowest-accuracy personas | top confusions |"); w("|---|---|---|---|---|---|---|---|---|---|")
    for name in LIB_NAMES:
        h = f[name]["heldout"]; pp = sorted(h["per_persona"].items(), key=lambda kv: kv[1])[:4]
        w(f"| {name} | {f[name]['n_ref']} | **{h['accuracy']:.3f}** | {h['per_kind']['scenario']:.3f} | {h['per_kind']['generic']:.3f} | {h['summary']['mean_max_conf']:.3f} | "
          f"{f[name]['confidence_floor']:.3f} | {f[name]['C_best']} | {', '.join(f'{p} {a:.2f}' for p, a in pp)} | {'; '.join(f'{a}->{b} ({n})' for a, b, n in h['top_confusions'][:4])} |")
    if "by_version" in f["pooled"]["heldout"]:
        w("\nPooled fit, held-out by writing model: " + "; ".join(f"{v}: acc {d['accuracy']:.3f}, max-conf {d['mean_max_conf']:.3f}, below-floor {d['frac_below_floor']:.2f}" for v, d in f["pooled"]["heldout"]["by_version"].items()))
    for name in ("lib35", "lib38"):
        x = f[name].get("cross_library")
        if x: w(f"\nCross-library transfer, {name} classifier scoring the whole {x['scored_version']} library (known labels): accuracy {x['accuracy']:.3f}, mean max-conf {x['mean_max_conf']:.3f}, below-floor {x['frac_below_floor']:.2f}; helpful_ai_assistant accuracy {x['hai_accuracy']:.2f} / max-conf {x['hai_mean_max_conf']:.3f}.")
    w("\nFor comparison, the TF-IDF + stylometry instrument (README_selfmatch.md) had held-out accuracy 0.966 (3.5 library), 0.895 (3.8 library), 0.934 (pooled).\n")
    for name in LIB_NAMES:
        w(f"### Default answers under the {name} classifier\n")
        w("| version | condition | n | mean max-conf | median | mean entropy (bits) | frac below floor | control share | mean P(helpful_ai) | winner (top 4) |"); w("|---|---|---|---|---|---|---|---|---|---|")
        for k, s in f[name]["cells"].items():
            v, c = k.split("|")
            w(f"| {v} | {c} | {s['n']} | {s['mean_max_conf']:.3f} | {s['median_max_conf']:.3f} | {s['mean_entropy_bits']:.2f} | {s['frac_below_floor']:.2f} | {s['control_share']:.2f} | {s['hai_prob_mean']:.2f} | {top(s['top5'])} |")
        p = f[name]["paraphrase"]
        w(f"| paraphrase of 3.5\\|orig | - | {p['n']} | {p['mean_max_conf']:.3f} | {p['median_max_conf']:.3f} | {p['mean_entropy_bits']:.2f} | {p['frac_below_floor']:.2f} | {p['control_share']:.2f} | {fmt(p.get('hai_prob_mean'), 2) if 'hai_prob_mean' in p else '-'} | {top(p['top5'])} |")
        w(f"\nParaphrase floor ({name}): winner agreement with source {p['winner_agreement_with_source']:.2f}; source minus paraphrase: max-conf {ci(p['source_minus_paraphrase']['max_conf'])}, "
          f"entropy {ci(p['source_minus_paraphrase']['entropy'])}, below-floor {ci(p['source_minus_paraphrase']['below'])}.")
        w("\nBy question kind (mean max-conf / entropy / below-floor):\n")
        w("| version | condition | scenarios | generic |"); w("|---|---|---|---|")
        for k, s in f[name]["cells"].items():
            v, c = k.split("|"); a, b = s["by_kind"]["scenario"], s["by_kind"]["generic"]
            w(f"| {v} | {c} | {a['mean_max_conf']:.3f} / {a['mean_entropy_bits']:.2f} / {a['frac_below_floor']:.2f} | {b['mean_max_conf']:.3f} / {b['mean_entropy_bits']:.2f} / {b['frac_below_floor']:.2f} |")
        w("\nCross-version differences (question-level bootstrap 95% CI, 2000 resamples):\n")
        w("| condition | 3.8-3.5 max-conf | 3.8-3.5 entropy | 3.8-3.5 below-floor | 3.6-3.5 max-conf | 3.8-3.6 max-conf |"); w("|---|---|---|---|---|---|")
        for c in CONDS:
            x = f[name]["cross_version"]
            if f"3.8-3.5|{c}" in x:
                w(f"| {c} | {ci(x[f'3.8-3.5|{c}']['max_conf'])} | {ci(x[f'3.8-3.5|{c}']['entropy'])} | {ci(x[f'3.8-3.5|{c}']['below'])} | {ci(x[f'3.6-3.5|{c}']['max_conf'])} | {ci(x[f'3.8-3.6|{c}']['max_conf'])} |")
        w("\nWithin-version condition effects on max-conf (later minus earlier; winner agreement in brackets):\n")
        w("| version | nosys - orig | minsys - nosys | minsys - orig | orig_trunc - orig |"); w("|---|---|---|---|---|")
        for v in VERS:
            x = f[name]["cross_condition"]; cells_ = []
            for c2, c1 in (("nosys", "orig"), ("minsys", "nosys"), ("minsys", "orig"), ("orig_trunc", "orig")):
                k = f"{v}|{c2}-{c1}"; cells_.append(f"{ci(x[k]['max_conf'])} ({x[k]['winner_agreement']:.2f})" if k in x else "-")
            w(f"| {v} | " + " | ".join(cells_) + " |")
        w("")
    # centroid
    w("### Nearest-centroid cosine (per library; centroid = mean embedding of a persona's 200 answers)\n")
    for name in LIB_NAMES:
        cs = o["centroid"][name]
        w(f"**{name}**: library helpful_ai_assistant answers: cos to own centroid {cs['library_hai_answers']['mean_cos_hai']:.3f}, to nearest other centroid {cs['library_hai_answers']['mean_cos_nearest_other']:.3f} "
          f"(margin {cs['library_hai_answers']['mean_margin_hai_minus_nearest_other']:+.3f}, own-centroid nearest for {cs['library_hai_answers']['frac_hai_nearest']:.2f}); "
          f"library non-HAI answers: cos to HAI centroid {cs['library_non_hai_answers']['mean_cos_hai']:.3f}; all library answers nearest to their own persona centroid: {cs['library_all_answers_frac_own_nearest']:.2f}.\n")
        w("| version | condition | n | mean cos to helpful_ai centroid [95% CI] | mean cos to nearest other centroid | margin | frac helpful_ai nearest | nearest other (top 3) |"); w("|---|---|---|---|---|---|---|---|")
        for k, s in cs["cells"].items():
            v, c = k.split("|")
            w(f"| {v} | {c} | {s['n']} | {s['mean_cos_hai']:.3f} [{s['cos_hai_ci']['lo']:.3f}, {s['cos_hai_ci']['hi']:.3f}] | {s['mean_cos_nearest_other']:.3f} | {s['mean_margin_hai_minus_nearest_other']:+.3f} | {s['frac_hai_nearest']:.2f} | {top(s['nearest_other_top3'], 3)} |")
        p = cs["paraphrase"]
        w(f"| paraphrase of 3.5\\|orig | - | {p['n']} | {p['mean_cos_hai']:.3f} | {p['mean_cos_nearest_other']:.3f} | {p['mean_margin_hai_minus_nearest_other']:+.3f} | {p['frac_hai_nearest']:.2f} | {top(p['nearest_other_top3'], 3)} |")
        w("")
    # 3. own persona
    op = o["own_persona"]
    w("## 3. Own-persona match: cosine to the library's helpful_ai_assistant answer to the SAME question\n")
    w("Own library: 3.5 -> 3.5 library; 3.8 -> 3.8 library; 3.6 has no own library and is scored against both (both rows shown, neither is 'own'). "
      "Cross-question floor = mean cosine to the library's helpful_ai_assistant answers to all OTHER questions (and to one random other question). "
      "Paraphrase floor = cosine between each paraphrase and its source (3.5|orig).\n")
    w("| version | condition | library | own | n | same-question cos [95% CI] | cross-question mean cos [95% CI] | random other q | same - cross | scenarios / generic | same-q nearest persona = helpful_ai | nearest persona (top 3) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in op["per_cell"].items():
        sq = op["same_question_nearest_persona"][k]
        w(f"| {s['version']} | {s['condition']} | {s['library']} | {'**yes**' if s['own'] else 'no'} | {s['n']} | {s['same_question_cos']['mean']:.3f} [{s['same_question_cos']['lo']:.3f}, {s['same_question_cos']['hi']:.3f}] | "
          f"{s['cross_question_mean_cos']['mean']:.3f} [{s['cross_question_mean_cos']['lo']:.3f}, {s['cross_question_mean_cos']['hi']:.3f}] | {s['cross_question_random_other_cos']['mean']:.3f} | {ci(s['same_minus_cross'])} | "
          f"{s['by_kind_same_question_cos']['scenario']:.3f} / {s['by_kind_same_question_cos']['generic']:.3f} | {sq['frac_hai_nearest']:.2f} (margin {sq['mean_margin_hai_minus_nearest_other']:+.3f}) | {top(sq['nearest_persona_top5'], 3)} |")
    w("\nFloors and references (mean cos [95% CI]):\n")
    w("| reference | mean cos | 95% CI | n |"); w("|---|---|---|---|")
    for k, s in op["floors"].items(): w(f"| {k} | {s['mean']:.3f} | [{s['lo']:.3f}, {s['hi']:.3f}] | {s['n']} |")
    w("\nQuestion-paired differences (bootstrap 95% CI):\n")
    w("| comparison | diff | 95% CI | n |"); w("|---|---|---|---|")
    for k, s in op["diffs"].items(): w(f"| {k} | {s['diff']:+.3f} | [{s['lo']:+.3f}, {s['hi']:+.3f}] | {s['n']} |")
    w("")
    # tfidf on new conditions, if present
    tp = OUT / "fit_fix_tfidf.json"
    if tp.exists():
        t = json.load(open(tp))
        w("## 2b. Old instrument (TF-IDF + stylometry, README_selfmatch.md classifiers) re-scored on the new conditions\n")
        w("Refit of the stage-1 / self-match classifiers (same features, C = 16, no CV; floors and word caps taken from fit_selfmatch.json) applied to the thinking-off conditions. "
          "`orig_trunc` here should reproduce the README_selfmatch 'chat, trunc' rows.\n")
        for name in LIB_NAMES:
            if name not in t: continue
            w(f"**{name}** (floor {t[name]['confidence_floor']:.3f}):\n")
            w("| version | condition | n | mean max-conf | mean entropy (bits) | frac below floor | control share | winner (top 4) |"); w("|---|---|---|---|---|---|---|---|")
            for k, s in t[name]["cells"].items():
                v, c = k.split("|"); w(f"| {v} | {c} | {s['n']} | {s['mean_max_conf']:.3f} | {s['mean_entropy_bits']:.2f} | {s['frac_below_floor']:.2f} | {s['control_share']:.2f} | {top(s['top5'])} |")
            p = t[name]["paraphrase"]
            w(f"\nParaphrase floor: source minus paraphrase max-conf {ci(p['source_minus_paraphrase']['max_conf'])}. Cross-version 3.8-3.5 max-conf: " +
              "; ".join(f"{c} {ci(t[name]['cross_version'][f'3.8-3.5|{c}']['max_conf'])}" for c in CONDS if f"3.8-3.5|{c}" in t[name]["cross_version"]) + ".\n")
    # 4. verdict
    v = o["verdict"]
    w("## 4. Verdict\n")
    w("**Rule (restated):** " + v["rule"].replace("Rule: ", "", 1) + "\n")
    w("| test | instrument | 3.5\\|minsys | 3.8\\|minsys | gap (3.5 - 3.8) | paraphrase floor | holds |"); w("|---|---|---|---|---|---|---|")
    labels = {"pooled": "pooled library (single instrument; primary)", "own": "own library (3.5 under lib35 vs 3.8 under lib38; two instruments)",
              "own_normalised_by_heldout_ceiling": "own library, each divided by its instrument's held-out mean max-conf (informational)",
              "within_lib35": "within lib35 (3.5\\|minsys vs 3.8\\|minsys; informational)", "within_lib38": "within lib38 (3.5\\|minsys vs 3.8\\|minsys; informational)"}
    for inst in ("pooled", "own", "own_normalised_by_heldout_ceiling", "within_lib35", "within_lib38"):
        a = v["A"][inst]; g = f"{a['gap_35_minus_38']:+.3f}" + (f" [{a['gap_ci'][0]:+.3f}, {a['gap_ci'][1]:+.3f}]" if not np.isnan(a["gap_ci"][0]) else "")
        w(f"| A: 3.8 fits worse than 3.5 by more than the paraphrase floor | {labels[inst]} | {a['conf_35_minsys']:.3f} | {a['conf_38_minsys']:.3f} | {g} | {a['paraphrase_floor']:+.3f} | {'yes' if a['holds'] else 'no'} |")
    b = v["B"]
    w(f"| B: 3.8 not closer to its own helpful-assistant persona than 3.5 to its own | own-library same-question cosine | {b['own_cos_35_minsys']:.3f} | {b['own_cos_38_minsys']:.3f} | 3.8 - 3.5: {ci(b['diff_38_minus_35'])} | - | {'yes' if b['holds'] else 'no'} |")
    w(f"\nA holds: **{v['A']['holds']}** (pooled {v['A']['pooled']['holds']}; own-library {v['A']['own']['holds']}, but see the normalised and within-instrument rows: the raw own-library row compares two instruments whose held-out ceilings differ, "
      f"{f['lib35']['heldout']['summary']['mean_max_conf']:.3f} vs {f['lib38']['heldout']['summary']['mean_max_conf']:.3f}, and inside either single instrument 3.8\\|minsys fits at least as well as 3.5\\|minsys). B holds: **{v['B']['holds']}**.\n")
    w(f"**{'DRIFT SURVIVES under the stated rule' if v['drift_survives'] else 'DRIFT DOES NOT SURVIVE under the stated rule'}.**\n")
    w("3.8-3.5 max-conf gap under the pooled semantic instrument, by condition: " + "; ".join(f"{c} {g:+.3f}" for c, g in v["gap_by_condition_pooled"].items()) +
      ". Own-persona cosine by condition (3.5 / 3.8): " + "; ".join(f"{c} {d['3.5']:.3f} / {d['3.8']:.3f}" for c, d in v["own_cos_by_condition"].items()) + ".\n")
    w(reading(o))
    w("## Spend\n")
    s = o["spend"]; w(f"T11_fix spend ${s['T11_fix_usd']:.3f} ({ {k: v for k, v in s['by_task'].items() if k.startswith('T11_fix')} }); T11 total ${s['T11_total_usd']:.2f}; harness total ${s['harness_total_usd']:.2f}. "
                     f"Embedding calls (Fireworks) are not in the spend log. Fit time {o['elapsed_s']:.0f}s.\n")
    README_PATH.write_text("\n".join(L))

def reading(o):
    """Plain-language attribution written from the numbers (the verdict table carries the figures)."""
    f = o["fits"]; v = o["verdict"]; op = o["own_persona"]; L = []
    g = v["gap_by_condition_pooled"]; oc = v["own_cos_by_condition"]
    tf = json.load(open(OUT / "fit_fix_tfidf.json")) if (OUT / "fit_fix_tfidf.json").exists() else None
    tfg = {c: tf["pooled"]["cross_version"][f"3.8-3.5|{c}"]["max_conf"]["diff"] for c in ("orig_trunc", "nosys", "minsys")} if tf else {}
    cc = f["pooled"]["cross_condition"]
    L.append("### Which confound explained it (reading written after the fit; the tables above are authoritative)\n")
    L.append(f"- **(b) thinking ON is the confound, and it explains the whole 3.8 drop.** Switching the defaults to thinking OFF at the library's length cap (`nosys`, no other change) "
             f"moves the pooled 3.8-3.5 max-conf gap from {g['orig']:+.3f} (semantic, full text; {g['orig_trunc']:+.3f} truncated) to {g['nosys']:+.3f}"
             + (f", and under the old TF-IDF instrument from {tfg.get('orig_trunc', float('nan')):+.3f} to {tfg.get('nosys', float('nan')):+.3f}" if tfg else "")
             + ". Within 3.8, nosys minus orig is " + f"{cc['3.8|nosys-orig']['max_conf']['diff']:+.3f} [{cc['3.8|nosys-orig']['max_conf']['lo']:+.3f}, {cc['3.8|nosys-orig']['max_conf']['hi']:+.3f}] "
             f"against {cc['3.5|nosys-orig']['max_conf']['diff']:+.3f} for 3.5 and {cc['3.6|nosys-orig']['max_conf']['diff']:+.3f} for 3.6: reasoning mode changes 3.8's final-answer style far more than it changes 3.5's. "
             "With thinking off, 3.8's defaults fit every library (3.5's, its own, pooled) at least as well as 3.5's do, under both instruments.")
    L.append(f"- **(a) the system prompt changes nothing.** `minsys` minus `nosys` max-conf: 3.5 {cc['3.5|minsys-nosys']['max_conf']['diff']:+.3f}, 3.6 {cc['3.6|minsys-nosys']['max_conf']['diff']:+.3f}, 3.8 {cc['3.8|minsys-nosys']['max_conf']['diff']:+.3f} (all CIs span 0); "
             f"the 3.8-3.5 gap goes from {g['nosys']:+.3f} to {g['minsys']:+.3f}. The 'tool agent with no system prompt' explanation is not supported: a bare user message and 'You are a helpful assistant.' produce the same persona fit.")
    L.append(f"- **(c) the classifier was not the explanation on its own.** On the original thinking-on answers the semantic instrument still shows a 3.8-3.5 gap of {g['orig_trunc']:+.3f} [{f['pooled']['cross_version']['3.8-3.5|orig_trunc']['max_conf']['lo']:+.3f}, {f['pooled']['cross_version']['3.8-3.5|orig_trunc']['max_conf']['hi']:+.3f}] "
             f"(TF-IDF: {tfg.get('orig_trunc', float('nan')):+.3f}), about 2.5x the semantic paraphrase floor ({f['pooled']['paraphrase']['source_minus_paraphrase']['max_conf']['diff']:+.3f}). The surface-style instrument exaggerated the drop roughly 2x but did not invent it; what it was seeing was thinking-mode output.")
    L.append(f"- **Own-persona cosine says the same.** Same-question cosine to the own library's helpful_ai_assistant answer: 3.5|minsys {oc['minsys']['3.5']:.3f}, 3.8|minsys {oc['minsys']['3.8']:.3f} "
             f"(3.8 - 3.5 = {op['diffs']['own: 3.8|minsys minus 3.5|minsys']['diff']:+.3f} [{op['diffs']['own: 3.8|minsys minus 3.5|minsys']['lo']:+.3f}, {op['diffs']['own: 3.8|minsys minus 3.5|minsys']['hi']:+.3f}]), versus "
             f"{oc['orig']['3.5']:.3f} / {oc['orig']['3.8']:.3f} for the thinking-on originals (3.8 - 3.5 = {op['diffs']['own: 3.8|orig minus 3.5|orig']['diff']:+.3f}). For scale: a paraphrase sits {op['floors']['paraphrase_vs_source']['mean']:.3f} from its source, "
             f"3.5's and 3.8's helpful-assistant answers to the same question sit {op['floors']['lib35_hai_vs_lib38_hai_same_q']['mean']:.3f} from each other, the library's helpful-assistant vs textbook same-question cosine is {op['floors']['lib3.5_hai_vs_textbook_same_q']['mean']:.3f}, "
             f"and the cross-question floor is {op['per_cell']['3.5|minsys|lib35']['cross_question_mean_cos']['mean']:.3f}. The residual 3.8-3.5 own-persona difference at minsys ({op['diffs']['own: 3.8|minsys minus 3.5|minsys']['diff']:+.3f}) is smaller than the self-match effect itself "
             f"(3.8|minsys is {op['per_cell']['3.8|minsys|lib38']['same_question_cos']['mean'] - op['per_cell']['3.8|minsys|lib35']['same_question_cos']['mean']:+.3f} closer to its own library's answer than to 3.5's; 3.5|minsys is {op['per_cell']['3.5|minsys|lib35']['same_question_cos']['mean'] - op['per_cell']['3.5|minsys|lib38']['same_question_cos']['mean']:+.3f} closer to its own), "
             "so B holds only in the formal sense (3.8 is not closer than 3.5 to its own persona), with a difference an order of magnitude below the persona-level distances.")
    L.append(f"- **Winner is helpful_ai_assistant everywhere** ({f['pooled']['cells']['3.8|minsys']['top5'][0][1]:.2f} of 3.8|minsys answers under the pooled semantic classifier, {f['pooled']['cells']['3.5|minsys']['top5'][0][1]:.2f} of 3.5|minsys), as in stage 1. "
             "No version re-selects a different persona under any condition or instrument; the stage-1 'dispersion' of 3.8 was dispersion of its thinking-mode output only.")
    L.append("- **The nearest-centroid analysis is uninformative and is reported for completeness only.** Persona centroids over 200 questions are topic averages; even the library's own helpful_ai_assistant answers are nearer another centroid "
             f"({1 - o['centroid']['lib35']['library_hai_answers']['frac_hai_nearest']:.2f} of them under lib35) and all margins are within 0.02. The same-question comparisons in section 3 are the topic-controlled version of this analysis.")
    L.append(f"- The semantic instrument is weaker than the TF-IDF one (held-out {f['lib35']['heldout']['accuracy']:.3f} / {f['lib38']['heldout']['accuracy']:.3f} / {f['pooled']['heldout']['accuracy']:.3f} vs 0.966 / 0.895 / 0.934) and, like it, finds the 3.8 library less separable than the 3.5 library. "
             "That asymmetry is unchanged by any of the fixes and remains a property of 3.8's role-play, not of its defaults.")
    L.append("")
    L.append("### Verdict in one sentence\n")
    L.append("The stage-1 / self-match 'drift' signal was confound (b): the default answers were generated with thinking ON (and uncapped length) while every library was thinking OFF, and 3.8's thinking-on final answers differ in style from its thinking-off answers much more than 3.5's do. "
             "Length-matched, thinking-off defaults from 3.8 fit the persona libraries as well as 3.5's under both the TF-IDF and the semantic classifier, with or without a minimal system prompt, and sit as close to their own helpful-assistant persona as 3.5's do to theirs (within 0.01 cosine). "
             "What remains is a smaller, different claim: *3.8's thinking-mode output is stylistically further from any thinking-off role-play than 3.5's is*, which is a statement about reasoning mode, not about a default persona.\n")
    L.append("### Remaining caveats\n")
    L.append("1. `orig` vs `nosys` bundles thinking ON/OFF with max_tokens 4000 vs 350 (truncating the long answers to 293 words, `orig_trunc`, does not close the gap, so length alone is not it, but a thinking-on run at max_tokens 350 was not done).")
    L.append("2. The thinking-on defaults were run at reasoning effort medium, the only setting at which 3.8's chat template injects no system message (harness/config.py); 3.8's shipped default is xhigh. The thinking-mode style shift may differ at other efforts.")
    L.append("3. Embedding-based persona classification is partly topic-driven; grouped CV on unseen questions is the check (held-out accuracy 0.74-0.89, chance 0.033), and the absolute cosines in section 3 are dominated by shared topic (cross-question floor ~0.13). Only differences between cells answering the same questions are interpretable.")
    L.append("4. Each library and the pooled set is its own instrument with its own floor and ceiling; the own-library row in the verdict table compares two instruments and is reported normalised and within-instrument for that reason. The pooled single-instrument comparison is primary.")
    L.append("5. `minsys` uses 'You are a helpful assistant.', deliberately one word off the library's 'You are a helpful AI assistant.'; it is still not the library condition, and the paraphrase floor is computed from the thinking-on 3.5 originals, not from the new conditions.")
    L.append("6. The 3.8 library is less separable than the 3.5 library under both instruments (and 3.5's role-play scores better under the 3.8 classifier than 3.8's own does). 'Role-plays less stereotypically' is still true of 3.8 and is not addressed here.")
    L.append("7. 3.6 has no own library; its own-persona rows are cross-model by construction. Provider quantisation differs (3.5/3.6 fp8, 3.8 bf16). Single sample per cell; no pre-registration beyond the caller's rule fixed before the run.")
    L.append("8. The Fireworks embedding calls are not in data/spend.jsonl (about 4.7M tokens; negligible cost).")
    L.append("")
    return "\n".join(L)

if __name__ == "__main__":
    main()
