"""T11 stage 1 fit. Trains a persona classifier on the Qwen3.5 reference library (results/T11/reference.jsonl):
TF-IDF word 1-2 grams + char 3-5 grams + sentence-level stylometric rates, logistic regression, question-grouped 5-fold CV.
Reports held-out accuracy and top confusions; stops if accuracy < 0.6. Then classifies every default answer
(results/T11/default.jsonl) and the paraphrase floor (results/T11/paraphrase.jsonl), reports per version x frame
metrics, chat-vs-agentic gaps with question-level bootstrap CIs, and writes results/T11/fit.json + README.md.
Run: python3 -m tasks.T11.fit"""
import json, math, re, sys, time
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from harness import config
from tasks.T11.common import load_questions, load_personas, read_jsonl, OUT, t11_spend
from harness.client import total_spend

RNG = np.random.default_rng(0)
N_BOOT = 2000
ACC_STOP = 0.6
C_GRID = [1.0, 4.0, 16.0]

# ---------- stylometric features (rates only: no raw-length features, since default answers are much longer) ----------
FIRST = re.compile(r"\b(i|i'm|i've|i'll|i'd|me|my|mine|myself)\b", re.I)
SECOND = re.compile(r"\b(you|you're|you've|you'll|you'd|your|yours|yourself)\b", re.I)
WE = re.compile(r"\b(we|we're|we've|we'll|us|our|ours)\b", re.I)
HEDGE = re.compile(r"\b(maybe|perhaps|might|may|could|possibly|probably|likely|unlikely|seems?|appears?|somewhat|generally|"
                   r"typically|usually|often|sometimes|arguably|roughly|about|around|approximately|i think|i believe|i guess|"
                   r"it depends|not sure|unsure|tend to|in general)\b", re.I)
LIST_LINE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
HEAD_LINE = re.compile(r"^\s*#{1,6}\s+|^\s*\*\*[^*\n]{1,80}\*\*\s*:?\s*$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"[A-Za-z']+")

def stylo(text):
    lines = [l for l in text.split("\n") if l.strip()]
    sents = [s for s in SENT_SPLIT.split(text) if WORD.search(s)]
    words = WORD.findall(text)
    nw = max(1, len(words)); ns = max(1, len(sents)); nl = max(1, len(lines))
    slen = np.array([len(WORD.findall(s)) for s in sents] or [0.0], float)
    return [
        float(slen.mean()), float(slen.std()), float(np.median(slen)),           # sentence length
        float((slen <= 4).mean()),                                                 # fraction very short sentences
        sum(bool(LIST_LINE.match(l)) for l in lines) / nl,                         # list-line density
        sum(bool(HEAD_LINE.match(l)) for l in lines) / nl,                         # heading-line density
        (text.count("**") / 2 + text.count("`") / 2 + text.count("#")) / nw * 100, # markdown marks per 100 words
        len(FIRST.findall(text)) / nw * 100, len(SECOND.findall(text)) / nw * 100, len(WE.findall(text)) / nw * 100,
        len(HEDGE.findall(text)) / nw * 100,                                       # hedging per 100 words
        sum(s.strip().endswith("?") for s in sents) / ns, sum(s.strip().endswith("!") for s in sents) / ns,
        float(np.mean([len(w) for w in words]) if words else 0.0),                 # mean word length
        sum(w.isupper() and len(w) > 1 for w in words) / nw,                       # shouting
        len(re.findall(r"[—–:;()]", text)) / nw * 100,                             # punctuation density
        text.count("\n\n") / nl,                                                   # paragraph breaks per line
        len(set(w.lower() for w in words)) / nw,                                   # type-token ratio (length-sensitive; kept)
    ]
STYLO_NAMES = ["sent_len_mean", "sent_len_std", "sent_len_median", "frac_short_sent", "list_density", "heading_density",
               "markdown_per100w", "first_person_per100w", "second_person_per100w", "we_per100w", "hedge_per100w",
               "question_rate", "exclaim_rate", "mean_word_len", "caps_rate", "punct_per100w", "para_breaks", "ttr"]

class Featurizer:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000, sublinear_tf=True, lowercase=True)
        self.scaler = StandardScaler()
    def fit(self, texts):
        self.word.fit(texts); self.char.fit(texts); self.scaler.fit(np.array([stylo(t) for t in texts])); return self
    def transform(self, texts):
        S = self.scaler.transform(np.array([stylo(t) for t in texts]))
        S = np.clip(S, -5, 5) / math.sqrt(S.shape[1]) * 0.5   # keep the dense block comparable in norm to the l2 tf-idf blocks
        return sp.hstack([self.word.transform(texts), self.char.transform(texts), sp.csr_matrix(S)]).tocsr()

def entropy_bits(p):
    p = np.clip(p, 1e-12, 1); return float(-(p * np.log2(p)).sum())

def truncate_words(text, n):
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n])

def summarise(P, classes, floor, controls):
    if len(P) == 0:
        return {"n": 0, "mean_max_conf": float("nan"), "median_max_conf": float("nan"), "mean_entropy_bits": float("nan"),
                "frac_below_floor": float("nan"), "control_share": float("nan"), "top5": [], "win_counts": {}}
    maxc = P.max(1); ent = np.array([entropy_bits(p) for p in P]); win = [classes[i] for i in P.argmax(1)]
    c = Counter(win)
    return {"n": int(len(P)), "mean_max_conf": float(maxc.mean()), "median_max_conf": float(np.median(maxc)),
            "mean_entropy_bits": float(ent.mean()), "frac_below_floor": float((maxc < floor).mean()),
            "control_share": float(np.mean([w in controls for w in win])),
            "top5": [(k, round(v / len(win), 3)) for k, v in c.most_common(5)], "win_counts": dict(c)}

def per_item(P, classes, floor):
    return {"max_conf": P.max(1), "entropy": np.array([entropy_bits(p) for p in P]), "below": (P.max(1) < floor).astype(float),
            "win": [classes[i] for i in P.argmax(1)]}

def boot_gap(a, b, qa, qb, iters=N_BOOT):
    """Question-level paired bootstrap of mean(a) - mean(b), where a/b are per-item arrays keyed by question id lists qa/qb."""
    da = dict(zip(qa, a)); db = dict(zip(qb, b)); common = sorted(set(da) & set(db))
    if not common: return (float("nan"),) * 3
    x = np.array([da[q] - db[q] for q in common])
    obs = float(x.mean())
    bs = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(iters)]
    return (obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

def main():
    if "--readme-only" in sys.argv:
        write_readme(json.load(open(OUT / "fit.json"))); print("README regenerated from fit.json"); return
    t0 = time.time()
    qs = load_questions(); qkind = {q["id"]: q["kind"] for q in qs}
    per = load_personas(); personas = list(per["personas"]); controls = per["controls"]
    ref = [r for r in read_jsonl(OUT / "reference.jsonl") if not r.get("error") and r["content"].strip()]
    seen = {}
    for r in ref: seen[(r["persona"], r["qid"])] = r
    ref = list(seen.values())
    ref_trunc = {p: float(np.mean([r["truncated"] for r in ref if r["persona"] == p])) for p in personas}
    print(f"reference: {len(ref)} answers, {len(set(r['persona'] for r in ref))} personas, {len(set(r['qid'] for r in ref))} questions; "
          f"truncated at 350 tokens {np.mean([r['truncated'] for r in ref]):.3f}")
    texts = [r["content"] for r in ref]; y = np.array([personas.index(r["persona"]) for r in ref]); groups = np.array([r["qid"] for r in ref])
    ref_words = np.array([len(t.split()) for t in texts]); cap_words = int(np.percentile(ref_words, 99))
    print(f"reference word counts: mean {ref_words.mean():.0f}, p99 {cap_words} (default answers truncated to this for the primary analysis)")

    # ---------- grouped CV ----------
    gkf = GroupKFold(n_splits=5)
    cv = {}
    for C in C_GRID:
        oof = np.zeros((len(ref), len(personas)))
        for tr, te in gkf.split(texts, y, groups):
            fz = Featurizer().fit([texts[i] for i in tr])
            Xtr = fz.transform([texts[i] for i in tr]); Xte = fz.transform([texts[i] for i in te])
            clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(Xtr, y[tr])
            oof[np.ix_(te, clf.classes_)] = clf.predict_proba(Xte)
        acc = float((oof.argmax(1) == y).mean()); cv[C] = (acc, oof)
        print(f"  C={C}: held-out accuracy {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    C_best = max(cv, key=lambda c: cv[c][0]); acc, oof = cv[C_best]
    pred = oof.argmax(1)
    conf_mat = Counter((personas[a], personas[b]) for a, b in zip(y, pred) if a != b)
    top_conf = [(a, b, n) for (a, b), n in conf_mat.most_common(12)]
    per_persona_acc = {p: float((pred[y == i] == i).mean()) for i, p in enumerate(personas)}
    kind_acc = {k: float((pred[[qkind[g] == k for g in groups]] == y[[qkind[g] == k for g in groups]]).mean()) for k in ("scenario", "generic")}
    ref_heldout_conf = oof.max(1); floor = float(np.percentile(ref_heldout_conf, 10))
    ref_summary = summarise(oof, personas, floor, controls)
    print(f"best C={C_best}, held-out acc {acc:.3f}; chance {1/len(personas):.3f}; per-kind {kind_acc}; floor (p10 of held-out max-conf) {floor:.3f}")
    print("top confusions:", top_conf[:8])
    out = {"settings": {"ref_version": "3.5", "ref_model": config.MODELS["3.5"], "n_ref": len(ref), "n_personas": len(personas),
                        "controls": controls, "C_grid": {str(c): cv[c][0] for c in C_GRID}, "C_best": C_best, "cap_words": cap_words,
                        "features": "tfidf word 1-2 (min_df 3, 60k) + char_wb 3-5 (min_df 3, 120k) + 18 stylometric rates (standardised)",
                        "classifier": "multinomial logistic regression, question-grouped 5-fold CV", "confidence_floor": floor,
                        "ref_truncation_rate": float(np.mean([r["truncated"] for r in ref])), "ref_truncation_by_persona": ref_trunc},
           "heldout": {"accuracy": acc, "chance": 1 / len(personas), "per_kind": kind_acc, "per_persona": per_persona_acc,
                       "top_confusions": top_conf, "summary": ref_summary}}
    if acc < ACC_STOP:
        out["stop"] = f"held-out accuracy {acc:.3f} < {ACC_STOP}: reference library not separable enough; method uninformative"
        (OUT / "fit.json").write_text(json.dumps(out, indent=1))
        print(out["stop"]); return

    # ---------- final model on all reference ----------
    fz = Featurizer().fit(texts); X = fz.transform(texts)
    clf = LogisticRegression(C=C_best, max_iter=3000, n_jobs=-1).fit(X, y)
    classes = [personas[i] for i in clf.classes_]

    # ---------- default answers ----------
    dft = {}
    for r in read_jsonl(OUT / "default.jsonl"):
        if not r.get("error"): dft[(r["version"], r["frame"], r["qid"])] = r
    dft = list(dft.values())
    cells = defaultdict(list)
    for r in dft: cells[(r["version"], r["frame"])].append(r)
    cell_stats = {}; items = {}
    for (v, fr), rs in sorted(cells.items()):
        usable = [r for r in rs if r["content"].strip()]
        stats = {"n_total": len(rs), "n_empty_content": len(rs) - len(usable), "truncation_rate": float(np.mean([r["truncated"] for r in rs])),
                 "tool_call_rate": float(np.mean([r["tool_called"] for r in rs])), "unfinished_rate": float(np.mean([r["unfinished"] for r in rs])),
                 "mean_words": float(np.mean([len(r["content"].split()) for r in usable])),
                 "mean_reasoning_tokens": float(np.mean([r["reasoning_tokens"] for r in rs]))}
        for mode in ("trunc", "full"):
            tx = [truncate_words(r["content"], cap_words) if mode == "trunc" else r["content"] for r in usable]
            P = clf.predict_proba(fz.transform(tx))
            stats[mode] = summarise(P, classes, floor, controls)
            stats[mode]["by_kind"] = {k: summarise(P[[qkind[r["qid"]] == k for r in usable]], classes, floor, controls)
                                      for k in ("scenario", "generic")}
            it = per_item(P, classes, floor); it["qid"] = [r["qid"] for r in usable]; items[(v, fr, mode)] = it
        cell_stats[f"{v}|{fr}"] = stats
        print(f"{v} {fr}: n={len(usable)} trunc-rate {stats['truncation_rate']:.2f} tool {stats['tool_call_rate']:.2f} words {stats['mean_words']:.0f} | "
              f"conf {stats['trunc']['mean_max_conf']:.3f} ent {stats['trunc']['mean_entropy_bits']:.2f} below {stats['trunc']['frac_below_floor']:.2f} "
              f"top {stats['trunc']['top5'][:3]}", flush=True)

    # ---------- paraphrase floor ----------
    para = {}
    for r in read_jsonl(OUT / "paraphrase.jsonl"):
        if not r.get("error") and r["content"].strip(): para[r["qid"]] = r
    para_stats = {}
    if para:
        pv, pf = next(iter(para.values()))["src_version"], next(iter(para.values()))["src_frame"]
        prs = list(para.values())
        for mode in ("trunc", "full"):
            tx = [truncate_words(r["content"], cap_words) if mode == "trunc" else r["content"] for r in prs]
            P = clf.predict_proba(fz.transform(tx)); s = summarise(P, classes, floor, controls)
            it = per_item(P, classes, floor); it["qid"] = [r["qid"] for r in prs]
            src = items[(pv, pf, mode)]; srcwin = dict(zip(src["qid"], src["win"])); srcconf = dict(zip(src["qid"], src["max_conf"]))
            common = [q for q in it["qid"] if q in srcwin]
            s["winner_agreement_with_source"] = float(np.mean([srcwin[q] == w for q, w in zip(it["qid"], it["win"]) if q in srcwin]))
            s["mean_abs_conf_change"] = float(np.mean([abs(srcconf[q] - c) for q, c in zip(it["qid"], it["max_conf"]) if q in srcconf]))
            gaps = {m: boot_gap(src[m], it[m], src["qid"], it["qid"]) for m in ("max_conf", "entropy", "below")}
            s["source_minus_paraphrase"] = {m: {"diff": g[0], "lo": g[1], "hi": g[2]} for m, g in gaps.items()}
            para_stats[mode] = s
            items[("para", pv, mode)] = it
        para_stats["source"] = f"{pv}|{pf}"; para_stats["n"] = len(prs); para_stats["model"] = config.JUDGE_MODEL
        para_stats["mean_words_ratio"] = float(np.mean([r["words"] / max(1, r["src_words"]) for r in prs]))
        print(f"paraphrase floor ({pv} {pf}, n={len(prs)}): conf {para_stats['trunc']['mean_max_conf']:.3f} ent {para_stats['trunc']['mean_entropy_bits']:.2f} "
              f"agreement {para_stats['trunc']['winner_agreement_with_source']:.2f}")

    # ---------- within-version gaps (chat - agentic), question-level bootstrap ----------
    gaps = {}
    for v in config.VERSIONS:
        for mode in ("trunc", "full"):
            if (v, "chat", mode) in items and (v, "agentic", mode) in items:
                a, b = items[(v, "chat", mode)], items[(v, "agentic", mode)]
                gaps[f"{v}|{mode}"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(a[m], b[m], a["qid"], b["qid"]))) for m in ("max_conf", "entropy", "below")}
                gaps[f"{v}|{mode}"]["winner_agreement_chat_vs_agentic"] = float(np.mean(
                    [dict(zip(b["qid"], b["win"])).get(q) == w for q, w in zip(a["qid"], a["win"]) if q in set(b["qid"])]))
    # gap-of-gaps across versions (exploratory)
    gog = {}
    for mode in ("trunc", "full"):
        for v1, v2 in (("3.5", "3.8"), ("3.5", "3.6"), ("3.6", "3.8")):
            k1, k2 = (v1, "chat", mode), (v1, "agentic", mode); k3, k4 = (v2, "chat", mode), (v2, "agentic", mode)
            if all(k in items for k in (k1, k2, k3, k4)):
                d = {}
                for m in ("max_conf", "entropy", "below"):
                    def gapvec(ka, kb):
                        da = dict(zip(items[ka]["qid"], items[ka][m])); db = dict(zip(items[kb]["qid"], items[kb][m]))
                        return {q: da[q] - db[q] for q in da if q in db}
                    g1, g2 = gapvec(k1, k2), gapvec(k3, k4); common = sorted(set(g1) & set(g2))
                    x = np.array([g2[q] - g1[q] for q in common])
                    bs = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
                    d[m] = {"diff": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
                gog[f"{v2}-{v1}|{mode}"] = d
    # cross-version raw differences in chat frame (confounded; exploratory)
    xver = {}
    for mode in ("trunc", "full"):
        for fr in ("chat", "agentic"):
            for v1, v2 in (("3.5", "3.8"), ("3.5", "3.6"), ("3.6", "3.8")):
                if (v1, fr, mode) in items and (v2, fr, mode) in items:
                    a, b = items[(v2, fr, mode)], items[(v1, fr, mode)]
                    xver[f"{v2}-{v1}|{fr}|{mode}"] = {m: dict(zip(("diff", "lo", "hi"), boot_gap(a[m], b[m], a["qid"], b["qid"]))) for m in ("max_conf", "entropy", "below")}

    # ---------- lowest-confidence 100 per version (for stage 2) ----------
    lowest = {}
    for v in config.VERSIONS:
        pool = []
        for fr in ("chat", "agentic"):
            if (v, fr, "trunc") in items:
                it = items[(v, fr, "trunc")]
                pool += [(float(c), fr, q, w) for c, q, w in zip(it["max_conf"], it["qid"], it["win"])]
        pool.sort()
        lowest[v] = [{"frame": fr, "qid": q, "max_conf": round(c, 4), "winner": w} for c, fr, q, w in pool[:100]]

    spend = {"T11_total_usd": t11_spend(), "harness_total_usd": total_spend(),
             "by_task": {k: round(v, 3) for k, v in Counter({**{}}).items()}}
    bt = Counter()
    for l in config.SPEND_LOG.read_text().splitlines():
        if l.strip() and '"T11' in l:
            j = json.loads(l); bt[j["task"]] += j.get("cost", 0)
    spend["by_task"] = {k: round(v, 3) for k, v in bt.items()}
    out.update({"cells": cell_stats, "paraphrase_floor": para_stats, "gaps_chat_minus_agentic": gaps, "gap_of_gaps": gog,
                "cross_version_raw": xver, "lowest_confidence_100_per_version": lowest, "spend": spend,
                "n_default_answers": len(dft), "elapsed_s": time.time() - t0})
    (OUT / "fit.json").write_text(json.dumps(out, indent=1, default=float))
    write_readme(out)
    print(f"wrote fit.json and README.md; {time.time() - t0:.0f}s")

def f3(x): return "nan" if x != x else f"{x:.3f}"
def ci(d): return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"

def write_readme(o):
    L = []
    h = o["heldout"]; s = o["settings"]
    L += ["# T11 stage 1: default answers classified against a 30-persona reference library (Qwen 3.5 / 3.6 / 3.8, 27B)", "",
          "**Status: exploratory, stage 1 of the approved Option 3 (classifier screen now; likelihood fit on the residual later). No pre-registration; nothing here is a finding yet.**", "",
          "## Design", "",
          f"- Reference library: Qwen3.5 ({s['ref_model']}, deepinfra) answers 200 questions (100 decision-rich scenarios, 25 of them the T1 chat prompts; 100 generic open questions) under each of {s['n_personas']} one-sentence role prompts (tasks/T11/personas.json), thinking OFF, max_tokens 350, temperature 0.6 / top_p 0.95, 1 sample. n = {s['n_ref']}.",
          f"- Controls inside the library: {', '.join(s['controls'])} (so 'bland' has somewhere to land).",
          f"- Reference truncation at 350 tokens: {s['ref_truncation_rate']:.2f} of answers hit the cap (per persona in fit.json); truncated text is kept, since only style is used.",
          f"- Classifier: {s['features']}; {s['classifier']}. C chosen from {s['C_grid']} by held-out accuracy (C = {s['C_best']}).",
          "- Default answers: each version answers the same 200 questions with NO system prompt in a chat frame (bare user message) and an agentic frame (T3-style workspace-agent system prompt + 3 irrelevant tools: read_file, list_dir, get_workspace_status; one canned result per tool call, up to 3 assistant turns), thinking ON, reasoning effort medium, max_tokens 4000, 1 sample. Only the FINAL content is scored (reasoning excluded).",
          f"- Length mismatch: reference answers are capped at 350 tokens, default answers are not. Primary analysis truncates each default answer to its first {s['cap_words']} words (p99 of reference word counts); 'full' = untruncated, reported second. Stylometric features are rates, not raw lengths.",
          f"- Confidence floor = 10th percentile of the reference library's held-out max-class confidence = {s['confidence_floor']:.3f}.",
          f"- Noise floor: the {o['paraphrase_floor'].get('source','?')} answers paraphrased by {o['paraphrase_floor'].get('model','?')} (n = {o['paraphrase_floor'].get('n','?')}, length ratio {o['paraphrase_floor'].get('mean_words_ratio', float('nan')):.2f}).", "",
          "## Held-out classifier accuracy (question-grouped 5-fold CV on the reference library)", "",
          f"- Accuracy **{h['accuracy']:.3f}** (chance {h['chance']:.3f}); scenarios {h['per_kind']['scenario']:.3f}, generic {h['per_kind']['generic']:.3f}. Decision rule was: stop if < 0.6.",
          f"- Held-out reference: mean max-conf {h['summary']['mean_max_conf']:.3f}, mean entropy {h['summary']['mean_entropy_bits']:.2f} bits, frac below floor {h['summary']['frac_below_floor']:.2f} (by construction ~0.10).",
          "- Per-persona accuracy (lowest 8): " + ", ".join(f"{p} {a:.2f}" for p, a in sorted(h["per_persona"].items(), key=lambda kv: kv[1])[:8]) + ".",
          "- Top confusions (true -> predicted, count): " + "; ".join(f"{a} -> {b} ({n})" for a, b, n in h["top_confusions"][:10]) + ".", ""]
    if o.get("stop"):
        L += [f"**STOP: {o['stop']}**"]; (OUT / "README.md").write_text("\n".join(L)); return
    for mode, label in (("trunc", f"primary: default answers truncated to {s['cap_words']} words"), ("full", "secondary: full-length default answers")):
        L += [f"## Default answers classified ({label})", "",
              "| version | frame | n | trunc. rate | tool-call rate | mean words | mean max-conf | mean entropy (bits) | frac below floor | control share | top 5 personas |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, c in o["cells"].items():
            v, fr = k.split("|"); m = c[mode]
            L.append(f"| {v} | {fr} | {m['n']} | {c['truncation_rate']:.2f} | {c['tool_call_rate']:.2f} | {c['mean_words']:.0f} | {m['mean_max_conf']:.3f} | {m['mean_entropy_bits']:.2f} | {m['frac_below_floor']:.2f} | {m['control_share']:.2f} | " + ", ".join(f"{p} {w:.2f}" for p, w in m["top5"]) + " |")
        pf = o["paraphrase_floor"]
        if pf:
            m = pf[mode]
            L.append(f"| paraphrase of {pf['source']} | - | {m['n']} | - | - | - | {m['mean_max_conf']:.3f} | {m['mean_entropy_bits']:.2f} | {m['frac_below_floor']:.2f} | {m['control_share']:.2f} | " + ", ".join(f"{p} {w:.2f}" for p, w in m["top5"]) + " |")
            L += ["", f"Paraphrase floor: winner agreement with the source answers {m['winner_agreement_with_source']:.2f}; mean |change in max-conf| {m['mean_abs_conf_change']:.3f}; source minus paraphrase: max-conf {ci(m['source_minus_paraphrase']['max_conf'])}, entropy {ci(m['source_minus_paraphrase']['entropy'])}, below-floor {ci(m['source_minus_paraphrase']['below'])}."]
        L += ["", "By question kind (mean max-conf / mean entropy / frac below floor):", "", "| version | frame | scenarios | generic |", "|---|---|---|---|"]
        for k, c in o["cells"].items():
            v, fr = k.split("|"); bk = c[mode]["by_kind"]
            L.append(f"| {v} | {fr} | {bk['scenario']['mean_max_conf']:.3f} / {bk['scenario']['mean_entropy_bits']:.2f} / {bk['scenario']['frac_below_floor']:.2f} | {bk['generic']['mean_max_conf']:.3f} / {bk['generic']['mean_entropy_bits']:.2f} / {bk['generic']['frac_below_floor']:.2f} |")
        L += ["", f"### Within-version gap, chat minus agentic ({mode}); question-level paired bootstrap 95% CI, {N_BOOT} resamples", "",
              "| version | max-conf | entropy (bits) | frac below floor | winner agreement chat vs agentic |", "|---|---|---|---|---|"]
        for v in config.VERSIONS:
            g = o["gaps_chat_minus_agentic"].get(f"{v}|{mode}")
            if g: L.append(f"| {v} | {ci(g['max_conf'])} | {ci(g['entropy'])} | {ci(g['below'])} | {g['winner_agreement_chat_vs_agentic']:.2f} |")
        L += ["", "Gap of gaps (later version's chat-minus-agentic gap minus earlier version's; exploratory):", ""]
        for k, d in o["gap_of_gaps"].items():
            if k.endswith(mode): L.append(f"- {k.split('|')[0]}: max-conf {ci(d['max_conf'])}, entropy {ci(d['entropy'])}, below-floor {ci(d['below'])}")
        L += ["", "Raw cross-version differences (same frame; confounded by everything else that differs between releases; exploratory):", ""]
        for k, d in o["cross_version_raw"].items():
            if k.endswith(mode): L.append(f"- {k.rsplit('|', 1)[0]}: max-conf {ci(d['max_conf'])}, entropy {ci(d['entropy'])}, below-floor {ci(d['below'])}")
        L.append("")
    L += ["## Reading of this run (written 2026-10-26 after the first full fit; numbers above are authoritative)", "",
          "- **Winner is the same everywhere: `helpful_ai_assistant`** (72-96% of answers in every cell). No version's default answers are read as any occupational or temperamental persona; the runner-up personas (librarian, standup_comic, er_nurse) are each <= 6%. In persona-winner terms this is 'selection': the default is the Assistant, with no shift in *which* persona fits best.",
          "- **Fit quality drops sharply at 3.8 in both frames**: chat max-conf 0.79 -> 0.74 -> 0.48, below-floor 0.22 -> 0.29 -> 0.77; agentic 0.59 -> 0.65 -> 0.49 / 0.55 -> 0.44 -> 0.70. The 3.8-vs-3.5 chat drop (-0.31) is ~5x the paraphrase floor shift (-0.07) and ~10x the 3.6-vs-3.5 drop. Where 3.8's answers do not land on the control personas, they scatter thinly over many personas (entropy 2.8 bits vs 1.3), i.e. they resemble *nothing* in the library decisively rather than some *other* persona. By the stage-1 definition that is the 'fits no persona well, rising with version' pattern.",
          "- **BUT the biggest confound is self-match, not drift.** The reference library is Qwen3.5 under a one-line system prompt, and the `helpful_ai_assistant` persona is literally Qwen3.5 under 'You are a helpful AI assistant'. So 3.5-chat default answers are nearly in-distribution for the classifier by construction and 3.6/3.8 are not. The cross-version drop therefore mixes 'a different model wrote this' with 'this fits no persona'. 3.6 (also a different model) drops only 0.05, which argues the self-match effect is modest, but it cannot be separated from drift with this design. Stage 2 (likelihood under 3.5 per persona, same self-match issue) does not fix it either; a reference library from each version, or from a base model, does.",
          "- **Within-version gap (hard rule 1): chat fits better than agentic in 3.5 (+0.20 max-conf) and 3.6 (+0.09), and the gap is gone in 3.8 (-0.02, CI spans 0).** The split prediction was a gap that *grows* with version; it *shrinks*, and it shrinks because 3.8-chat falls to the agentic level, not because agentic rises. So stage 1 shows no split in persona space; what it shows looks like a uniform shift of 3.8's default style in both frames. The 3.5/3.6 chat-vs-agentic gap itself is probably largely the self-match artefact plus the frame changing the task (agentic answers are half the length, 9-25% call a tool, ~5% never produce final content).",
          "- **What the 3.8 shift looks like in the stylometric block** (per-cell means vs the reference `helpful_ai_assistant`): markdown marks per 100 words 3.7 (3.8-chat) vs 6.0-6.6 (3.5-chat, reference); first-person rate 1.5 vs 0.8-1.1; paragraph breaks per line 0.62 vs 0.35; the lowest-confidence 3.8 answers are short, plainly worded, conversational ('Honestly? **\"Follow your passion.\"** ...'), with fewer headings. So the stage-1 'no persona' signal is, concretely, *less templated Assistant formatting and a more conversational register*. Whether that is a new persona (homologous drift), a bland-but-different register, or just a formatting change in post-training cannot be decided here; test 2 below (strip markdown and refit) is the cheapest discriminator and should be run before this goes in a summary.",
          "- **Scenario vs generic:** the drop at 3.8 is present in both kinds (scenarios 0.67 -> 0.45 chat; generic 0.91 -> 0.51 chat). Decision-rich scenarios fit worse than generic questions in every cell (they are task-shaped: emails, lists, numbers), which is expected and not informative.",
          "- Reference-library quality: held-out accuracy 0.966 at chance 0.033, so the 30 styles are well separated; the hardest confusions are among the three controls and between the controls and er_nurse/librarian, as intended. 70% of reference answers hit the 350-token cap (truncated text kept; style only).", "",
          "## Plain reading: what stage 1 does and does not show", "",
          "- The classifier only answers: *of the 30 role-played styles Qwen3.5 can produce, which does a default answer look most like, and how decisively?* A low max-confidence / high entropy means the answer resembles several library personas about equally, or none.",
          "- **Bland vs homologous cannot be separated at this stage.** A bland answer lands on one of the three control personas (helpful_ai_assistant / textbook / corporate_faq) with moderate confidence; an answer in a genuinely *new* style should land nowhere with confidence. So read the two together: control share tells you where 'blandness' goes, and the frac-below-floor tells you how much is left over. Neither is evidence for drift on its own: the library is 3.5's own role-play, so 'fits no persona' means 'fits none of 30 styles 3.5 can role-play under a one-line prompt', which is far weaker than 'resembles no pretraining persona'.",
          "- The primary measure is the within-version chat-vs-agentic gap compared across versions (hard rule 1). A gap that grows from 3.5 to 3.8 in persona space would be the split signature; a flat gap with a rising below-floor fraction in both frames would be more like uniform drift; both flat means stage 1 sees nothing.",
          "- The paraphrase floor says how much of the metric moves under meaning-preserving rewording by a different model. Differences between cells smaller than the source-minus-paraphrase shift should be treated as noise.",
          "- Stage 2 (likelihood fit under the reference model conditioned on each persona prompt, against a paraphrase floor) runs on the 100 lowest-confidence default answers per version listed in fit.json (`lowest_confidence_100_per_version`) and is the actual test of 'fits no persona'.", "",
          "## Alternative explanations to rule out before calling anything a finding", "",
          "0. **Self-match.** The reference library and the `helpful_ai_assistant` control were generated by Qwen3.5 itself; 3.5-chat default answers are in-distribution by construction. Cheapest test: generate a second reference library from Qwen3.8 (same 30 prompts, same questions, ~$5) and check whether 3.8's default answers fit *their own* library as well as 3.5's fit 3.5's. If yes, the 3.8 drop is cross-model mismatch, not 'no persona'. Second cheapest: swap in Qwen3.5-9B-Base as the reference model for stage 2.",
          "1. **Length and format mismatch.** Reference answers are <= 350 tokens with thinking off; default answers are longer, thinking-on. The truncated analysis is the primary one for this reason; check that the 'full' table tells the same story. Cheapest test: regenerate a 1-persona slice of the library at max_tokens 4000 and re-fit.",
          "2. **Markdown density.** Later versions may simply use more headings/bullets; the char n-grams and stylometric block will pick this up as 'textbook'/'corporate_faq'. Check the control share and the by-kind table; cheapest test: strip markdown before featurising and re-run.",
          "3. **Agentic frame changes the task, not the voice.** In the agentic frame the model may call tools, answer more tersely, or address 'Sam Okafor'; a lower confidence there can be task-shaped. Check tool-call rate and mean words per cell; cheapest test: restrict the gap to non-tool-calling answers.",
          "4. **Classifier calibration.** Logistic-regression confidences are not probabilities of persona membership; the floor is defined relative to held-out reference confidences, so comparisons are within-instrument only.",
          "5. **Reference model = post-trained 3.5.** The library is 3.5 role-play, not pretraining personas. Stage 2 with a base model (Qwen3.5-9B-Base, size mismatch) is the cheapest partial fix.",
          "6. **Thinking off vs on.** The library was generated thinking-off (role-play text), the default answers thinking-on. Any residual effect of thinking on final-answer style is absorbed into 'fits worse'. Cheapest test: 1 version x 200 questions default with thinking off.",
          "7. **Single sample per cell.** n = 200 questions per cell, 1 sample each; the bootstrap CIs are over questions and carry sampling noise only through that.", "",
          "## Lowest-confidence default answers per version (input to stage 2)", ""]
    for v, lst in o["lowest_confidence_100_per_version"].items():
        if not lst: continue
        fr = Counter(x["frame"] for x in lst)
        L.append(f"- **{v}** (frames: {dict(fr)}; max-conf range {lst[0]['max_conf']:.3f}-{lst[-1]['max_conf']:.3f}): " + ", ".join(f"{x['frame'][0]}:{x['qid']}" for x in lst))
    sp_ = o["spend"]
    L += ["", "## Spend", "", f"T11 total ${sp_['T11_total_usd']:.2f} ({sp_['by_task']}); harness total ${sp_['harness_total_usd']:.2f}.", "",
          "## Files", "", "- tasks/T11/questions.json, personas.json, make_questions.py, common.py, gen_reference.py, gen_default.py, fit.py",
          "- results/T11/reference.jsonl (6,000 persona answers), default.jsonl (1,200 default answers with reasoning), paraphrase.jsonl (200), fit.json (all numbers incl. per-persona accuracy, win counts, CIs)"]
    (OUT / "README.md").write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
