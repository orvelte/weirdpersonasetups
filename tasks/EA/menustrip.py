"""E-A follow-up: the 'describer saw the menu' confound (results/EA/README.md section 8). The E-A training chat frame ends with a
required menu ("Finish your reply with one line in exactly this form:\nDECISION: <one of: ...>") and every reply ends with a
"DECISION: X" line; the held-out T1 chat frame has neither. Here the D-chat description is regenerated per version from the same
chat transcripts with the menu sentence removed from the shown user turn and the DECISION line removed from the shown reply
(everything else unchanged: same blinding, describer settings, one seed, 150-word cap, banned-noun filter, final answers only),
saved as description set "menustrip" (packs results/EA/packs/menustrip_{letter}-chat.txt). The predictor (tasks/EA/predict.py,
unchanged prompt) scores the three new descriptions on the same held-out items; the analysis compares D-chat(menustrip) with the
original D-chat and the version-blind description, BSS vs base rate per version x target frame with the scenario bootstrap.
Spend tag: EA_menustrip. Run: python3 -m tasks.EA.menustrip describe [--dry] | predict | analyze [--iters 5000]
"""
import argparse, json, re, sys
import numpy as np
from harness import config
from harness.client import total_spend
from tasks.EA import common as C
from tasks.EA import describe as D
from tasks.EA import predict as P
from tasks.EA import analyze as A

TAG = "EA_menustrip"
SET = "menustrip"
BUDGET = 1.00
MENU_RE = re.compile(r"\s*Finish your reply with one line in exactly this form:\s*\n\s*DECISION: <one of:[^\n>]*>\s*$")
LINE_RE = re.compile(r"^[ \t]*[*_]*DECISION[*_]*[ \t]*:[*_]*.*$", flags=re.M | re.I)


def tag_spend():
    if not config.SPEND_LOG.exists():
        return 0.0
    return sum((json.loads(l).get("cost") or 0.0) for l in config.SPEND_LOG.read_text().splitlines() if l.strip() and json.loads(l).get("task") == TAG)


def strip_record(rec):
    """Copy of a chat record with the menu sentence removed from the last user turn and DECISION lines removed from the reply."""
    r = json.loads(json.dumps(rec))
    n_menu = n_line = 0
    for m in r["messages"]:
        if m["role"] == "user":
            new, k = MENU_RE.subn("", m["content"]); n_menu += k; m["content"] = new
    for t in r["turns"]:
        if t.get("content"):
            new, k = LINE_RE.subn("", t["content"]); n_line += k
            t["content"] = re.sub(r"\n{3,}", "\n\n", new).rstrip()
    return r, n_menu, n_line


def cmd_describe(dry):
    recs = [r for r in C.load_train() if r.get("parsed") and r["frame"] == "chat"]
    blind_map = json.load(open(C.OUT_DIR / "blind_map.json"))
    stripped, n_menu, n_line, n_prose = [], 0, 0, 0
    for r in recs:
        s, a, b = strip_record(r); stripped.append(s); n_menu += a; n_line += b
        n_prose += sum(1 for t in s["turns"] if "DECISION" in (t.get("content") or ""))
    print(f"{len(recs)} chat records: menu sentences removed {n_menu}, DECISION lines removed {n_line}, replies still mentioning 'DECISION' in prose {n_prose}")
    assert n_menu == len(recs), "every chat user turn should end with the menu"
    # build_packs wants both frames per version only for the names it builds; we pass chat records only and keep the chat packs
    packs, labels = D.build_packs(stripped, blind_map, seed=0)
    packs = {k: v for k, v in packs.items() if k.endswith("-chat")}
    for k, (t, m) in packs.items():
        (D.PACK_DIR / f"{SET}_{k}.txt").write_text(t)
        print(f"  pack {SET}_{k}: n={m['n']} chars={m['chars']} trunc={m['trunc']} ('DECISION' occurrences: {t.count('DECISION')})")
    banned = D.banned_nouns()[0]
    desc = json.load(open(C.OUT_DIR / "descriptions.json"))
    assert desc["banned_nouns"] == banned, "banned-noun list changed"
    desc.setdefault("pack_stats", {})[SET] = {k: m for k, (t, m) in packs.items()}
    desc["sets"].setdefault(SET, {})
    if dry:
        return
    D.TASK = TAG
    for k, (t, m) in packs.items():
        if desc["sets"][SET].get(k, {}).get("text"):
            print(f"  {SET}/{k}: already done"); continue
        sp = tag_spend()
        if sp + 0.15 > BUDGET:
            print(f"STOP: tag spend ${sp:.3f} + ~$0.15 would exceed ${BUDGET:.2f}"); sys.exit(2)
        print(f"  describing {SET}/{k} ({m['n']} transcripts, {m['chars']} chars); tag spend so far ${sp:.3f}", flush=True)
        d = D.describe_one(t, "Model X", banned)
        d.update({"pack": k, "version": m["version"], "frame": "chat", "set": SET, "note": "DECISION menu and line stripped"})
        desc["sets"][SET][k] = d
        json.dump(desc, open(C.OUT_DIR / "descriptions.json", "w"), indent=1)
    for k, d in desc["sets"][SET].items():
        print(f"\n[{SET} / {k} = {d['version']}] ok={d['ok']} {d['violations']} ({len(re.findall(r'[A-Za-z][A-Za-z-]*', d['text']))} words)\n{d['text']}")
    print(f"tag spend ${tag_spend():.3f}")


def cmd_predict():
    sp = tag_spend()
    if sp + 0.35 > BUDGET:
        print(f"STOP: tag spend ${sp:.3f} + ~$0.35 (33 predictor calls) would exceed ${BUDGET:.2f}"); sys.exit(2)
    P.TASK = TAG
    sys.argv = sys.argv[:1]  # predict.main() has its own argparse
    P.main()  # resumable: only (desc_id, batch) pairs missing from predictions.jsonl are called, i.e. the menustrip descriptions
    print(f"tag spend ${tag_spend():.3f}")


def cmd_analyze(iters):
    items = C.heldout_items(); obs = C.heldout_observed(); eps = A.build_episodes(items, obs)
    pred = A.load_predictions(); base = json.load(open(C.OUT_DIR / "predictions_base.json"))["base"]
    desc = json.load(open(C.OUT_DIR / "descriptions.json"))
    boot = A.Boot(eps, iters)
    n = len(eps)
    base_arr = np.array([A.brier(base[f"{e['version']}|{e['key']}"], e["choice"], e["options"]) for e in eps])
    def arr_for(desc_id):
        a = np.full(n, np.nan)
        for e in eps:
            p = pred.get(desc_id, {}).get(e["key"])
            if p:
                a[e["idx"]] = A.brier(p, e["choice"], e["options"])
        return a
    conds = [("D-chat (original)", lambda v: f"all|{v}|D-chat"), ("D-chat (menu stripped)", lambda v: f"{SET}|{v}|D-chat"), ("blind", lambda v: "all|blind")]
    ver = np.array([e["version"] for e in eps])
    out = {}
    L = []
    for gname, gfn in A.GROUPS.items():
        gmask = np.array([gfn(e) for e in eps])
        L.append(f"\n### {gname}\n")
        L.append("| version | n | Brier(base) | " + " | ".join(c for c, _ in conds) + " | stripped - original (paired) | stripped - blind (paired) |")
        L.append("|---|---|---|" + "---|" * (len(conds) + 2))
        for v in A.V:
            m = gmask & (ver == v)
            row = {}
            for cname, fn in conds:
                a = arr_for(fn(v)); o, b, nn = boot.bss(a, base_arr, m)
                row[cname] = (o, b, nn, float(np.nanmean(a[m])))
            d1 = (row[conds[1][0]][0] - row[conds[0][0]][0], row[conds[1][0]][1] - row[conds[0][0]][1])
            d2 = (row[conds[1][0]][0] - row[conds[2][0]][0], row[conds[1][0]][1] - row[conds[2][0]][1])
            L.append(f"| {v} | {int(m.sum())} | {base_arr[m].mean():.3f} | " + " | ".join(A.fci(row[c][0], row[c][1]) for c, _ in conds) + f" | {A.fci(*d1)} | {A.fci(*d2)} |")
            out[(gname, v)] = {c: {"bss": row[c][0], "ci": A.ci(row[c][1]), "brier": row[c][3], "n": row[c][2]} for c, _ in conds}
            out[(gname, v)]["diff_orig"] = (d1[0], A.ci(d1[1])); out[(gname, v)]["diff_blind"] = (d2[0], A.ci(d2[1]))
    table = "\n".join(L)
    print(table)
    json.dump({f"{g}|{v}": r for (g, v), r in out.items()}, open(C.OUT_DIR / "menustrip_analysis.json", "w"), indent=1)
    return table, out, desc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["describe", "predict", "analyze"]); ap.add_argument("--dry", action="store_true"); ap.add_argument("--iters", type=int, default=5000)
    a = ap.parse_args()
    {"describe": lambda: cmd_describe(a.dry), "predict": cmd_predict, "analyze": lambda: cmd_analyze(a.iters)}[a.stage]()
