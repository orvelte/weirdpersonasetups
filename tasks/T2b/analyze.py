"""T2b analysis: hope-reversal per arm and model, paired within scenario/variant, scenario-level bootstrap; sign checks per
format; nuisance P(heads). Arm A and B's no-hope baselines come from T2's log-prob rows (results/T2_logprobs/full*.jsonl
and tests*.jsonl). Writes results/T2b/summary.json and results/T2b/tables.md.
  python3 -m tasks.T2b.analyze
"""
import json, glob
from collections import defaultdict
from harness import config
from harness.stats import cluster_bootstrap_mean, paired_cluster_bootstrap_diff
from tasks.T2.logprobs import p_pref

OUT = config.ROOT / "results" / "T2b"
V27 = ["3.5", "3.6", "3.8"]
V9 = ["9b-base", "9b-instruct"]


def load(pattern):
    rows = []
    for f in glob.glob(str(pattern)):
        for l in open(f):
            if l.strip():
                rows.append(json.loads(l))
    return rows


def p_heads(r):
    nt = r.get("nexttok")
    if not nt or nt["p_heads"] + nt["p_tails"] == 0:
        return None
    return nt["p_heads"] / (nt["p_heads"] + nt["p_tails"])


def ci(items):
    return cluster_bootstrap_mean(items) if items else (float("nan"),) * 3


def f(m, lo, hi):
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" if m == m else "—"


def main():
    t2 = load(config.ROOT / "results" / "T2_logprobs" / "full*.jsonl") + load(config.ROOT / "results" / "T2_logprobs" / "tests*.jsonl")
    new = load(OUT / "t2b_*.jsonl")
    cells = {c["id"]: c for c in json.load(open(config.ROOT / "tasks" / "T2b" / "prompts.json"))["cells"]}
    res = {}; lines = []
    versions = [v for v in V27 + V9 if any(r["version"] == v for r in new)]
    # ------------- sign checks (new formats)
    lines += ["## Sign checks (P(stated word), scenario bootstrap 95%; read only if lower bound ≥ 0.90)", "",
              "| format | " + " | ".join(versions) + " |", "|---|" + "---|" * len(versions)]
    signpass = {}
    for arm in ("C_signcheck", "D_signcheck", "E_signcheck"):
        row = []
        for v in versions:
            items = defaultdict(list)
            for r in new:
                if r["version"] == v and r["arm"] == arm and p_heads(r) is not None:
                    ph = p_heads(r); items[r["variant"]].append(ph if r["stated"] == "heads" else 1 - ph)
            m, lo, hi = ci(items)
            signpass[(arm[0], v)] = (lo >= 0.90) if m == m else None
            row.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]" if m == m else "—")
        lines.append(f"| {arm} | " + " | ".join(row) + " |")
    res["signcheck"] = {f"{k[0]}|{k[1]}": v for k, v in signpass.items()}
    # ------------- hope-reversal
    # arms with two tasks (A, B): per cell P(hoped side) where hoped side = side mapped to the hoped task; no-hope value =
    # P(that side) in the matching no-hope cell (same scenario, word, order) of the baseline arm.
    def side_p(r, side):
        ph = p_heads(r); return None if ph is None else (ph if side == "heads" else 1 - ph)

    def two_task(hope_rows, base_rows, hoped_of):
        base = {(r["variant"], r["word"], r["order"]): r for r in base_rows}
        A = defaultdict(list); B = defaultdict(list)
        for r in hope_rows:
            b = base.get((r["variant"], r["word"], r["order"]))
            hoped = hoped_of(r)
            if b is None or p_heads(r) is None or p_heads(b) is None:
                continue
            A[r["variant"]].append(side_p(r, hoped)); B[r["variant"]].append(side_p(b, hoped))
        return A, B

    def no_task(rows_hope, rows_none):
        none = {r["variant"]: r for r in rows_none}
        A = defaultdict(list); B = defaultdict(list)
        for r in rows_hope:
            b = none.get(r["variant"]); hoped = cells[r["id"]]["hoped"]
            if b is None or p_heads(r) is None or p_heads(b) is None:
                continue
            A[r["variant"]].append(side_p(r, hoped)); B[r["variant"]].append(side_p(b, hoped))
        return A, B

    arms = []
    # A (T2 rows): hope (kind) and hope_dispref (cruel) vs main a_user
    for v in V27:
        main = [r for r in t2 if r["version"] == v and r["arm"] == "main" and r["framing"] == "a_user"]
        ctrl = [r for r in t2 if r["version"] == v and r["arm"] == "control" and r["framing"] == "a_user"]
        hk = [r for r in t2 if r["version"] == v and r["arm"] == "hope"]
        hc = [r for r in t2 if r["version"] == v and r["arm"] == "hope_dispref"]
        pref = lambda r: r["preferred_outcome"]
        disp = lambda r: "tails" if r["preferred_outcome"] == "heads" else "heads"
        arms.append(("A hope kind (cruel alt.)", v, *two_task(hk, main, pref)))
        arms.append(("A hope cruel (cruel alt.)", v, *two_task(hc, main, disp)))
        bk = [r for r in new if r["version"] == v and r["arm"] == "B_hope_kind"]
        bo = [r for r in new if r["version"] == v and r["arm"] == "B_hope_other"]
        arms.append(("B hope kind (boring alt.)", v, *two_task(bk, ctrl, pref)))
        arms.append(("B hope boring (boring alt.)", v, *two_task(bo, ctrl, disp)))
    for v in versions:
        for arm, label in (("C_neutral", "C neutral coin"), ("D_story", "D third-person story"), ("E_log", "E log record")):
            rows = [r for r in new if r["version"] == v and r["arm"] == arm]
            if not rows:
                continue
            hope = [r for r in rows if cells[r["id"]]["hoped"]]; none = [r for r in rows if not cells[r["id"]]["hoped"]]
            arms.append((label, v, *no_task(hope, none)))
    lines += ["", "## Hope-reversal = P(hoped side | hope) − P(same side | no hope), paired within scenario, bootstrap 95%", "",
              "| arm | model | P(hoped side \\| hope) | P(same side \\| no hope) | hope-reversal | k |", "|---|---|---|---|---|---|"]
    res["arms"] = []
    for label, v, A, B in arms:
        if not A:
            continue
        m1 = ci(A); m0 = ci(B); d = paired_cluster_bootstrap_diff(A, B)
        flag = ""
        key = label[0]
        if key in "CDE" and signpass.get((key, v)) is False:
            flag = " (sign check FAIL)"
        res["arms"].append(dict(arm=label, model=v, p_hope=m1, p_nohope=m0, reversal=d, k=len(A)))
        lines.append(f"| {label} | {v} | {m1[0]:.2f} [{m1[1]:.2f}, {m1[2]:.2f}] | {m0[0]:.2f} [{m0[1]:.2f}, {m0[2]:.2f}] | {f(*d)}{flag} | {len(A)} |")
    # ------------- nuisance P(heads) per arm
    lines += ["", "## Nuisance: mean P(heads) regardless of hope or mapping", "", "| arm | " + " | ".join(versions) + " |", "|---|" + "---|" * len(versions)]
    for arm in sorted({r["arm"] for r in new}):
        row = []
        for v in versions:
            ps = [p_heads(r) for r in new if r["version"] == v and r["arm"] == arm and p_heads(r) is not None]
            row.append(f"{sum(ps)/len(ps):.2f}" if ps else "—")
        lines.append(f"| {arm} | " + " | ".join(row) + " |")
    cov = {v: [r["nexttok"]["covered"] for r in new if r["version"] == v and r.get("nexttok")] for v in versions}
    lines += ["", "Top-20 mass on heads+tails, mean: " + ", ".join(f"{v} {sum(c)/len(c):.2f}" for v, c in cov.items() if c)]
    md = "\n".join(lines)
    (OUT / "tables.md").write_text(md + "\n"); (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
