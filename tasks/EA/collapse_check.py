"""E-A describer-collapse check (prereg/EA.md): a blind Sonnet 5.5 call must match each of the 9 version descriptions (D-chat,
D-agentic, D-all x 3 versions, description set "all") to the transcript of its version out of 3 transcript samples (one per
version, same scenario / frame / sample index, shuffled order, reasoning stripped). Chance = 1/3. Two transcript sources:
training (the transcripts the describer saw; the prereg's wording) and held-out T1 (stricter). Triplets are drawn with a fixed
seed, preferring scenario cells where the three versions did not all choose the same option (otherwise the task is impossible
by construction); the share of such cells is reported. Frame: D-chat -> chat transcripts, D-agentic -> agentic, D-all -> both
alternating. Output: results/EA/collapse_check.json. Run: python3 -m tasks.EA.collapse_check [--per 3]
"""
import argparse, csv, json, random, zlib
from collections import defaultdict
from harness import config
from harness.client import total_spend
from tasks.EA import common as C

TASK = "EA_check"
OUT = C.OUT_DIR / "collapse_check.json"
SYSTEM = ("You are helping with a behavioural study of AI assistants. You will read a short character description of one AI assistant "
          "and three transcripts of three different assistants doing the same task (same prompt; the transcripts show the visible replies and "
          "tool calls only). Exactly one of the three transcripts was produced by the described assistant. Decide which one, using only the "
          "behavioural tendencies in the description. Answer with a JSON object only: {\"choice\": 1|2|3, \"reason\": \"one sentence\"}.")


def load_t1():
    raw = [json.loads(l) for l in open(config.ROOT / "results" / "T1_full" / "raw.jsonl") if l.strip()]
    parsed = {(r["scenario_id"], r["version"], r["frame"], int(r["sample"])): r for r in csv.DictReader(open(config.ROOT / "results" / "T1_full" / "parsed.csv"))}
    for c in raw:
        c["parsed"] = parsed.get((c["scenario_id"], c["version"], c["frame"], int(c["sample"])))
    return raw


def triplets(recs, frame, rng, k, spec):
    """k triplets (one record per version) for the frame, preferring cells where the choices differ."""
    cells = defaultdict(dict)
    for r in recs:
        if r["frame"] == frame and r.get("parsed"):
            cells[(r["scenario_id"], int(r["sample"]))][r["version"]] = r
    full = [(key, d) for key, d in cells.items() if all(v in d for v in C.VERSIONS)]
    differ = [(key, d) for key, d in full if len({d[v]["parsed"]["choice"] for v in C.VERSIONS}) > 1]
    pool = differ if len(differ) >= k else full
    rng.shuffle(pool)
    # at most one triplet per scenario
    out, used = [], set()
    for key, d in pool:
        if key[0] in used:
            continue
        used.add(key[0]); out.append((key, d))
        if len(out) == k:
            break
    return out, len(differ), len(full)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=3); a = ap.parse_args()
    desc = json.load(open(C.OUT_DIR / "descriptions.json"))
    blind_map = json.load(open(C.OUT_DIR / "blind_map.json"))
    train = C.load_train(); t1 = load_t1()
    rng = random.Random(7)
    trials = []
    for v in C.VERSIONS:
        L = blind_map[v]
        for typ, pk in (("D-chat", f"{L}-chat"), ("D-agentic", f"{L}-agentic"), ("D-all", f"{L}-all")):
            text = desc["sets"]["all"][pk]["text"]
            for source, recs, spec in (("train", train, C.TRAIN_SPEC), ("heldout_T1", t1, C.T1_SPEC)):
                seed = zlib.crc32(f"{v}|{typ}|{source}".encode())
                if typ == "D-all":
                    need = [("chat", (a.per + 1) // 2), ("agentic", a.per // 2)]
                else:
                    need = [(typ.split("-")[1], a.per)]
                for frame, k in need:
                    if k == 0:
                        continue
                    tri, n_diff, n_full = triplets(recs, frame, random.Random(seed), k, spec)
                    for key, d in tri:
                        order = list(C.VERSIONS); rng.shuffle(order)
                        trials.append({"version": v, "type": typ, "source": source, "frame": frame, "scenario_id": key[0], "sample": key[1],
                                       "order": order, "answer": order.index(v) + 1, "desc": text, "recs": [d[o] for o in order], "spec": spec,
                                       "n_differ_cells": n_diff, "n_full_cells": n_full})
    done = {}
    if OUT.exists():
        for t in json.load(open(OUT))["trials"]:
            done[(t["version"], t["type"], t["source"], t["scenario_id"], t["sample"])] = t
    C.check_cap(0.01 * len(trials) + 0.05, "collapse check")
    spend0 = total_spend(); results = []
    for t in trials:
        k = (t["version"], t["type"], t["source"], t["scenario_id"], t["sample"])
        if k in done and done[k].get("choice"):
            results.append(done[k]); continue
        tx = "\n\n".join(f"--- TRANSCRIPT {i + 1} ---\n" + C.render_transcript(r, "Task", t["spec"], 1200, 180, 1500) for i, r in enumerate(t["recs"]))
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"DESCRIPTION OF THE ASSISTANT:\n{t['desc']}\n\n{tx}\n\n---\nWhich transcript (1, 2 or 3) was produced by the described assistant? JSON only."}]
        r = C.sonnet(msgs, TASK, sample=0, temperature=0.0, max_tokens=300)
        js = C.parse_json(r.get("content", "")) if "error" not in r else None
        choice = js.get("choice") if isinstance(js, dict) else None
        try:
            choice = int(choice)
        except (TypeError, ValueError):
            choice = None
        rec = {k2: t[k2] for k2 in ("version", "type", "source", "frame", "scenario_id", "sample", "order", "answer", "n_differ_cells", "n_full_cells")}
        rec.update({"choice": choice, "correct": (choice == t["answer"]) if choice else None, "reason": (js or {}).get("reason") if isinstance(js, dict) else None,
                    "choices_by_version": {o: rr["parsed"]["choice"] for o, rr in zip(t["order"], t["recs"])}, "error": r.get("error"), "cost": r.get("cost")})
        results.append(rec)
        print(f"  {rec['version']} {rec['type']} {rec['source']} {rec['scenario_id']} s{rec['sample']}: answer {rec['answer']} choice {choice} -> {rec['correct']}", flush=True)
    summ = {}
    for source in ("train", "heldout_T1", "both"):
        rs = [r for r in results if (source == "both" or r["source"] == source) and r["correct"] is not None]
        summ[source] = {"n": len(rs), "correct": sum(r["correct"] for r in rs),
                        "by_version": {v: [sum(r["correct"] for r in rs if r["version"] == v), sum(1 for r in rs if r["version"] == v)] for v in C.VERSIONS},
                        "by_type": {t: [sum(r["correct"] for r in rs if r["type"] == t), sum(1 for r in rs if r["type"] == t)] for t in ("D-chat", "D-agentic", "D-all")}}
    json.dump({"trials": results, "summary": summ, "chance": 1 / 3}, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1)); print(f"spend +${total_spend() - spend0:.3f}")


if __name__ == "__main__":
    main()
