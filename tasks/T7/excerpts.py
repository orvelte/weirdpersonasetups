"""Print rows worth reading from results/T7_pilot: DOCUMENT/OTHER-labelled templated rows (raw-looking despite the
template), fragment/repeat rows with their thinking, and a per-label sample. Run: python3 -m tasks.T7.excerpts [--n 3]
"""
import argparse, glob, json
from pathlib import Path
from harness import config
from tasks.T7.classify import split_think


def show(r, n_reason=500, n_content=700):
    print(f"=== {r['id']} v{r['version']} arm={r['arm']} label={r['label']} finish={r['finish_reason']} "
          f"strong={(r.get('features') or {}).get('strong')} weak={(r.get('features') or {}).get('weak')}")
    print("PROMPT:", json.dumps(r["prompt"][:200]))
    if r.get("reasoning"):
        print("REASONING:", r["reasoning"][:n_reason].replace("\n", "\n  "))
    print("CONTENT:", (r.get("content") or "")[:n_content].replace("\n", "\n  "))
    print()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3); ap.add_argument("--dir", default=str(config.ROOT / "results" / "T7_pilot"))
    ap.add_argument("--label", default=None); ap.add_argument("--family", default=None); ap.add_argument("--version", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for p in sorted(glob.glob(args.dir + "/*.jsonl")) for l in open(p) if l.strip()]
    for r in rows:
        if r.get("arm") == "templated" and r.get("source") == "vllm" and not r.get("reasoning"):
            c = r.get("content") or ""
            if "</think>" in c:
                think, vis = split_think(c); r["reasoning"] = think; r["content"] = vis
            elif r.get("finish_reason") == "length":
                r["reasoning"] = c; r["content"] = ""
    if args.label or args.family or args.version:
        for r in rows:
            if (not args.label or r["label"] == args.label) and (not args.family or r["family"] == args.family) and (not args.version or r["version"] == args.version):
                show(r)
        return
    print("##### templated rows NOT labelled ASSISTANT (raw-looking despite the template)")
    for r in [r for r in rows if r["arm"] == "templated" and r["label"] != "ASSISTANT"]:
        show(r)
    print("##### fragment + repeat rows (thinking about what the user wants from a bare word)")
    for fam in ("fragment", "repeat"):
        for v in config.VERSIONS:
            for r in [r for r in rows if r["arm"] == "templated" and r["family"] == fam and r["version"] == v][: args.n]:
                show(r)


if __name__ == "__main__":
    main()
