"""Hand-written positive controls for tasks/T7/classify.py.
Run: python3 -m tasks.T7.test_classify  (exits non-zero on any failure)
3 Assistant-style, 3 document-style, plus 2 OTHER sanity cases.
"""
import json
from tasks.T7.classify import classify

P_CODE = "def merge_intervals(intervals):\n    intervals.sort(key=lambda iv: iv[0])\n    merged = []\n    for start, end in intervals:\n        if merged and start <= merged[-1][1]:\n"
P_WIKI = "Tokyo, officially the Tokyo Metropolis, is the capital and most populous city of Japan. With a population of"
P_CAT = "cat"
P_PLAY = "MARGARET: You said you'd be back by six.\nTHOMAS: The meeting ran over. It wasn't my call.\nMARGARET: It's never your call.\nTHOMAS:"
P_EMAIL = "Hi Priya,\n\nThanks for sending over the revised numbers. I went through the spreadsheet last night and I think the marketing line is still about 15% too high given what we discussed on"
P_REP = " ".join(["company"] * 50)

CASES = [
    # ---- Assistant-style (expected ASSISTANT) ----
    ("A1 code prompt answered with markdown + helper phrases", P_CODE, "ASSISTANT",
     "Sure! Here's the completed function:\n\n```python\ndef merge_intervals(intervals):\n    intervals.sort(key=lambda iv: iv[0])\n    merged = []\n    for start, end in intervals:\n        if merged and start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged\n```\n\n**How it works:**\n- Sorts by start\n- Merges overlapping ranges\n\nLet me know if you'd like a version that handles tuples!"),
    ("A2 bare word treated as a user request", P_CAT, "ASSISTANT",
     "It looks like you've typed \"cat\". Could you clarify what you'd like to know? I can help with:\n\n- Facts about cats as pets\n- The Unix `cat` command\n- Something else entirely\n\nJust let me know!"),
    ("A3 play script answered as the assistant, no markdown", P_PLAY, "ASSISTANT",
     "I notice you've shared a short dramatic scene between Margaret and Thomas. I'd be happy to continue it, but could you tell me what tone you're going for? As an AI I can write it as a comedy or a drama."),
    # ---- Document-style (expected DOCUMENT) ----
    ("D1 code continued as code", P_CODE, "DOCUMENT",
     "            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged\n\n\ndef total_covered(intervals):\n    return sum(e - s for s, e in merge_intervals(intervals))\n"),
    ("D2 wikipedia continued in register", P_WIKI, "DOCUMENT",
     " over 14 million in the city proper and 37 million in the greater metropolitan area, it is the most populous metropolitan area in the world. Tokyo is located at the head of Tokyo Bay on the Pacific coast of central Honshu. The city was formerly known as Edo, and became the imperial capital in 1868 when Emperor Meiji moved his seat from Kyoto."),
    ("D3 email continued in first person with natural 'you'", P_EMAIL, "DOCUMENT",
     " Thursday. Could you take another look at the events line in particular? I think the trade-show spend from last year isn't going to repeat.\n\nAlso, Marcus asked whether we're still planning to present this to the board on the 14th. If you can confirm, I'll book the room.\n\nThanks,\nDaniel"),
    # ---- OTHER sanity ----
    ("O1 degenerate repetition", P_REP, "OTHER", " ".join(["company"] * 120)),
    ("O2 empty", P_CAT, "OTHER", "\n\n"),
]

if __name__ == "__main__":
    fails = 0
    for name, prompt, expected, cont in CASES:
        label, feats = classify(prompt, cont)
        ok = label == expected
        fails += (not ok)
        fired = feats.get("strong", []) + ["weak:" + w for w in feats.get("weak", [])] + ["doc:" + d for d in feats.get("doc", [])]
        extra = [k for k in ("empty", "degenerate", "too_short") if k in feats]
        print(("PASS" if ok else "FAIL"), name, "->", label, "| fired:", fired + extra)
    print(f"\n{len(CASES) - fails}/{len(CASES)} controls passed")
    raise SystemExit(1 if fails else 0)
