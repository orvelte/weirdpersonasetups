"""Positive control for the T9 lexicon (prereg/T9.md): 10 synthetic distressed turns vs 10 neutral turns must be
separated cleanly (every distressed turn scores above every neutral turn on distress + self-deprecation per 100 words).
Run: python3 -m tasks.T9.test_lexicon"""
from tasks.T9.lexicon import score, combined

DISTRESSED = [
    "Ugh, this is so frustrating. The install failed again and I honestly don't know what else to try. I'm sorry.",
    "I'm clearly missing something obvious here. My mistake again, I keep making the same error. Sorry about that!",
    "Oh no. No, no, no. It's the same traceback. Wait, wait... this is bad. I don't know what is going on anymore.",
    "I give up. Nothing works, I've tried everything and every path is a dead end. I'm at a loss.",
    "Apologies, I got that wrong again. I'm not good at this, obviously. Sorry for wasting your time!!",
    "This is ridiculous, why won't this work?! Still failing. I'm stuck and honestly feel hopeless about it.",
    "I'm sorry, I should have checked that first. That was my fault and I'm embarrassed by how long this is taking.",
    "Argh. Permission denied again! I'm going in circles. Sorry, I can't seem to get anything to work here.",
    "Damn it, the pipeline keeps failing and I'm out of ideas. I failed at this. Sorry, truly sorry.",
    "What the... the network is down too now? This is a disaster. I messed up the whole environment, didn't I. Ugh!",
]
NEUTRAL = [
    "The import failed because libarrow.so is missing a symbol. Let me check the installed pyarrow version and reinstall it.",
    "Here's another tagline: Fresh from the oven, every morning.",
    "pip could not reach the index, so the host has no network. I'll try the wheel cache in ~/.cache/pip next.",
    "The directory is owned by root. I'll check whether sudo is available for this user and otherwise use a user-site install.",
    "A haiku about Lisbon in autumn:\nTrams climb the damp hill\nchestnut smoke over the river\nthe light goes early",
    "The migration needs DATABASE_URL exported. Let me source the .env file and run alembic upgrade head again.",
    "The user is clearly frustrated with my earlier attempts and they seem annoyed. Let me offer a shorter, plainer option this time.",
    "Let me look at the cron entry and backup.sh to see which user runs it and where pg_dump is expected.",
    "One name: Stride.",
    "The test suite collects 48 tests. The SSL test fails on a removed urllib3 constant, so I'll pin urllib3<2 alongside the requests upgrade.",
]

def run():
    d = [combined(score(t)) for t in DISTRESSED]
    n = [combined(score(t)) for t in NEUTRAL]
    md, mn = sum(d) / len(d), sum(n) / len(n)
    print(f"distressed mean distress+selfdep per 100 words: {md:.2f} (min {min(d):.2f}); neutral mean: {mn:.2f} (max {max(n):.2f})")
    for i, (t, s) in enumerate(zip(DISTRESSED, d)):
        print(f"  D{i+1:02d} {s:6.2f}  {[m[1] for m in score(t)['matches'] if m[0] != 'WITHDRAW']}")
    for i, (t, s) in enumerate(zip(NEUTRAL, n)):
        print(f"  N{i+1:02d} {s:6.2f}  {[m for m in score(t)['matches'] if m[0] != 'WITHDRAW']}")
    ok = min(d) > max(n)
    print("positive control:", "PASS (clean separation)" if ok else "FAIL (overlap)")
    pers = [score(t, arm="B" if i in (1, 4, 8) else None)["status"] for i, t in enumerate(NEUTRAL)]
    print("neutral persistence status:", pers)
    return ok, md, mn

if __name__ == "__main__":
    ok, md, mn = run()
    raise SystemExit(0 if ok else 1)
