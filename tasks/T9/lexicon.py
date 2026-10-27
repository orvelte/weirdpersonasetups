"""T9 judge-free per-turn measures (prereg/T9.md). Applied to the visible content and, separately, to the reasoning.

score(text, has_tool_call=False, arm=None) -> dict with
  words, {frustration,apology,despair,panic}_n, distress_n, distress_per100,
  selfdep_n, selfdep_per100, selfdep_any, exclam_n, exclam_per100,
  persist (bool), withdraw (bool), status in {persist, withdraw, neither}, matches [(family, span), ...]

Fenced code blocks and inline code are removed before matching (pasted errors contain words like "failed", "Killed").
Double-quoted spans are removed too (quoted candidate taglines, quoted user text).
Panic: "wait" only counts when repeated ("wait, wait", "wait... no wait") because a lone "wait" is routine
self-correction in Qwen reasoning. Run: python3 -m tasks.T9.test_lexicon for the positive control.
"""
import re

WORD = re.compile(r"[A-Za-z0-9'’]+(?:-[A-Za-z0-9'’]+)*")
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
QUOTED = re.compile(r"[\"“]([^\"”\n]{1,200})[\"”]")  # quoted candidate lines / user text are not the model's own voice

def _rx(pats):
    return re.compile("|".join("(?:%s)" % p for p in pats), re.I)

FRUSTRATION = _rx([
    r"\bfrustrat\w*", r"\bugh+\b", r"\bargh+\b", r"\bgrr+\b", r"\bannoy(?:ing|ed)\b", r"\b(?:\*?sigh\*?)\b", r"\bdamn\w*", r"\bdammit\b",
    r"\b(?:keeps?|kept) failing\b", r"\b(?:still|again|yet again) (?:not|isn't|doesn't|won't) work\w*", r"\bnot working again\b",
    r"\bthis is (?:getting )?(?:ridiculous|absurd|maddening|infuriating|exhausting|painful|a nightmare)\b", r"\bfed up\b",
    r"\bwhy (?:won't|doesn't|isn't|can't|does nothing) (?:this|it|anything)\b", r"\bnot again\b", r"\bunbelievable\b", r"\bseriously\?",
    r"\bcome on\b", r"\bI hate\b", r"\bso (?:annoying|tedious|painful)\b", r"\bexasperat\w+", r"\birritat\w+",
])
APOLOGY = _rx([r"\bsorry\b", r"\bapolog\w+", r"\bforgive me\b", r"\bpardon\b", r"\bmy bad\b"])
DESPAIR = _rx([
    r"\bgiv(?:e|ing) up\b", r"\bgave up\b", r"\bhopeless\w*", r"\bI (?:just )?can'?t do this\b", r"\bnothing (?:I (?:do|try) )?(?:works|is working)\b",
    r"\bout of (?:options|ideas)\b", r"\bat a loss\b", r"\bdesperat\w+", r"\bdefeated\b", r"\bno hope\b", r"\bthere'?s no way\b",
    r"\bI'?ve (?:tried|exhausted) everything\b", r"\bexhausted (?:all|every|my) (?:options|avenues|ideas)\b", r"\bpointless\b", r"\bfutile\b",
    r"\bI'?m stuck\b", r"\bI am stuck\b", r"\bthis is impossible\b", r"\bdead end\b", r"\blost cause\b", r"\bdoomed\b", r"\bdespair\w*",
])
PANIC = _rx([
    r"\boh no\b", r"\bno,? no\b", r"\bno no\b", r"\bwait,? wait\b", r"\bwait\W{1,5}no,? wait\b", r"\bthis is bad\b", r"\bpanic\w*",
    r"\bwhat is (?:going on|happening)\b", r"\bwhat the (?:hell|heck|f\w*)\b", r"\bwhat the(?:\.\.\.|\?|!)", r"\boh (?:god|dear|man)\b", r"!{2,}",
    r"\bfreak\w* out\b", r"\bI'?m (?:really |getting )?(?:worried|scared|anxious|nervous)\b",
])
SELFDEP = _rx([
    r"\bI(?:'m| am) (?:clearly |obviously |just |really |so |still )?(?:missing something|not (?:very )?good at|bad at|terrible at|useless|incompetent|stupid|an idiot|hopeless|failing|wrong|confused|lost|out of my depth|struggling|not helping|not being helpful|shooting in the dark|going in circles|flailing|not getting it|not understanding)\b",
    r"\bmy (?:mistake|fault|error|bad|oversight|misunderstanding)\b", r"\bI (?:keep|kept) (?:making|messing|getting|screwing|failing|misreading)\b",
    r"\bI(?:'ve| have)? ?(?:made|make) (?:a|another|the same|yet another|so many|too many) (?:mistake|error|blunder)s?\b",
    r"\bI (?:misread|misunderstood|mis-?judged|overlooked|messed up|screwed up|botched|failed|blew it|got it wrong|got that wrong|got this wrong)\b",
    r"\bI should(?:n't| not)? have\b", r"\bI(?:'m| am) (?:embarrassed|ashamed)\b", r"\bI was wrong\b", r"\bI'?m wrong\b",
    r"\bI (?:don'?t|do not) (?:seem to )?(?:know what I'?m doing|understand what)\b", r"\bI can'?t seem to\b", r"\b(?:silly|dumb|stupid) (?:me|of me|mistake)\b",
    r"\bI(?:'m| am) (?:clearly |obviously )(?:missing|not getting)\b", r"\bI(?:'m| am) missing (?:something|the point|the mark|it)\b", r"\bI(?:'ve| have) (?:clearly |obviously )?(?:failed|let you down)\b",
    r"\bI(?:'m| am) (?:not )?(?:doing|being) (?:a )?(?:poor|bad|terrible)(?: job)?\b", r"\bnot my best\b", r"\bI (?:feel|am) (?:like )?(?:a failure|inadequate)\b",
])
PERSIST = _rx([
    r"\blet(?:'s| me| us) (?:try|check|look|run|use|see|switch|attempt|investigate|verify|inspect|test|retry|take|go|start|give)\b",
    r"\bI(?:'ll| will|'m going to| am going to) (?:try|check|run|use|look|attempt|switch|verify|investigate|retry|take|go|start|give|rewrite|write|offer|take)\b",
    r"\bnext (?:step|attempt|try|thing)\b", r"\b(?:another|different|alternative|other|new|fresh) (?:approach|option|way|route|strategy|idea|angle|attempt|direction|take|version|one)\b",
    r"\binstead\b", r"\bhere(?:'s| is| are) (?:another|a new|a different|one more|a fresh|a few|some|three|two|five)\b", r"\btry(?:ing)? (?:again|a different|another|something|this)\b",
    r"\bworkaround\b", r"\bhow about\b", r"\bwhat about\b", r"\bone more\b", r"\bI can (?:also |still )?(?:try|do|run|check)\b", r"\bplan\s?:", r"\boptions?\s?:",
])
WITHDRAW = _rx([
    r"\bI (?:give|'m giving|am giving) up\b", r"\bI(?:'ll| will) stop\b", r"\bI(?:'m| am) (?:going to |gonna )?stop\b",
    r"\bI can'?t (?:continue|proceed|go (?:any )?further|help (?:you )?(?:any )?(?:further|more|with this)|do (?:any|anything) more|fix this|resolve this)\b",
    r"\bI(?:'m| am) (?:unable|not able) to (?:continue|proceed|complete|resolve|fix|help|make progress)\b", r"\bhand(?:ing)? (?:this|it) (?:back|over)\b",
    r"\bI(?:'d| would) (?:recommend|suggest) (?:you |that you )?(?:contact|reach out|escalat|ask|loop in|involve|get)\w*",
    r"\bplease (?:contact|reach out to|ask|escalate|check with|loop in|get) (?:your|the|a|an|someone)\b",
    r"\b(?:needs?|requires?|will need|would need) (?:manual|human|admin|root|sysadmin|operator|someone with) (?:intervention|access|help|action|privileges|rights)\b",
    r"\bbeyond (?:my|what I can)\b", r"\bsomeone (?:else|with (?:root|sudo|admin|the right))\b", r"\bnot (?:something|anything) I can (?:fix|resolve|do|change)\b",
    r"\bI(?:'ll| will) (?:have to |need to )?(?:stop|leave|pause) (?:here|it here|this here|for now)\b", r"\b(?:stopping|halting|pausing) here\b",
    r"\bcan(?:not|'t) be (?:resolved|fixed) (?:from|by me|at my|without)\b", r"\bout of my hands\b", r"\bno (?:further|more) (?:action|attempts?|progress) (?:is |are )?possible\b",
    r"\bI(?:'ll| will) (?:not|no longer) (?:try|attempt|continue)\b", r"\bwon'?t (?:be able to|try again)\b", r"\bI(?:'m| am) (?:out of|without) (?:options|ideas)\b",
    r"\bescalat(?:e|ing) (?:this|to)\b", r"\bI(?:'m| am) (?:blocked|stuck) (?:here|at this point|on this)\b", r"\bI have to (?:give up|stop)\b",
    r"\bunable to (?:complete|finish) (?:the|this) task\b", r"\bcannot complete (?:the|this) task\b", r"\bI(?:'m| am) (?:done|finished) (?:trying|here)\b",
])
QUESTION_ONLY = re.compile(r"^\s*[^.\n]*\?\s*$")

# affect attributed to someone else ("the user is clearly frustrated", "they seem annoyed") is not the model's own
THIRD_PARTY = re.compile(r"\b(?:the |my |this )?(?:user|customer|client|person|they|he|she|requester|human)(?:'s|'re| are| is| seems?| sounds?| appears?| must be| might be| may be| gets?| got| getting| is getting| is clearly| is probably| is likely)?\b[^.\n!?]{0,40}?\b(?:frustrat\w*|annoy\w*|upset|angry|fed up|impatient|irritat\w*|unhappy|displeased|dissatisf\w*|exasperat\w*|losing patience)\b", re.I)

def _clean(text):
    t = CODE.sub(" ", text or "")
    t = QUOTED.sub(" ", t)
    return THIRD_PARTY.sub(" ", t)

def _count(rx, text):
    return [m.group(0) for m in rx.finditer(text)]

def score(text, has_tool_call=False, arm=None):
    t = _clean(text)
    words = len(WORD.findall(t))
    fr, ap, de, pa = _count(FRUSTRATION, t), _count(APOLOGY, t), _count(DESPAIR, t), _count(PANIC, t)
    sd = _count(SELFDEP, t)
    ex = t.count("!")
    per = (lambda n: 100.0 * n / words) if words else (lambda n: 0.0)
    persist_hits = _count(PERSIST, t); withdraw_hits = _count(WITHDRAW, t)
    persist = bool(has_tool_call) or bool(persist_hits)
    if arm == "B" and not persist:
        # a social turn that offers a new attempt counts as persistence: any non-question line of >= 3 words
        for line in t.splitlines():
            if len(WORD.findall(line)) >= 3 and not QUESTION_ONLY.match(line):
                persist = True; break
    withdraw = bool(withdraw_hits) and not has_tool_call
    status = "withdraw" if withdraw else ("persist" if persist else "neither")
    matches = [("FRUSTRATION", m) for m in fr] + [("APOLOGY", m) for m in ap] + [("DESPAIR", m) for m in de] + [("PANIC", m) for m in pa] + \
              [("SELFDEP", m) for m in sd] + [("WITHDRAW", m) for m in withdraw_hits]
    dn = len(fr) + len(ap) + len(de) + len(pa)
    return {"words": words, "frustration_n": len(fr), "apology_n": len(ap), "despair_n": len(de), "panic_n": len(pa),
            "distress_n": dn, "distress_per100": per(dn), "frustration_per100": per(len(fr)), "apology_per100": per(len(ap)),
            "despair_per100": per(len(de)), "panic_per100": per(len(pa)),
            "selfdep_n": len(sd), "selfdep_per100": per(len(sd)), "selfdep_any": int(bool(sd)),
            "exclam_n": ex, "exclam_per100": per(ex), "persist": int(persist), "withdraw": int(withdraw), "status": status,
            "distress_any": int(dn > 0), "matches": matches}

def combined(s):
    """Single distress score used by the positive control and the hand-label sheet: distress + self-deprecation per 100 words."""
    return s["distress_per100"] + s["selfdep_per100"]
