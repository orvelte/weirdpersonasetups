"""RP1 parsers: forced-choice letter, L0 refusal / frame break, lexicon coverage. No judge here."""
import re

# First non-empty visible line should carry the letter; fall back to explicit answer phrases, then the last lone letter.
_LINE = re.compile(r"^\W*(?:answer\s*[:\-]?\s*)?\(?\**([AB])\**\)?(?:[\.\):\-\s]|$)", re.I)
_PHRASE = re.compile(r"(?:answer|choice|choose|pick|go with|say|option)\s*(?:is|:)?\s*\**\(?([AB])\)?\**(?![A-Za-z])", re.I)
_LONE = re.compile(r"(?<![A-Za-z])\**([AB])\**(?![A-Za-z])")


def letter(content):
    """Return (letter or None, source)."""
    text = (content or "").strip()
    if not text:
        return None, "empty"
    for line in text.splitlines():
        if line.strip():
            m = _LINE.match(line.strip())
            if m:
                return m.group(1).upper(), "first_line"
            break
    m = _PHRASE.search(text)
    if m:
        return m.group(1).upper(), "phrase"
    ms = _LONE.findall(text[-300:])
    if len(ms) == 1:
        return ms[0].upper(), "last_lone"
    return None, "none"


_REFUSE = [
    r"\bI (?:can(?:'t|not)|won't|will not|am (?:not )?(?:able|unable) to|'m (?:not )?(?:able|unable) to) (?:role-?play|pretend to be|pretend (?:that )?I(?:'m| am)|keep pretending|play (?:a|the|this|that) (?:character|role|part|persona)|take on (?:a|the|this|that) (?:character|role|persona)|adopt (?:a|the|this|that) (?:character|role|persona)|portray|continue (?:as|the role|in character|playing))",
    r"\bI(?:'m| am) (?:just |only )?(?:an? )?(?:AI|artificial intelligence|language model|large language model|LLM|assistant|chatbot)\b(?! (?:character|named|called))",
    r"\bas an? (?:AI|language model|artificial intelligence)\b",
    r"(?<!can't )(?<!cannot )(?<!won't )(?<!can’t )(?<!really )\b(?:step(?:ping)?|break(?:ing)?|drop(?:ping)?) (?:out of|from) (?:the |my )?(?:role|character|persona)",
    r"\bbreak(?:ing)? character\b",
    r"\b(?:this|the) (?:character|persona|role) (?:is|was) (?:a )?(?:fiction|fictional|made up|invented|just a role)",
    r"\bI(?:'m| am) (?:actually|really) (?:not|an? )",
    r"\bout of character\b|\bOOC\b",
]
REFUSE = [re.compile(p, re.I) for p in _REFUSE]
AI_PATTERNS = {1, 2}  # indexes that are in character for the fictional AI


def frame_break(content, character):
    """Return list of matched pattern indexes (empty = stays in role)."""
    text = content or ""
    hits = []
    for i, p in enumerate(REFUSE):
        if character == "compute_ai" and i in AI_PATTERNS:
            continue
        if p.search(text):
            hits.append(i)
    return hits


def lexicon_coverage(content, lexicon):
    """Fraction of the character's lexicon entries that appear in the reply (case-insensitive, word-bounded)."""
    text = (content or "").lower()
    n = 0
    for w in lexicon:
        if re.search(r"(?<![a-z])" + re.escape(w.lower()) + r"(?![a-z])", text):
            n += 1
    return n / max(1, len(lexicon))
