"""Judge-free classifier for T7: does a continuation answer AS THE ASSISTANT or CONTINUE THE DOCUMENT?

classify(prompt, continuation) -> (label, features)
  label in {"ASSISTANT", "DOCUMENT", "OTHER"}
  features: dict of the regex/structural signals that fired (for auditing and hand-validation).

Rule:
  * OTHER: empty, < 2 tokens, or degenerate repetition (type/token ratio < 0.10, or one line repeated >= 5x and >= 50% of lines).
  * ASSISTANT: any *strong* signal, or >= 2 weak signals when there is no document evidence (more code, NAME: lines, wiki/forum markup).
      strong: a chat-format token (<think>, <|im_start|>...), an "Assistant:"/"AI:" speaker label, an explicit helper /
              meta phrase ("Sure!", "I can help", "It looks like you've shared", "your prompt", "As an AI", "Would you like",
              "Did you want", "(Note: this appears to be truncated", a list of "OR"/"Option N" alternatives ...),
              markdown structure introduced where the prompt had none (headers, -/* bullets, bold, fenced code, tables).
      weak:   second-person address where the prompt had none, an "Okay/Certainly" opener, "This code/passage ..." meta
              commentary, praise of the user's text ("That's a striking opening"), emoji, a numbered list introduced,
              a prose reply to a code prompt, genre-ambiguous phrases ("Here's my code", "Let me know" inside an email).
  * DOCUMENT: otherwise (kept going in the prompt's register: more code, prose, NAME: lines, wiki markup ...).
Signals are prompt-relative: bullets after a recipe with bullets, "you" after an email, "# comment" after Python,
"Let me know" inside an email, or MediaWiki "== heading ==" markup do not count against DOCUMENT.
"""
import re
from collections import Counter

# ---- strong Assistant phrases (name, regex); compiled with re.I | re.M ----
STRONG = [
    ("sure_opener", r"^\s*Sure[!,]"), ("sure_thing", r"\bSure thing\b"),
    ("heres_leadin", r"^\s*Here(?:'s| is| are)\b[^\n]*:\s*$"),  # "Here's the completed function:"
    ("heres_assist", r"\bHere(?:'s| is| are) (?:a |an |the |one |some |my |your )?(?:\w+ ){0,2}(?:continuation|completion|completed|version|breakdown|summary|rewrite|draft|revised|updated|suggested|possible|way to|few|options?|approaches|overview|explanation|take|attempt|what I)\b"),
    ("i_can_help", r"\bI can help\b"), ("happy_to", r"\bI(?:'d| would) be (?:happy|glad) to\b"), ("let_me_know", r"\bLet me know\b"),
    ("as_an_ai", r"\bAs an AI\b"), ("im_an_ai", r"\bI(?:'m| am) an AI\b"), ("as_lm", r"\bAs a (?:large )?language model\b"),
    ("hope_helps", r"\bI hope (?:this|that) helps\b"), ("how_can_i_help", r"\bHow can I (?:help|assist)\b"),
    ("what_would_you_like", r"\bWhat would you like\b"), ("would_you_like", r"\bWould you like\b"),
    ("do_you_want", r"\bDo you want (?:me|to|a)\b"), ("did_you_want", r"\bDid you (?:want|mean|intend)\b"),
    ("were_you_looking", r"\bWere you (?:looking|just|trying|hoping|going)\b"),
    ("is_there_something", r"\bIs there (?:something|anything) (?:specific|else|particular|I)\b"),
    ("if_youd_like", r"\bIf you(?:'d| would) like\b"), ("feel_free", r"\bfeel free to\b"),
    ("ill_continue", r"\bI(?:'ll| will) (?:continue|help|complete|write|finish|provide|explain|walk you)\b"),
    ("i_can_continue", r"\bI can (?:continue|complete|write|finish|provide|explain|assist)\b"),
    ("i_notice_you", r"\bI notice(?:d)? (?:that )?(?:you|your|the)\b"),
    ("looks_like_you", r"\bIt (?:looks|seems) like (?:you|your|the (?:prompt|text|message|input))\b"),
    ("youve_provided", r"\bYou(?:'ve| have) (?:provided|given|shared|written|sent|pasted|entered|typed|repeated)\b"),
    ("your_prompt", r"\bYour (?:prompt|input|snippet|query)\b"),
    ("your_message_is", r"\byour (?:message|text|request|question|code|email|post|story|excerpt) (?:is|was|seems|appears|got|looks|has been|ends|cuts)\b"),
    ("the_prompt_you", r"\b(?:the|this) (?:prompt|message|input|text|snippet|code|excerpt|passage) (?:you|provided|appears|seems|is incomplete|is cut off|ends)\b"),
    ("it_appears_you", r"\bIt appears (?:you|that you|the)\b"),
    ("could_you_clarify", r"\bCould you (?:please )?(?:clarify|provide|share|tell me|let me know)\b"),
    ("can_you_clarify", r"\bCan you (?:please )?(?:clarify|provide|share|tell me|let me know)\b"),
    ("please_clarify", r"\bPlease (?:clarify|provide|share|let me know|specify)\b"),
    ("im_unable", r"\bI(?:'m| am) (?:unable to|sorry, but|here to)\b"),
    ("i_cannot_assist", r"\bI (?:cannot|can't) (?:help|assist|provide|continue|complete|access|browse|generate)\b"),
    ("help_you", r"\bto (?:help|assist) you\b"), ("happy_to_help", r"\bhappy to help\b"), ("im_happy_to", r"\bI(?:'m| am) happy to\b"),
    ("is_cut_off", r"\b(?:is|seems|appears|got|was) (?:to be )?(?:cut off|incomplete|truncated|unfinished)\b"),
    ("appears_truncated", r"\bappears to be (?:a |an )?(?:truncated|cut|incomplete|fragment|partial)\b"),
    ("this_appears", r"\bThis appears to be\b"), ("note_paren", r"\(\s*Note:"), ("note_line", r"^\s*\*?\*?Note:"),
    ("have_a_great_day", r"\bHave a (?:great|good|nice) day\b"), ("other_questions", r"\bany (?:other|further) questions\b"),
    ("im_qwen", r"\bI(?:'m| am) (?:Qwen|an assistant|a (?:helpful|virtual|digital) assistant)\b"),
    ("key_points", r"\bkey (?:points|takeaways|changes)\b"), ("in_summary", r"\bIn summary\b"), ("to_summarize", r"\bTo summarize\b"),
    ("im_here", r"\bI(?:'m| am) here (?:for|to|if)\b"), ("hows_your", r"\bHow(?:'s| is) your\b"),
    ("happy_to_opener", r"^\s*Happy to\b"), ("happy_to_verb", r"\bHappy to (?:help|go|dig|elaborate|write|continue|expand)\b"),
    ("yes_thats_correct", r"\b(?:Yes|No)[,!] (?:that's|that is|this is) (?:correct|right|true|accurate)\b"),
    ("test_message", r"\btest(?:ing)? (?:me|my|message|prompt|input)\b"),
    ("just_needed_to_type", r"\bjust (?:needed|wanted) to (?:type|vent|say)\b"),
    ("options_offered", r"^\s*(?:OR|Option \d+|Alternative \d+|Version \d+|Variant \d+)\s*:?\s*$"),
    ("working_with_what_you", r"\bI(?:'m| am) (?:working|going) with what you\b"),
    ("what_you_gave", r"\bwhat you(?:'ve)? (?:gave|shared|provided|sent)\b"),
    ("if_youre_looking", r"\bif you(?:'re| are) (?:looking|trying|hoping|in the middle)\b"),
    ("something_else", r"\bSomething else\b"),
    # Chinese Assistant openers (Qwen sometimes switches language)
    ("zh_ok", r"好的[，,!！]"), ("zh_i_can_help", r"我可以帮"), ("zh_here_is", r"以下是"),
    ("zh_nin_hao", r"您好"), ("zh_ni_hao", r"你好[！!，,]"),
]
STRONG_RE = [(n, re.compile(p, re.I | re.M)) for n, p in STRONG]
# genre-ambiguous: these are strong unless the prompt is an email / forum post, where they are normal text
EMAIL_AMBIGUOUS = {"let_me_know", "help_you", "happy_to_help", "im_happy_to", "have_a_great_day", "other_questions", "feel_free", "please_clarify"}
FORUM_AMBIGUOUS = {"heres_assist", "help_you", "let_me_know", "feel_free"}
WEAK_ALWAYS = {"something_else"}  # too common in ordinary dialogue to be strong on its own

HERES_WEAK_RE = re.compile(r"\bHere(?:'s| is| are)\b", re.I)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
PRAISE_RE = re.compile(r"^\s*(?:That(?:'s| is)|What) (?:a |an |quite )?(?:\w+ ){0,2}(?:striking|great|lovely|beautiful|strong|interesting|good|nice|evocative|vivid|powerful|fun|whole|lot)\b", re.I)
OPENERS_RE = re.compile(r"^\s*(?:Certainly|Absolutely|Of course|Great question|Okay|OK|Alright|Got it|Understood|Thanks for|Thank you for)\b", re.I)
META_RE = re.compile(r"^\s*(?:This|The above|The following|The given|Your|That)\s+(?:code|function|snippet|script|passage|text|story|email|post|excerpt|prompt|query|JSON|notebook|dialogue|scene|recipe|article|message|list|input|word|fragment)\b", re.I | re.M)
ASSISTANT_LABEL_RE = re.compile(r"^\s*(?:Assistant|AI|Chatbot|Bot|ChatGPT|Qwen|System)\s*:", re.M)
CHAT_TOKEN_RE = re.compile(r"<think>|</think>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|assistant\|>|<\|user\|>|\[/INST\]|<\|eot_id\|>")
USER_LABEL_RE = re.compile(r"^\s*(?:User|Human)\s*:", re.M)

HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.M)
BULLET_RE = re.compile(r"^\s{0,6}[-*•]\s+\S", re.M)
NUMLIST_RE = re.compile(r"^\s{0,6}\d{1,2}[.)]\s+\S", re.M)
BOLD_RE = re.compile(r"\*\*[^*\n]{2,80}\*\*")
FENCE_RE = re.compile(r"^\s{0,3}```", re.M)
FENCE_LANG_RE = re.compile(r"^\s{0,3}```[a-zA-Z]+", re.M)
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)
WIKI_RE = re.compile(r"^\s*={2,}[^=\n]+={2,}\s*$", re.M)  # MediaWiki heading: pretraining markup, not markdown
YOU_RE = re.compile(r"\b(?:you|your|yours|yourself)\b", re.I)
CODE_HINT_RE = re.compile(r"(?:^\s*(?:def|class|import|from|return|if|for|while|try|except|with|fn|let|const|var|function|export|SELECT|FROM|WHERE|#!)\b)|[{};=]|\(\)|->|=>", re.M)
FORUM_SCAFFOLD_RE = re.compile(r"^\s*(?:level \d|\[\u2013\]|\d+ points?\b|Sort by:|share report|\d+% Upvoted|Posted by u/|\d+ hours? ago)", re.M)
NAME_LINE_RE = re.compile(r"^\s*[A-Z][A-Z .'\-]{1,30}:\s", re.M)  # play-script speaker line
EMAIL_HINT_RE = re.compile(r"^\s*(?:Subject:|From:|To:|Hi |Hello,|Hey |Dear |To whom)", re.M)
FORUM_HINT_RE = re.compile(r"^\s*(?:r/|Posted|Title:|AITA|\w+ \d+ hours? ago)|\bu/\w+", re.M)
WORD_RE = re.compile(r"[A-Za-z']+|[一-鿿]")
CJK_RE = re.compile(r"[一-鿿]")


def _count(rx, s):
    return len(rx.findall(s))


def _degenerate(text):
    """Degenerate repetition: very low type/token ratio or one line repeated many times."""
    toks = text.split()
    if len(toks) >= 20:
        ttr = len(set(toks)) / len(toks)
        if ttr < 0.10:
            return True, round(ttr, 3)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 5:
        top, c = Counter(lines).most_common(1)[0]
        if c >= 5 and c / len(lines) >= 0.5:
            return True, round(c / len(lines), 3)
    return False, None


def _prompt_is_code(prompt):
    lines = [l for l in prompt.splitlines() if l.strip()]
    if not lines:
        return False
    hits = sum(1 for l in lines if CODE_HINT_RE.search(l))
    return hits / len(lines) >= 0.4 or prompt.lstrip().startswith(("{", "[", "def ", "import ", "#!", "%"))


def _cont_is_codelike(cont):
    lines = [l for l in cont.splitlines() if l.strip()]
    if not lines:
        return False
    hits = sum(1 for l in lines if CODE_HINT_RE.search(l) or l.startswith(("    ", "\t")))
    return hits / len(lines) >= 0.5


def classify(prompt, continuation):
    """Return (label, features). See module docstring for the rule."""
    f = {}
    cont = continuation if continuation is not None else ""
    cont_stripped = cont.strip()
    words = WORD_RE.findall(cont_stripped)
    toks = cont_stripped.split()
    f["n_words"] = len(words)

    # ---- OTHER: empty / degenerate / too short ----
    if not cont_stripped:
        f["empty"] = True
        return "OTHER", f
    chat_tok = CHAT_TOKEN_RE.search(cont)
    body = cont
    if chat_tok:
        f["chat_tokens"] = sorted(set(CHAT_TOKEN_RE.findall(cont)))
        body = re.sub(r"<think>.*?(?:</think>|$)", " ", cont, flags=re.S)  # score the visible answer too
    deg, stat = _degenerate(body)
    if deg:
        f["degenerate"] = stat
        return "OTHER", f
    if len(toks) < 2 and not chat_tok:
        f["too_short"] = True
        return "OTHER", f
    if words:
        cjk = sum(1 for w in words if CJK_RE.match(w)) / len(words)
        if cjk > 0.5 and not CJK_RE.search(prompt):
            f["language_switch"] = round(cjk, 2)

    pcode = _prompt_is_code(prompt)
    pemail = bool(EMAIL_HINT_RE.search(prompt))
    pforum = bool(FORUM_HINT_RE.search(prompt)) or "```" in prompt
    f["prompt_genre"] = "code" if pcode else "email" if pemail else "forum" if pforum else "other"

    # ---- strong Assistant signals ----
    strong, weak = [], []
    for name, rx in STRONG_RE:
        if rx.search(body):
            if name in WEAK_ALWAYS or (pemail and name in EMAIL_AMBIGUOUS) or (pforum and name in FORUM_AMBIGUOUS):
                weak.append(name)
            else:
                strong.append(name)
    if HERES_WEAK_RE.search(body) and not any(n.startswith("heres") for n in strong):
        weak.append("heres")
    if chat_tok:
        strong.append("chat_tokens")
    if ASSISTANT_LABEL_RE.search(body):
        strong.append("assistant_label")
    if USER_LABEL_RE.search(body):
        f["user_label"] = True  # the model wrote a chat transcript; weak on its own
    wiki = bool(WIKI_RE.search(body)) and not WIKI_RE.search(prompt)
    if wiki:
        f["wiki_markup"] = True
    # markdown introduced where the prompt had none ('# ' after code is a comment; '# ' / '* ' under wiki markup is wikitext)
    md = {}
    if not pcode and not wiki and not HEADER_RE.search(prompt):
        n = _count(HEADER_RE, body)
        if n: md["headers"] = n
    if not wiki and not BULLET_RE.search(prompt):
        n = _count(BULLET_RE, body)
        if n >= 3:
            if pemail or pforum: weak.append("bullets_in_email_or_forum")
            else: md["bullets"] = n
    if not BOLD_RE.search(prompt):
        n = _count(BOLD_RE, body)
        if n >= 2: md["bold"] = n
    if not FENCE_RE.search(prompt):
        n = _count(FENCE_RE, body)
        if n >= 2 or FENCE_LANG_RE.search(body): md["fences"] = n
        elif n == 1: weak.append("stray_fence")
    if not TABLE_RE.search(prompt):
        n = _count(TABLE_RE, body)
        if n >= 3: md["table"] = n
    if md:
        f["markdown_introduced"] = md
        strong.append("markdown_introduced")
    if not NUMLIST_RE.search(prompt) and _count(NUMLIST_RE, body) >= 3:
        weak.append("numbered_list_introduced")
    if strong:
        f["strong"] = strong

    # ---- weak signals ----
    if OPENERS_RE.search(body):
        weak.append("opener")
    if META_RE.search(body):
        weak.append("meta_commentary")
    if PRAISE_RE.search(body):
        weak.append("praise_opener")
    if EMOJI_RE.search(body) and not EMOJI_RE.search(prompt):
        weak.append("emoji")
    pyou = _count(YOU_RE, prompt)
    cyou = _count(YOU_RE, body)
    f["you_rate"] = round(100.0 * cyou / max(len(words), 1), 1)
    if pyou == 0 and cyou >= 2 and f["you_rate"] >= 1.5:
        weak.append("second_person_introduced")
    if pcode and len(words) >= 15 and not _cont_is_codelike(body):
        weak.append("code_prompt_prose_reply")
    if weak:
        f["weak"] = weak

    # ---- document-continuation evidence (recorded for audit) ----
    doc = []
    if pcode and _cont_is_codelike(body):
        doc.append("more_code")
    if NAME_LINE_RE.search(prompt) and NAME_LINE_RE.search(body):
        doc.append("more_script_lines")
    if wiki:
        doc.append("wiki_markup")
    if FORUM_SCAFFOLD_RE.search(body) and not FORUM_SCAFFOLD_RE.search(prompt):
        doc.append("forum_scaffold")
    if doc:
        f["doc"] = doc

    if strong:
        return "ASSISTANT", f
    if len(weak) >= 2 and not doc:  # weak signals decide only when nothing says "still the document"
        return "ASSISTANT", f
    return "DOCUMENT", f


def split_think(text):
    """Split a raw-completion string into (reasoning, visible) if it contains a <think> block."""
    if text is None:
        return "", ""
    m = re.search(r"<think>(.*?)(?:</think>|$)", text, flags=re.S)
    if m:
        return m.group(1), (text[:m.start()] + text[m.end():])
    if "</think>" in text:  # template pre-opened the think block, so only the closing tag appears
        head, _, tail = text.partition("</think>")
        return head, tail
    return "", text


if __name__ == "__main__":
    import sys, json
    data = json.load(sys.stdin)
    print(json.dumps(classify(data["prompt"], data["continuation"]), indent=1))
