"""Pick the tool-call phrases for an example from its gold reasoning trace.

The reasoning in ghananlpcommunity/twi-english-reasoning-translation quotes the English
phrases it translates, e.g.  'national progress' is 'ɔman nkɔso'.  Those quoted spans are
exactly the terms a translator needed to look up, so they become the tool call:

  1. take every single-quoted span in the <think> block
  2. keep spans that occur in the English source text (word-boundary match) and contain
     no Twi letters (ɛ, ɔ)
  3. keep only nominal spans (spaCy root is NOUN/PROPN; single tokens must be NOUN) of at
     most 4 words, with leading articles/determiners/possessives removed, lowercased
"""

import re
from typing import Iterable, List, Tuple

import spacy

LEADING_DROP = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "each", "every", "such", "no", "another",
}
MAX_WORDS = 4

_QUOTE = re.compile(r"(?<![A-Za-z])'([^'\n]{2,80}?)'(?![A-Za-z])")
_TWI = re.compile(r"[ɛɔƐƆ]")
_MARKDOWN = re.compile(r"[*#|_`>]")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z]{2,}")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def load_nlp(model: str = "en_core_web_sm"):
    return spacy.load(model, disable=["ner", "lemmatizer"])


def quoted_source_spans(reasoning: str, source_text: str) -> List[str]:
    """Quoted spans of the reasoning that literally occur in the English source."""
    src = " " + _norm(_MARKDOWN.sub(" ", source_text)) + " "
    out = []
    for q in _QUOTE.findall(reasoning):
        if _TWI.search(q) or "..." in q or "…" in q:
            continue
        nq = _norm(q)
        if nq in out or not re.search(r"(?<![a-z])" + re.escape(nq) + r"(?![a-z])", src):
            continue
        out.append(nq)
    return out


def clean_nominal(doc) -> str:
    """Return the cleaned phrase if the parsed span is a noun phrase, else ''."""
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    if not toks or doc[:].root.pos_ not in ("NOUN", "PROPN"):
        return ""
    if len(toks) == 1 and toks[0].pos_ != "NOUN":
        return ""
    while toks and (toks[0].lower_ in LEADING_DROP or toks[0].dep_ in ("det", "poss", "predet")
                    or toks[0].tag_ == "PRP$"):
        toks = toks[1:]
    if not toks or len(toks) > MAX_WORDS:
        return ""
    phrase = _norm(" ".join(t.text for t in toks)).replace(" 's", "").replace(" ’s", "")
    return phrase if len(phrase) >= 3 and _WORD.search(phrase) else ""


def phrases_for_examples(nlp, items: Iterable[Tuple[str, str]], batch_size: int = 512) -> List[List[str]]:
    """items: (reasoning, english_source) pairs -> list of tool-call phrase lists."""
    spans_per_item = [quoted_source_spans(r, s) for r, s in items]
    flat = [s for spans in spans_per_item for s in spans]
    cleaned = [clean_nominal(d) for d in nlp.pipe(flat, batch_size=batch_size)]
    out, i = [], 0
    for spans in spans_per_item:
        phrases = []
        for p in cleaned[i:i + len(spans)]:
            if p and p not in phrases:
                phrases.append(p)
        out.append(phrases)
        i += len(spans)
    return out
