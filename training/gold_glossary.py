"""Training glossaries whose Twi terms agree with the gold translation.

Dictionary terms agree with the human reference only ~31% of the time, so a model trained on
dictionary-pretranslated inputs learns to overrule the inserted terms. Here a training chunk
gets the same spaCy noun phrases as at inference (glossary_for), but each phrase's Twi is
taken from what its gold Twi chunk actually uses:

  1. an 'English' -> 'Twi' pair quoted in the example's reasoning trace (written with the gold
     translation in view) whose English is the phrase, if that Twi is in the gold chunk;
  2. else the phrase's dictionary translation (the single-translation dictionary, then any
     candidate of the multi-candidate one), if it is in the gold chunk.

Phrases with neither stay English, so the model learns: Twi in the input is kept, English is
translated. Validation and inference keep using plain dictionary lookups (glossary_for).
"""

import re
from typing import Dict, List, Optional, Tuple

from twi_translate import TwiLookup
from twi_translate.seq2seq import glossary_for

# a quoted span: the quotes are not inside a word, so Twi elisions like n'anidasoɔ survive
_QUOTED = re.compile(r"(?<!\w)'(.+?)'(?!\w)")
_CONNECTOR = re.compile(r"\b(?:as|is|are|was|to|into|becomes?|rendered|translated|maps?|uses?|gives?|means?)\b"
                        r"|->|→|=|:", re.IGNORECASE)
_LEADING = re.compile(r"^(?:the|a|an|this|that|these|those|his|her|its|their|our|your|my)\s+", re.IGNORECASE)


def phrase_key(text: str) -> str:
    """Lowercase, no leading article/determiner, single spaces: how dictionary phrases are keyed."""
    return _LEADING.sub("", " ".join(re.sub(r"[^\w\s'-]", " ", text).split())).lower()


def contains(text: str, phrase: str) -> Optional[str]:
    """The phrase as written in text (case-insensitive, whole words, any spacing), else None."""
    words = phrase.split()
    if not words:
        return None
    m = re.search(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", text, re.IGNORECASE)
    return m.group(0) if m else None


def reasoning_pairs(reasoning: str) -> List[Tuple[str, str]]:
    """'English' ... 'Twi' pairs of a reasoning trace: two consecutive quoted spans joined by a
    short translation connector ("as", "is rendered as", "->", ...), not by a list comma."""
    quotes = list(_QUOTED.finditer(reasoning or ""))
    pairs = []
    for x, y in zip(quotes, quotes[1:]):
        between = reasoning[x.end():y.start()]
        if len(between) <= 45 and _CONNECTOR.search(between) and x.group(1).strip() and y.group(1).strip():
            pairs.append((x.group(1).strip(), y.group(1).strip()))
    return pairs


def gold_glossary(en: str, tw: str, pairs: List[Tuple[str, str]], lookup: TwiLookup,
                  multi: Optional[TwiLookup] = None) -> List[Dict]:
    """The chunk's spaCy phrases (as glossary_for) with a Twi used by the gold chunk tw;
    phrases without one are left out (they stay English)."""
    from_reasoning = {}
    for a, b in pairs:
        twi = contains(tw, b)
        if twi and not contains(en, b):
            from_reasoning.setdefault(phrase_key(a), twi)
    out = []
    for g in glossary_for(en, lookup):
        twi, source = from_reasoning.get(phrase_key(g["phrase"])), "reasoning"
        if not twi:
            source = "dictionary"
            candidates = (g["translations"] if g["found"] else []) + (
                multi.lookup(g["phrase"])["translations"] if multi else [])
            twi = next((t for t in map(lambda c: contains(tw, c), candidates) if t), None)
        if twi and twi.lower() != g["phrase"].lower():
            out.append({"phrase": g["phrase"], "translations": [twi], "found": True, "source": source})
    return out
