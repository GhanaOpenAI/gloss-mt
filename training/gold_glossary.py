"""Training terms that agree with the gold translation (no phrase extraction needed).

Dictionary terms agree with the human reference only ~31% of the time, so a model trained on
dictionary-pretranslated inputs learns to overrule the inserted terms. Each source example of
ghanaopenai/twi-translation-tool-calling already lists the noun phrases it looked up
(tool_phrases) and a reasoning trace, written with the gold translation in view, that quotes
how they were translated ('English' -> 'Twi'). For a chunk, every tool phrase that occurs in it
gets its Twi from:

  1. the reasoning trace's pair for that phrase, if that Twi is in the gold chunk;
  2. else a dictionary translation of it (single-translation dictionary, then the
     multi-candidate one), if that is in the gold chunk.

Chunks are line-aligned, so the Twi must be on the same line of the gold chunk as the phrase
is in the English one (this rejects reasoning pairs that quote two neighbouring sentences).
Phrases with neither stay English, so the model learns: Twi in the input is kept, English is
translated. Validation and inference use plain dictionary lookups (glossary_for) instead.
"""

import re
from typing import Dict, List, Optional, Tuple

from twi_translate import TwiLookup
from twi_translate.seq2seq import split_lines

# a quoted span: the quotes are not inside a word, so Twi elisions like n'anidasoɔ survive
_QUOTED = re.compile(r"(?<!\w)'(.+?)'(?!\w)")
_CONNECTOR = re.compile(r"\b(?:as|is|are|was|to|into|becomes?|rendered|translated|maps?|uses?|gives?|means?)\b"
                        r"|->|→|=|:", re.IGNORECASE)
_LEADING = re.compile(r"^(?:the|a|an|this|that|these|those|his|her|its|their|our|your|my)\s+", re.IGNORECASE)


def phrase_key(text: str) -> str:
    """Lowercase, hyphens as spaces, no punctuation or leading article/determiner: how the
    tool phrases and dictionary keys are written."""
    return _LEADING.sub("", " ".join(re.sub(r"[^\w\s']", " ", text.replace("-", " ")).split())).lower()


def contains(text: str, phrase: str) -> Optional[str]:
    """The phrase as written in text (case-insensitive, whole words, spaces or hyphens between
    words), else None."""
    words = phrase.replace("-", " ").split()
    if not words:
        return None
    m = re.search(r"(?<!\w)" + r"[\s-]+".join(map(re.escape, words)) + r"(?!\w)", text, re.IGNORECASE)
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


def aligned(en: str, tw: str, phrase: str, twi: str) -> Optional[str]:
    """twi as written in the gold chunk tw, on a line whose English line contains phrase."""
    en_lines, tw_lines = split_lines(en)[0], split_lines(tw)[0]
    if len(en_lines) != len(tw_lines):
        return contains(tw, twi) if contains(en, phrase) else None
    for e, t in zip(en_lines, tw_lines):
        if contains(e, phrase):
            found = contains(t, twi)
            if found:
                return found
    return None


def gold_glossary(en: str, tw: str, tool_phrases: List[str], pairs: List[Tuple[str, str]],
                  lookup: TwiLookup, multi: Optional[TwiLookup] = None) -> List[Dict]:
    """Glossary results for the tool phrases that occur in chunk en, each with a Twi that the
    gold chunk tw uses on the same line; phrases without one are left out (they stay English)."""
    from_reasoning = {}
    for a, b in pairs:
        from_reasoning.setdefault(phrase_key(a), []).append(b)
    out, seen = [], set()
    for phrase in sorted(set(tool_phrases), key=len, reverse=True):
        surface, key = contains(en, phrase), phrase_key(phrase)
        if not surface or key in seen:
            continue
        twi, source = None, "reasoning"
        for b in from_reasoning.get(key, []):
            if not contains(en, b):
                twi = aligned(en, tw, phrase, b)
                if twi:
                    break
        if not twi:
            source = "dictionary"
            candidates = lookup.lookup(key)["translations"] + (multi.lookup(key)["translations"] if multi else [])
            twi = next((t for t in (aligned(en, tw, phrase, c) for c in candidates) if t), None)
        if twi and twi.lower() != surface.lower():
            seen.add(key)
            out.append({"phrase": surface, "translations": [twi], "found": True, "source": source})
    return out
