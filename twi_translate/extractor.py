"""Noun phrase extraction following GhanaNLP's nkrane library:
https://github.com/GhanaNLP/nkrane/tree/main

Extracts noun phrases from text using spaCy noun chunks, filtering out stopwords
and extracting content words to identify terminology for dictionary lookup.
"""

import re
from typing import List, Optional

_nlp = None


def get_nlp(model_name: str = "en_core_web_sm"):
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
        except Exception:
            # Fallback if spacy model is not downloaded
            try:
                _nlp = spacy.load("en_core_web_sm")
            except Exception:
                return None
    return _nlp


def extract_noun_phrases(text: str, nlp=None) -> List[str]:
    """
    Extract noun phrases from English text using spaCy, filtering stopwords,
    following the nkrane terminology extraction approach.
    """
    nlp = nlp or get_nlp()
    if nlp is None:
        # Fallback regex extraction of words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return list(dict.fromkeys(words))

    doc = nlp(text)
    noun_phrases: List[str] = []
    seen = set()

    for chunk in doc.noun_chunks:
        tokens = list(chunk)
        # Skip chunks where all tokens are stopwords
        if all(token.is_stop for token in tokens):
            continue

        # Filter out stopwords to retain content words (the core noun phrase)
        content_tokens = [token for token in tokens if not token.is_stop and not token.is_punct]
        if not content_tokens:
            continue

        raw = " ".join(t.text for t in content_tokens).strip().lower()
        cleaned = re.sub(r"\s*-\s*", "-", raw)
        cleaned = re.sub(r"\s*['’]\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if len(cleaned) >= 2 and cleaned not in seen and re.search(r"[a-z]{2,}", cleaned):
            seen.add(cleaned)
            noun_phrases.append(cleaned)

    return noun_phrases
