"""Encoder-decoder (umT5) variant of the Twi translator.

Seq2seq models have no chat template or native tool calling, so the tool loop of the
decoder-only runtime becomes a fixed pipeline in code, and the model only translates:

  1. split the English text into lines and pack consecutive lines into chunks of at most
     `chunk_tokens` source tokens (umT5 degrades well past its 1024-token pretraining length)
  2. extract each chunk's noun phrases with spaCy (twi_translate.extractor, nkrane-style)
  3. look them up in the offline dictionary (TwiLookup, same as the LLM tool)
  4. replace every dictionary phrase in the chunk with its Twi translation in angle brackets,
     in place (pretranslate: "the <ɔman> has no <tumi>"); phrases not in the dictionary stay
     English. Translate the mixed chunk: the model writes the terms in brackets too, and
     clean_output removes them
  5. join the Twi chunks with the English separators

Training data (scripts/build_chunked_dataset.py) uses the same chunking and phrase lookup (glossary_for).
The same pipeline (ChunkedTranslator) drives NLLB here and TranslateGemma in translategemma.py.

The SentencePiece normaliser collapses newlines into spaces, so "\\n" is registered as an
added token (prepare_tokenizer) to keep the markdown line structure inside a chunk.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from .extractor import extract_noun_phrases
from .lookup import DEFAULT_PATH, TwiLookup

# One translation per phrase (scripts/rebuild_dictionary_single.py); the canonical multi-candidate
# twi_dictionary.tsv.gz stays the default of the Gemma tool-calling runtime.
GLOSSARY_DICT = os.path.join(os.path.dirname(DEFAULT_PATH), "twi_dictionary_single.tsv.gz")
TRANSLATE_PREFIX = "translate English to Twi: "
MADLAD_PREFIX = "<2ak> "  # MADLAD-400 picks the target language with a "<2xx>" token; Twi is Akan

_SPLIT = re.compile(r"(\n\s*)")
_NL_SPACES = re.compile(r" *\n *")
_INSERTED = re.compile(r"<([^<>\n]{1,120})>")  # a pretranslated term, as pretranslate writes it
_STRAY = re.compile(r"<(?=\S)|(?<=\S)>")  # an unpaired bracket on a word
_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")
_MARKUP = re.compile(r"[|#*`<>\[\]{}=~]|---")
_LEADERS = re.compile(r"(\s*(?:\.\s?){4,}\s*)")  # dot leaders of a table of contents
_PAGE_REF = re.compile(r"\s*[ivxlcdm\d][ivxlcdm\d\s.,–-]*", re.IGNORECASE)


def prepare_tokenizer(tok) -> int:
    """Register "\\n" as a literal token; returns the number of tokens added."""
    from tokenizers import AddedToken

    return tok.add_tokens([AddedToken("\n", normalized=False, special=False)])


def nllb_langs(tok) -> Optional[Tuple[str, str]]:
    """(src, tgt) language codes when the tokenizer is NLLB's, else None."""
    return ("eng_Latn", "twi_Latn") if "nllb" in type(tok).__name__.lower() else None


def is_madlad(tok) -> bool:
    """True for MADLAD-400's tokenizer, the only one here with "<2xx>" target-language tokens."""
    return tok.convert_tokens_to_ids("<2ak>") not in (None, tok.unk_token_id)


def task_prefix(tok) -> str:
    """The text prefix the model's inputs start with: MADLAD-400 takes a target-language token,
    umT5 (another T5 tokenizer) is told the task in text; NLLB takes language codes and an
    OPUS-MT (Marian) model is a single language pair."""
    if is_madlad(tok):
        return MADLAD_PREFIX
    return TRANSLATE_PREFIX if "t5" in type(tok).__name__.lower() else ""


def default_chunk_tokens(tok) -> int:
    """192 source tokens per chunk, or a quarter of the model's length limit when that is smaller
    (Marian, 512 -> 128): pretranslated Twi terms make the input longer than the plain English."""
    limit = tok.model_max_length if tok.model_max_length < 1e6 else 4096
    return min(192, limit // 4)


def load_seq2seq(path: str, **kwargs):
    """AutoModelForSeq2SeqLM, falling back to UMT5 for hub configs that lack `model_type`."""
    from transformers import AutoModelForSeq2SeqLM, UMT5ForConditionalGeneration

    try:
        return AutoModelForSeq2SeqLM.from_pretrained(path, **kwargs)
    except ValueError:
        return UMT5ForConditionalGeneration.from_pretrained(path, **kwargs)


def split_lines(text: str) -> Tuple[List[str], List[str]]:
    """Return the non-empty lines and the separator that follows each one ("" for the last)."""
    parts = _SPLIT.split(text.strip())
    lines, seps = [], []
    for i in range(0, len(parts), 2):
        if parts[i].strip():
            lines.append(parts[i].strip())
            sep = parts[i + 1] if i + 1 < len(parts) else ""
            seps.append("\n\n" if sep.count("\n") > 1 else ("\n" if sep else ""))
        elif seps:  # whitespace-only line: fold it into the previous separator
            seps[-1] = "\n\n"
    if seps:
        seps[-1] = ""
    return lines, seps


def pack_lines(lengths: List[int], max_tokens: int) -> List[Tuple[int, int]]:
    """Greedily group consecutive lines into [start, end) spans of at most max_tokens."""
    spans, start, total = [], 0, 0
    for i, n in enumerate(lengths):
        if i > start and total + n > max_tokens:
            spans.append((start, i))
            start, total = i, 0
        total += n
    if lengths:
        spans.append((start, len(lengths)))
    return spans


def join_span(lines: List[str], seps: List[str], start: int, end: int) -> str:
    return "".join(lines[i] + (seps[i] if i < end - 1 else "") for i in range(start, end))



def pretranslate(chunk: str, results: List[Dict], recase: bool = True) -> str:
    """Replace every occurrence of each found phrase in the chunk with its Twi translation in
    angle brackets, "<twi>" (case-insensitive, whole words, longest phrase first, one pass so a
    translation is never replaced again). The translation takes the match's casing: all caps,
    capitalised (sentence start or a proper noun), or lowercase; recase=False inserts it exactly
    as given. Phrases without a translation, or not verbatim in the chunk, stay English."""
    table = {" ".join(r["phrase"].lower().split()): r["translations"][0] for r in results if r["translations"]}
    if not table:
        return chunk
    alts = [r"\s+".join(map(re.escape, k.split())) for k in sorted(table, key=len, reverse=True)]
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE)

    def sub(m):
        src, twi = m.group(0), table[" ".join(m.group(0).lower().split())]
        if not recase:
            return f"<{twi}>"
        if sum(c.isalpha() for c in src) > 1 and src.isupper():
            return f"<{twi.upper()}>"
        return "<" + (twi[:1].upper() if src[:1].isupper() else twi[:1].lower()) + twi[1:] + ">"

    return pattern.sub(sub, chunk)


def translate_input(chunk: str, results: Optional[List[Dict]], prefix: str = TRANSLATE_PREFIX) -> str:
    """Model input: [prefix]chunk with its dictionary phrases pretranslated in place;
    results=None leaves the chunk as is."""
    return prefix + (pretranslate(chunk, results) if results else chunk)


def source_casing(phrase: str, chunk: str) -> str:
    """The phrase as first written in the chunk (the extractor lowercases). When it is not in the
    chunk verbatim (the extractor dropped "'s", hyphens or stopwords), each word takes the casing
    of its first occurrence instead."""
    def find(words):
        return re.search(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", chunk, re.IGNORECASE)

    m = find(phrase.split())
    if m:
        return " ".join(m.group(0).split())
    return " ".join(m.group(0) if (m := find([w])) else w for w in phrase.split())


def glossary_for(chunk: str, lookup: TwiLookup) -> List[Dict]:
    """Dictionary results for the chunk's spaCy noun phrases. Only found phrases are pretranslated;
    a phrase not in the dictionary is kept with found=False and no translations (it stays English in
    the chunk, for the model to translate), as written in the chunk. Not-found "phrases" that are
    markdown residue (table pipes, rules) or have no letters are dropped."""
    results = []
    for r in lookup.lookup_many(extract_noun_phrases(chunk)):
        r["found"] = bool(r["translations"])
        if not r["found"]:
            if _MARKUP.search(r["phrase"]) or not re.search(r"[^\W\d_]", r["phrase"]):
                continue
            r["phrase"] = source_casing(r["phrase"], chunk)
        results.append(r)
    return results


def clean_output(text: str) -> str:
    """Normalise line breaks and remove the term brackets the model writes ("<ɔman>" -> "ɔman"):
    unwrap each bracketed term, then drop any unpaired "<" or ">" left on a word, and the double
    spaces a term bracketed with inner spaces leaves. A free-standing "<"/">" ("x < 5") is kept."""
    text = _INSERTED.sub(lambda m: m.group(1).strip(), _NL_SPACES.sub("\n", text))
    text = _STRAY.sub("", text)
    return _SPACES.sub(" ", text).strip()


def protect_leaders(text: str) -> List[Tuple[str, bool]]:
    """Split text around dot leaders (4+ dots, with their surrounding spaces) into
    (segment, translate?) pairs; leaders and segments with no words to translate (page numbers,
    roman numerals) are kept as they are."""
    parts = _LEADERS.split(text)
    if len(parts) == 1:
        return [(text, bool(re.search(r"[^\W\d_]", text)))]
    return [(seg, i % 2 == 0 and bool(re.search(r"[^\W\d_]", seg)) and not (i and _PAGE_REF.fullmatch(seg)))
            for i, seg in enumerate(parts) if seg]


def chunk_text(text: str, tok, max_tokens: int) -> Tuple[List[str], List[str]]:
    """Chunk a document; returns the chunks and the separator that follows each one."""
    lines, seps = split_lines(text)
    lengths = [len(tok(line, add_special_tokens=False).input_ids) for line in lines]
    spans = pack_lines(lengths, max_tokens)
    return [join_span(lines, seps, s, e) for s, e in spans], [seps[e - 1] for _, e in spans]


class ChunkedTranslator:
    """Chunk -> spaCy phrases -> lookup -> translate each chunk -> re-join. Subclasses implement
    `encode` (chunk + lookup results -> model input) and `generate` (batched inputs -> outputs).
    With glossary=False the dictionary is skipped (zero-shot baselines)."""

    def __init__(self, tok, lookup: Optional[TwiLookup] = None, chunk_tokens: Optional[int] = None,
                 glossary: bool = True):
        self.tok, self.glossary = tok, glossary
        self.chunk_tokens = chunk_tokens or default_chunk_tokens(tok)
        self.lookup = (lookup or TwiLookup(GLOSSARY_DICT)) if glossary else None

    def encode(self, chunk: str, results: Optional[List[Dict]]):
        raise NotImplementedError

    def generate(self, inputs: List) -> List[str]:
        raise NotImplementedError

    def translate_many(self, texts: List[str]) -> List[Dict]:
        """Translate texts; dot leaders ("Contents ........ 23") are kept out of the model, which
        loops on them: only the text around them is translated (protect_leaders)."""
        parts = [protect_leaders(t) for t in texts]
        segs = list(dict.fromkeys(s for p in parts for s, translate in p if translate))
        done = dict(zip(segs, self._translate_texts(segs)))
        out = []
        for p in parts:
            hits = [done[s] for s, translate in p if translate]
            out.append({
                "translation": "".join(done[s]["translation"] if translate else s for s, translate in p).strip(),
                "phrases": [x for h in hits for x in h["phrases"]],
                "lookup_results": [x for h in hits for x in h["lookup_results"]],
            })
        return out

    def _translate_texts(self, texts: List[str]) -> List[Dict]:
        docs = [chunk_text(t, self.tok, self.chunk_tokens) for t in texts]
        flat = [c for chunks, _ in docs for c in chunks]
        results = [glossary_for(c, self.lookup) if self.glossary else [] for c in flat]
        twi = self.generate([self.encode(c, r if self.glossary else None) for c, r in zip(flat, results)])
        out, k = [], 0
        for chunks, seps in docs:
            n = len(chunks)
            out.append({
                "translation": "".join(twi[k + i] + seps[i] for i in range(n)).strip(),
                "phrases": [r["phrase"] for rs in results[k:k + n] for r in rs],
                "lookup_results": [r for rs in results[k:k + n] for r in rs],
            })
            k += n
        return out

    def translate(self, text: str) -> Dict:
        return self.translate_many([text])[0]


class Seq2SeqTranslator(ChunkedTranslator):
    """Encoder-decoder backend: fine-tuned umT5 (task prefix), NLLB (language codes) or OPUS-MT
    (Marian, no prefix or codes). An NLLB base model with glossary=False is the zero-shot NLLB baseline."""

    def __init__(self, model: str, lookup: Optional[TwiLookup] = None, device: Optional[str] = None,
                 chunk_tokens: Optional[int] = None, max_new_tokens: int = 1024, num_beams: int = 1,
                 batch_size: int = 16, glossary: bool = True):
        import torch
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
        super().__init__(tok, lookup, chunk_tokens, glossary)
        self.torch = torch
        self.langs = nllb_langs(tok)
        if self.langs:
            tok.src_lang, tok.tgt_lang = self.langs
        self.prefix = task_prefix(tok)
        self.max_len = min(1024, tok.model_max_length)  # Marian: 512 positions, input and output
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = load_seq2seq(model, dtype=dtype).to(self.device).eval()
        self.max_new_tokens, self.num_beams, self.batch_size = max_new_tokens, num_beams, batch_size

    def encode(self, chunk: str, results: Optional[List[Dict]]) -> str:
        return translate_input(chunk, results, self.prefix)

    def generate(self, inputs: List[str]) -> List[str]:
        """Batched greedy/beam generation, longest inputs first to limit padding."""
        extra = {"forced_bos_token_id": self.tok.convert_tokens_to_ids(self.langs[1])} if self.langs else {}
        order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i]))
        out = [""] * len(inputs)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            enc = self.tok([inputs[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len).to(self.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=min(self.max_new_tokens, self.max_len - 2),
                                          num_beams=self.num_beams,
                                          do_sample=False, **extra)
            for i, g in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = clean_output(g)
        return out
