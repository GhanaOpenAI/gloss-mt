"""Offline tests (no Hub downloads): phrase cleaning, tool-result template, example structure.

Run: python -m pytest tests -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from twi_translate.lookup import TwiLookup, normalize_phrase  # noqa: E402
from twi_translate.schema import LOOKUP_TOOL, TOOL_NAME, format_tool_result  # noqa: E402


class FakeLookup(TwiLookup):
    def __init__(self, table):
        self.table = table


@pytest.fixture(scope="module")
def nlp():
    from dataset_builder.phrases import load_nlp
    return load_nlp()


def test_phrases_from_reasoning(nlp):
    from dataset_builder.phrases import phrases_for_examples
    source = ("## National Progress\n\nOur politics affects **the national progress** of children. "
              "If we are speaking of technology, it means a lot.")
    reasoning = ("1. 'Our politics' is 'yɛn amanyɔsɛm'. 'national progress' is 'ɔman nkɔso'. "
                 "'are speaking' uses 're-'. 'ɔman' means nation. 'na' marks focus. "
                 "'quantum flux' is not in the source. 'If... it means' maps to 'Sɛ... a'.")
    [phrases] = phrases_for_examples(nlp, [(reasoning, source)])
    assert phrases == ["politics", "national progress"]


def test_lookup_found_and_not_found():
    lk = FakeLookup({"cylinder": ["silinda"], "war": ["ɔko", "sa"]})
    res = lk.lookup_many(["Cylinder", "war", "  quantum flux ", "cylinder"])
    assert res == [
        {"phrase": "cylinder", "translations": ["silinda"]},
        {"phrase": "war", "translations": ["ɔko", "sa"]},
        {"phrase": "quantum flux", "translations": []},
    ]
    assert normalize_phrase("  Road   Construction ") == "road construction"


def test_tool_result_template():
    out = format_tool_result([
        {"phrase": "cylinder", "translations": ["silinda"]},
        {"phrase": "quantum flux", "translations": []},
    ])
    assert out.splitlines()[0] == "Twi lookup results (1 of 2 phrases found):"
    assert "- cylinder -> silinda" in out
    assert out.endswith("Not found (translate from your own knowledge): quantum flux")
    empty = format_tool_result([{"phrase": "x y", "translations": []}])
    assert "(no translations found)" in empty


def test_build_example_structure():
    from dataset_builder.build_dataset import build_example
    lk = FakeLookup({"court": ["asennibea"]})
    row = {"id": "x1", "source": "pristine", "markdown": None, "text": "Did you hear the matter in court?",
           "reasoning": "1. 'court' is 'asɛnnibea'.", "translation": "Wote asɛm no?"}
    ex = build_example(row, ["matter", "court"], lk, 7)
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    call = ex["messages"][2]["tool_calls"][0]
    assert call["function"]["name"] == TOOL_NAME
    assert call["function"]["arguments"] == {"phrases": ["matter", "court"]}
    assert ex["messages"][3]["tool_call_id"] == call["id"]
    assert ex["messages"][-1] == {"role": "assistant", "reasoning_content": "1. 'court' is 'asɛnnibea'.",
                                  "content": "Wote asɛm no?"}
    assert ex["num_found"] == 1 and ex["num_phrases"] == 2
    assert ex["tools"] == [LOOKUP_TOOL]
    assert ex["tool_phrases"] == ["matter", "court"] and ex["tool_result"] == ex["messages"][3]["content"]
    assert ex["reasoning"] == "1. 'court' is 'asɛnnibea'." and ex["translation"] == "Wote asɛm no?"
    json.dumps(ex, ensure_ascii=False)

    # no phrases -> no tool call; reasoning + translation directly
    ex2 = build_example(row, [], lk, 8)
    assert [m["role"] for m in ex2["messages"]] == ["system", "user", "assistant"]


def test_parse_tool_calls():
    from twi_translate.runtime import parse_tool_calls
    raw = ('<tool_call>\n{"name": "lookup_twi_translation", "arguments": {"phrases": ["court"]}}\n</tool_call>')
    content, calls = parse_tool_calls(raw)
    assert content == ""
    assert calls[0]["name"] == TOOL_NAME and calls[0]["arguments"] == {"phrases": ["court"]}
    content, calls = parse_tool_calls("Wote asɛm no?")
    assert content == "Wote asɛm no?" and calls == []
    _, calls = parse_tool_calls("<tool_call>{not json}</tool_call>")
    assert calls == []


class _ScriptedBackend:
    """Plays back a tool call, then a translation; records what it was sent."""

    def __init__(self):
        self.seen = []

    def chat(self, messages, tools, temperature=0.0):
        self.seen.append([dict(m) for m in messages])
        if len(self.seen) == 1:
            return "", [{"id": "c1", "name": TOOL_NAME, "arguments": {"phrases": ["matter", "court"]}}], ""
        return "Wote asɛm no?", [], "1. 'court' -> 'asɛnnibea'."


def test_translator_loop_feeds_tool_results_back():
    from twi_translate.runtime import TwiTranslator
    backend = _ScriptedBackend()
    out = TwiTranslator(backend, FakeLookup({"court": ["asennibea"]})).translate("Did you hear the matter in court?")
    assert out["translation"] == "Wote asɛm no?"
    assert out["reasoning"].startswith("1.")
    second_call = backend.seen[1]
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "tool"]
    assert second_call[-1]["content"] == format_tool_result([
        {"phrase": "matter", "translations": []}, {"phrase": "court", "translations": ["asennibea"]}])


def test_openai_backend_serializes_arguments():
    from twi_translate.runtime import OpenAIBackend
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": TOOL_NAME, "arguments": {"phrases": ["ɔko"]}}}]}]
    api = OpenAIBackend._to_api(msgs)
    assert api[0]["tool_calls"][0]["function"]["arguments"] == '{"phrases": ["ɔko"]}'
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {"phrases": ["ɔko"]}


def test_parse_gemma4_tool_call_and_reasoning():
    from twi_translate.runtime import parse_tool_calls, split_reasoning
    raw = ('<|tool_call>call:lookup_twi_translation{phrases:[<|"|>cocoa heaps<|"|>,<|"|>a, b: c<|"|>]}'
           '<tool_call|><|tool_response>')
    content, calls = parse_tool_calls(raw)
    assert calls[0]["name"] == TOOL_NAME
    assert calls[0]["arguments"] == {"phrases": ["cocoa heaps", "a, b: c"]}
    assert split_reasoning(content) == ("", "")
    assert split_reasoning("1. 'ɛnsɛ sɛ' means should not.<channel|>Ɛnsɛ sɛ kokoo...<turn|>") == (
        "1. 'ɛnsɛ sɛ' means should not.", "Ɛnsɛ sɛ kokoo...")
    assert split_reasoning("<think>why</think>\n\nWote?") == ("why", "Wote?")
