"""Run a fine-tuned Twi tool-calling model end to end.

The loop:
  1. system prompt + user text + tool schema -> model
  2. model emits lookup_twi_translation(phrases=[...])
  3. phrases are looked up in the local copy of AfriSpeech/shola-machine-translations
  4. results are sent back as a tool message (same template as in training)
  5. model reasons about grammar/idioms (thinking), then returns the Twi translation

Two interchangeable backends:
  TransformersBackend - loads the model in-process (GPU recommended); no server needed
  OpenAIBackend       - any OpenAI-compatible server, e.g. for Gemma 4:
                        vllm serve <model> --enable-auto-tool-choice --tool-call-parser gemma4 \\
                            --reasoning-parser gemma4
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .lookup import TwiLookup
from .schema import SYSTEM_PROMPT, TOOL_NAME, TOOLS, USER_TEMPLATES, format_tool_result

ToolCall = Dict[str, Any]  # {"id": str, "name": str, "arguments": dict}

# Chat-template options used in training; the runtime must render prompts the same way.
# (Gemma 4: after the tool result the prompt opens the thought channel, so the model reasons
# before translating. Templates without this option ignore it.)
TEMPLATE_KWARGS = {"enable_thinking": True}

_HERMES_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)        # Qwen / Hermes
_GEMMA_CALL = re.compile(r"<\|tool_call>\s*call:([\w.\-]+)(.*?)<tool_call\|>", re.DOTALL)  # Gemma 4
_GEMMA_STR = '<|"|>'
_GEMMA_KEY = re.compile(r"([{,]\s*)([A-Za-z_]\w*)\s*:")
_TRAILING = ("<turn|>", "<|tool_response>", "<eos>", "<|im_end|>", "<|endoftext|>", "<end_of_turn>")


def _gemma_args(body: str) -> Dict[str, Any]:
    """Gemma 4 writes arguments as {key:<|"|>str<|"|>,...}; convert to JSON and parse."""
    out = []
    for i, part in enumerate(body.strip().split(_GEMMA_STR)):
        out.append(json.dumps(part) if i % 2 else _GEMMA_KEY.sub(r'\1"\2":', part))
    return json.loads("".join(out))


def parse_tool_calls(text: str) -> Tuple[str, List[ToolCall]]:
    """Extract tool calls (Hermes <tool_call>{json}</tool_call> or Gemma 4 <|tool_call>call:...)."""
    calls = []
    for raw in _HERMES_CALL.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": obj.get("name", ""), "arguments": args})
    for name, body in _GEMMA_CALL.findall(text):
        try:
            args = _gemma_args(body)
        except (json.JSONDecodeError, ValueError):
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": name, "arguments": args})
    content = _GEMMA_CALL.sub("", _HERMES_CALL.sub("", text))
    return content.strip(), calls


def split_reasoning(text: str) -> Tuple[str, str]:
    """Separate thinking from the answer: Gemma 4 '...<channel|>answer' or '<think>...</think>answer'."""
    for s in _TRAILING:
        text = text.replace(s, "")
    if "<channel|>" in text:
        reasoning, answer = text.split("<channel|>", 1)
        return reasoning.replace("<|channel>thought", "").strip(), answer.strip()
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "").strip(), answer.strip()
    # Numbered reasoning traces when special channel tokens were stripped
    lines = text.strip().split("\n")
    if lines and re.match(r"^\d+\.\s+", lines[0]):
        reasoning_lines, answer_lines = [], []
        is_reasoning = True
        for line in lines:
            if is_reasoning and re.match(r"^\d+\.\s+", line):
                reasoning_lines.append(line)
            else:
                is_reasoning = False
                answer_lines.append(line)
        answer = "\n".join(answer_lines).strip()
        reasoning = "\n".join(reasoning_lines).strip()
        if not answer and reasoning_lines:
            m = re.search(r"translated (?:into Twi |as |in Twi as )*as\s+['\"]([^'\"]+)['\"]", reasoning_lines[0], re.I)
            if m:
                answer = m.group(1).strip()
            else:
                quotes = re.findall(r"['\"]([^'\"]+)['\"]", reasoning_lines[0])
                if len(quotes) >= 2:
                    answer = quotes[1].strip()
        return reasoning, answer
    return "", text.strip()


class TransformersBackend:
    """Loads the model with transformers and generates in-process."""

    def __init__(self, model: str, device: Optional[str] = None, max_new_tokens: int = 2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.model.generation_config.do_sample = False
        self.max_new_tokens = max_new_tokens

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]], temperature: float = 0.0):
        """Returns (answer, tool_calls, reasoning)."""
        prompt = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                              tokenize=False, **TEMPLATE_KWARGS)
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id}
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with self.torch.inference_mode():
            out = self.model.generate(ids, **gen_kwargs)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        text, calls = parse_tool_calls(text)
        reasoning, answer = split_reasoning(text)
        return answer, calls, reasoning


class OpenAIBackend:
    """Talks to an OpenAI-compatible endpoint (vLLM, SGLang, TGI, ...)."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", max_tokens: int = 2048):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _to_api(messages: List[Dict]) -> List[Dict]:
        out = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["function"]["name"],
                                                 "arguments": json.dumps(c["function"]["arguments"], ensure_ascii=False)}}
                                   for c in m["tool_calls"]]
            out.append(m)
        return out

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]], temperature: float = 0.0):
        """Returns (answer, tool_calls, reasoning)."""
        kwargs = {"model": self.model, "messages": self._to_api(messages), "temperature": temperature,
                  "max_tokens": self.max_tokens,
                  "extra_body": {"chat_template_kwargs": TEMPLATE_KWARGS}}
        if tools:
            kwargs["tools"] = tools
        msg = self.client.chat.completions.create(**kwargs).choices[0].message
        calls = []
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": c.id, "name": c.function.name, "arguments": args})
        content = msg.content or ""
        if not calls and "<tool_call>" in content:  # server started without a tool parser
            content, calls = parse_tool_calls(content)
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        if not reasoning:  # server started without a reasoning parser
            reasoning, content = split_reasoning(content)
        return content.strip(), calls, reasoning.strip()


class TwiTranslator:
    def __init__(self, backend, lookup: Optional[TwiLookup] = None, max_tool_rounds: int = 2):
        self.backend = backend
        self.lookup = lookup or TwiLookup()
        self.max_tool_rounds = max_tool_rounds

    def run_tool(self, call: ToolCall) -> str:
        if call["name"] != TOOL_NAME:
            return f"Error: unknown tool '{call['name']}'."
        phrases = call["arguments"].get("phrases") or []
        if isinstance(phrases, str):
            phrases = [phrases]
        return format_tool_result(self.lookup.lookup_many([str(p) for p in phrases]))

    def translate(self, text: str, temperature: float = 0.0, verbose: bool = False) -> Dict[str, Any]:
        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATES[0].format(text=text)},
        ]
        lookups = []
        for _ in range(self.max_tool_rounds):
            content, calls, reasoning = self.backend.chat(messages, TOOLS, temperature)
            if not calls:
                return {"translation": content, "reasoning": reasoning, "lookups": lookups, "messages": messages}
            messages.append({
                "role": "assistant", "content": content,
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls],
            })
            for c in calls:
                result = self.run_tool(c)
                lookups.append({"call": c, "result": result})
                if verbose:
                    print(f"[tool call] {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})")
                    print(f"[tool result]\n{result}\n")
                messages.append({"role": "tool", "tool_call_id": c["id"], "name": TOOL_NAME, "content": result})
        # Tool budget used up: ask for the final answer without offering tools.
        content, _, reasoning = self.backend.chat(messages, None, temperature)
        return {"translation": content, "reasoning": reasoning, "lookups": lookups, "messages": messages}
