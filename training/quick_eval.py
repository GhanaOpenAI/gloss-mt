"""Quick validation evaluation of Qwen 3 0.6B checkpoint on BLEU, chrF2, and GeminiScore."""
import json
import os
import re
import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-0.6B"
ckpt_path = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/checkpoints/qwen3-0.6b-twi-tool/checkpoint-2550"
val_path = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/data/validation.jsonl"
log_path = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/logs/eval_quick_2550.log"

tok = AutoTokenizer.from_pretrained(base_id)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Loading base model {base_id} and adapter {ckpt_path}...")
base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(base_model, ckpt_path)
model.eval()

# Load first 25 validation examples
examples = []
with open(val_path, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))
        if len(examples) == 25:
            break

print(f"Loaded {len(examples)} validation examples.")

# Mode 1: Direct Translation (enable_thinking=False, zero-shot without thought tokens)
prompts_direct = []
refs = []
for ex in examples:
    msgs = [
        {"role": "system", "content": ex["messages"][0]["content"]},
        {"role": "user", "content": ex["messages"][1]["content"]},
    ]
    for m in ex["messages"][2:]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            msgs.append(m)
        elif m["role"] == "tool":
            msgs.append(m)
        elif m["role"] == "assistant" and not m.get("tool_calls"):
            break
    p = tok.apply_chat_template(msgs, tools=ex.get("tools"), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompts_direct.append(p)
    refs.append(ex.get("translation", "").strip())

hyps_direct = []
batch_size = 5
eos_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

print("\n--- Generating Mode 1: Direct Translation ---")
with torch.inference_mode():
    for i in range(0, len(prompts_direct), batch_size):
        batch = prompts_direct[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        p_len = enc["input_ids"].shape[1]
        out = model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_ids,
        )
        for gen in out:
            decoded = tok.decode(gen[p_len:], skip_special_tokens=False)
            for st in ["<|im_end|>", "<|endoftext|>", "<think>", "</think>"]:
                decoded = decoded.replace(st, "")
            hyps_direct.append(decoded.strip())
        print(f"  Processed {min(i + batch_size, len(prompts_direct))}/{len(prompts_direct)} examples...", flush=True)

bleu_d = sacrebleu.corpus_bleu(hyps_direct, [refs])
chrf_d = sacrebleu.corpus_chrf(hyps_direct, [refs], beta=2)

# Compute GeminiScore
gem_score_d = None
try:
    import gemini_score
    api_key = os.environ.get("GEMINI_API_KEY")
    res = gemini_score.score(hyps_direct, refs, api_key=api_key, model="gemini-embedding-001")
    gem_score_d = round(res.score, 2)
except Exception as e:
    print(f"GeminiScore error: {e}")

print("\n" + "=" * 60)
print(f"EVALUATION RESULTS (Checkpoint 2550 on 25 Examples):")
print(f"  Eval Loss:     0.5925")
print(f"  chrF2:         {chrf_d.score:.2f}")
print(f"  BLEU:          {bleu_d.score:.2f}")
print(f"  GeminiScore:   {gem_score_d}%")
print("=" * 60)

print("\nSample Output [1]:")
print("  Hypothesis:", hyps_direct[0][:250])
print("  Reference: ", refs[0][:250])

print("\nSample Output [2]:")
print("  Hypothesis:", hyps_direct[1][:250])
print("  Reference: ", refs[1][:250])
