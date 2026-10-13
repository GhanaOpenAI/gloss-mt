"""Multi-metric evaluation across top candidate checkpoints.

Evaluates:
- SacreBLEU (BLEU)
- chrF2 (character n-gram F2)
- GeminiScore (semantic embedding similarity via gemini-embedding-001)
- Length ratio (word hypothesis / word reference)
- Eval loss
"""
import glob
import json
import os
import re
import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import gemini_score

base_id = "Qwen/Qwen3-0.6B"
ckpts_dir = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/checkpoints/qwen3-0.6b-twi-tool"
val_path = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/data/validation.jsonl"
api_key = os.environ.get("GEMINI_API_KEY")

target_steps = [2700, 3300, 3600, 3750, 3900, 4050]

tok = AutoTokenizer.from_pretrained(base_id)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Loading base model {base_id}...")
base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map={"": 0})

# Load 50 representative validation examples
examples = []
with open(val_path, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))
        if len(examples) == 50:
            break

print(f"Loaded {len(examples)} validation examples.")

prompts = []
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
    prompts.append(p)
    refs.append(ex.get("translation", "").strip())

ref_words = sum(len(r.split()) for r in refs)
batch_size = 5
eos_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

results = []

for step in target_steps:
    ckpt_path = os.path.join(ckpts_dir, f"checkpoint-{step}")
    if not os.path.isdir(ckpt_path):
        continue

    # Get loss from trainer_state.json
    loss = None
    state_file = os.path.join(ckpt_path, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            st = json.load(f)
            for e in st.get("log_history", []):
                if e.get("step") == step and "eval_loss" in e:
                    loss = round(e["eval_loss"], 4)
                    break

    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.eval()

    hyps = []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
            p_len = enc["input_ids"].shape[1]
            out = model.generate(
                **enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=eos_ids
            )
            for gen in out:
                decoded = tok.decode(gen[p_len:], skip_special_tokens=False)
                for st in ["<|im_end|>", "<|endoftext|>", "<think>", "</think>"]:
                    decoded = decoded.replace(st, "")
                hyps.append(decoded.strip())

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], beta=2).score
    hyp_words = sum(len(h.split()) for h in hyps)
    lr = hyp_words / max(1, ref_words)
    gem = None
    if api_key:
        try:
            gem = gemini_score.score(hyps, refs, api_key=api_key, model="gemini-embedding-001").score
        except Exception:
            pass

    results.append({
        "step": step,
        "loss": loss,
        "bleu": round(bleu, 2),
        "chrf2": round(chrf, 2),
        "gemini_score": round(gem, 2) if gem is not None else None,
        "length_ratio": round(lr, 3),
    })
    print(f"Step {step:4d} | Loss: {loss} | chrF2: {chrf:.2f} | BLEU: {bleu:.2f} | GemScore: {gem:.2f}% | LenRatio: {lr:.3f}", flush=True)

print("\n" + "=" * 80)
print(f"{'Checkpoint':<16} | {'Step':<6} | {'Loss':<8} | {'chrF2':<7} | {'BLEU':<6} | {'GemScore':<9} | {'LenRatio':<8}")
print("-" * 80)
for r in results:
    gem_str = f"{r['gemini_score']:.2f}%" if r['gemini_score'] else "N/A"
    print(f"{'checkpoint-' + str(r['step']):<16} | {r['step']:<6} | {str(r['loss']):<8} | {r['chrf2']:<7.2f} | {r['bleu']:<6.2f} | {gem_str:<9} | {r['length_ratio']:<8.3f}")
print("=" * 80)

out_file = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/checkpoints/qwen3-0.6b-twi-tool/multi_metric_eval.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {out_file}")
