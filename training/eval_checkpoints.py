"""Evaluate fine-tuning checkpoints on BLEU and chrF2 on the validation set.

Loads the base model once and hot-swaps LoRA checkpoint adapters to evaluate:
  - BLEU (sacrebleu)
  - chrF2 (sacrebleu)
  - Length ratio
  - Evaluation loss (from trainer_state.json if available)

Usage:
  python training/eval_checkpoints.py \
      --base /mnt/volume_d2wey28/models/gemma-4-E4B-it \
      --checkpoints checkpoints/gemma4-e4b-twi-tool \
      --data repo/data/validation.jsonl \
      --limit 100  # or all 332
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple
import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPLATE_KWARGS = {"enable_thinking": True}
THOUGHT_RE = re.compile(r"<(?:thought|think)>.*?</(?:thought|think)>", re.DOTALL)


def extract_translation(text: str) -> str:
    """Extract translation by stripping thought tags or channels."""
    # 1. Strip <thought>...</thought> or <think>...</think> if present
    cleaned = THOUGHT_RE.sub("", text).strip()
    # 2. Strip Gemma 4 channel tokens if present
    if "<channel|>" in cleaned:
        cleaned = cleaned.split("<channel|>")[-1].strip()
    for stop_token in ["<turn|>", "<eos>", "<|im_end|>", "<|endoftext|>", "<end_of_turn>"]:
        cleaned = cleaned.replace(stop_token, "").strip()
    return cleaned.strip()


def load_val_examples(data_path: str, limit: int = -1) -> List[Dict]:
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append(ex)
            if 0 < limit <= len(examples):
                break
    return examples


def build_prompt_and_ref(ex: Dict, tok) -> Tuple[str, str]:
    """Prepare teacher-forced input prompt (system + user + tool calls + tool result) and reference translation."""
    messages = [
        {"role": "system", "content": ex["messages"][0]["content"]},
        {"role": "user", "content": ex["messages"][1]["content"]},
    ]
    # Include tool call and tool result turns if present
    for m in ex["messages"][2:]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            messages.append(m)
        elif m["role"] == "tool":
            messages.append(m)
        elif m["role"] == "assistant" and not m.get("tool_calls"):
            # Target turn — stop here for generation prompt
            break

    tools = ex.get("tools")
    prompt = tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS
    )
    ref = ex.get("translation", "").strip()
    return prompt, ref


def evaluate_checkpoint(
    model, tok, examples: List[Dict], batch_size: int = 4, max_new_tokens: int = 512
) -> Dict[str, float]:
    prompts_refs = [build_prompt_and_ref(ex, tok) for ex in examples]
    prompts = [p for p, _ in prompts_refs]
    refs = [r for _, r in prompts_refs]
    hyps = []

    # Identify all turn/sentence end tokens so the model stops immediately
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)
    for t_str in ["<turn|>", "<end_of_turn>", "<eos>", "<|im_end|>"]:
        tid = tok.convert_tokens_to_ids(t_str)
        if tid is not None and isinstance(tid, int) and tid not in eos_ids:
            eos_ids.append(tid)

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    model.eval()
    with torch.inference_mode():
        for b_idx, i in enumerate(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i : i + batch_size]
            enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            prompt_len = input_ids.shape[1]

            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids,
            )

            # Decode only the newly generated continuation
            for gen in out_ids:
                gen_tokens = gen[prompt_len:]
                decoded = tok.decode(gen_tokens, skip_special_tokens=False)
                clean_hyp = extract_translation(decoded)
                hyps.append(clean_hyp)

            if (b_idx + 1) % 5 == 0 or (b_idx + 1) == total_batches:
                print(f"  [Progress] Generated {min(i + batch_size, len(prompts))}/{len(prompts)} examples...", flush=True)

    # Compute sacrebleu metrics
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], beta=2)

    # Compute CER
    try:
        import editdistance
        total_dist = sum(editdistance.eval(r, h) for r, h in zip(refs, hyps))
        total_ref_chars = sum(len(r) for r in refs)
        cer = round(100.0 * total_dist / max(1, total_ref_chars), 2)
    except ImportError:
        cer = None

    # Compute GeminiScore (semantic embedding similarity via gemini-score)
    gemini_sem_ref = None
    gemini_sem_src = None
    gemini_tier = None
    try:
        import gemini_score
        api_key = os.environ.get("GEMINI_API_KEY")
        emb_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
        # 1. Target semantic similarity (Twi Hypothesis vs Twi Reference)
        res_ref = gemini_score.score(hyps, refs, api_key=api_key, model=emb_model)
        gemini_sem_ref = round(res_ref.score, 2)
        gemini_tier = res_ref.tier

        # 2. Cross-lingual semantic similarity (Twi Hypothesis vs English Source)
        srcs = [ex.get("english", "") for ex in examples]
        if any(srcs):
            res_src = gemini_score.score(hyps, srcs, api_key=api_key, model=emb_model)
            gemini_sem_src = round(res_src.score, 2)
    except Exception as e:
        print(f"  [Notice] GeminiScore skipped: {e}", flush=True)

    # Length ratio (words)
    hyp_words = sum(len(h.split()) for h in hyps)
    ref_words = sum(len(r.split()) for r in refs)
    lr = hyp_words / max(1, ref_words)

    return {
        "bleu": round(bleu.score, 2),
        "chrf2": round(chrf.score, 2),
        "cer": cer,
        "gemini_score": gemini_sem_ref,
        "gemini_score_src": gemini_sem_src,
        "gemini_tier": gemini_tier,
        "length_ratio": round(lr, 3),
        "hyp_words": hyp_words,
        "ref_words": ref_words,
        "predictions": hyps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Path or HF ID of base model")
    ap.add_argument("--checkpoints", required=True, help="Directory containing checkpoint-* folders")
    ap.add_argument("--data", required=True, help="Path to validation.jsonl")
    ap.add_argument("--limit", type=int, default=100, help="Number of validation examples (default 100)")
    ap.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate")
    ap.add_argument("--steps", nargs="*", type=int, default=None, help="Specific step numbers to evaluate (e.g. 900 1800)")
    ap.add_argument("--latest", action="store_true", help="Only evaluate the latest checkpoint")
    ap.add_argument("--out", default=None, help="Output JSON path (defaults to <checkpoints>/eval_metrics.json)")
    args = ap.parse_args()

    # Find checkpoints sorted by step
    ckpt_dirs = [
        d for d in glob.glob(os.path.join(args.checkpoints, "checkpoint-*")) if os.path.isdir(d)
    ]
    if not ckpt_dirs:
        print(f"No checkpoint-* directories found in {args.checkpoints}")
        return

    def get_step(path: str) -> int:
        match = re.search(r"checkpoint-(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else 0

    ckpt_dirs = sorted(ckpt_dirs, key=get_step)
    if args.steps:
        ckpt_dirs = [d for d in ckpt_dirs if get_step(d) in set(args.steps)]
    elif args.latest and ckpt_dirs:
        ckpt_dirs = [ckpt_dirs[-1]]
    print(f"Evaluating {len(ckpt_dirs)} checkpoint(s): {[os.path.basename(d) for d in ckpt_dirs]}")

    print(f"Loading tokenizer from {args.base}...")
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading validation examples from {args.data} (limit={args.limit})...")
    examples = load_val_examples(args.data, limit=args.limit)
    print(f"Loaded {len(examples)} examples.")

    print(f"Loading base model in bfloat16 on GPU from {args.base}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )

    results = []

    for ckpt in ckpt_dirs:
        step = get_step(ckpt)
        ckpt_name = os.path.basename(ckpt)

        # Get eval_loss from trainer_state.json if available
        trainer_state_path = os.path.join(ckpt, "trainer_state.json")
        eval_loss = None
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path) as f:
                ts = json.load(f)
                for entry in reversed(ts.get("log_history", [])):
                    if "eval_loss" in entry and entry.get("step") == step:
                        eval_loss = round(float(entry["eval_loss"]), 4)
                        break

        print(f"\n--- Evaluating {ckpt_name} (Step {step}, Eval Loss: {eval_loss}) ---", flush=True)
        peft_model = PeftModel.from_pretrained(base_model, ckpt)

        metrics = evaluate_checkpoint(
            peft_model, tok, examples, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens
        )
        metrics["step"] = step
        metrics["checkpoint"] = ckpt_name
        metrics["eval_loss"] = eval_loss

        # Save individual checkpoint predictions
        preds_file = os.path.join(args.checkpoints, f"preds_{step}.json")
        hyps_pred = metrics.pop("predictions", [])
        with open(preds_file, "w", encoding="utf-8") as f:
            json.dump({
                "step": step,
                "checkpoint": ckpt_name,
                "predictions": hyps_pred,
                "references": [ex.get("translation", "") for ex in examples],
                "sources": [ex.get("english", "") for ex in examples],
            }, f, indent=2)

        results.append(metrics)

        print(
            f"Result: Step {step} | Loss: {eval_loss} | chrF2: {metrics['chrf2']} | BLEU: {metrics['bleu']} | GemScore (Ref): {metrics.get('gemini_score')}% ({metrics.get('gemini_tier')}) | GemScore (Src): {metrics.get('gemini_score_src')}% | CER: {metrics.get('cer')}% | LenRatio: {metrics['length_ratio']}",
            flush=True,
        )

        # Unload adapter to free memory for the next checkpoint
        peft_model.unload()

    # Print summary table
    print("\n" + "=" * 105, flush=True)
    print(f"{'Checkpoint':<18} | {'Step':<6} | {'Loss':<8} | {'chrF2':<6} | {'BLEU':<6} | {'GemScore':<9} | {'GemSrc':<8} | {'CER':<6} | {'LenRatio':<8}", flush=True)
    print("-" * 105, flush=True)
    for r in results:
        loss_str = str(r["eval_loss"]) if r["eval_loss"] is not None else "N/A"
        cer_str = str(r["cer"]) if r.get("cer") is not None else "N/A"
        gem_str = f"{r['gemini_score']}%" if r.get("gemini_score") is not None else "N/A"
        gemsrc_str = f"{r['gemini_score_src']}%" if r.get("gemini_score_src") is not None else "N/A"
        print(f"{r['checkpoint']:<18} | {r['step']:<6} | {loss_str:<8} | {r['chrf2']:<6} | {r['bleu']:<6} | {gem_str:<9} | {gemsrc_str:<8} | {cer_str:<6} | {r['length_ratio']:<8}", flush=True)
    print("=" * 105, flush=True)

    out_file = args.out or os.path.join(args.checkpoints, "eval_metrics.json")
    existing = []
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    by_step = {r["step"]: r for r in existing}
    for r in results:
        by_step[r["step"]] = r
    merged_results = sorted(by_step.values(), key=lambda x: x["step"])

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2)
    print(f"\nSaved {len(merged_results)} metrics to {out_file}")


if __name__ == "__main__":
    main()
