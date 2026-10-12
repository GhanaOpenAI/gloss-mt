"""LoRA SFT of Gemma 4 on the Twi tool-calling + reasoning dataset (single GPU, bf16 + TF32).

Loss is computed only on what the model generates at inference:
  - the tool call   (<|tool_call>call:lookup_twi_translation{...}<tool_call|><|tool_response>)
  - the reasoning + translation  (... <channel|>Twi translation<turn|>)
System/user/tool-result tokens are masked. Conversations are rendered with
enable_thinking=True, which is also what the runtime uses, so after the tool result the
prompt opens the thought channel and the model reasons before translating.

Two backends: `unsloth` (default; Unsloth kernels + offloaded gradient checkpointing, optional
4-bit QLoRA) and `hf` (plain transformers + PEFT). Both train a LoRA on the language model
only and save the adapter; merge it into the bf16 base with training/merge_lora.py.

Usage (H200):
  python training/train_sft.py --model /mnt/volume_d2wey28/models/gemma-4-E4B-it \
      --data data --out checkpoints/gemma4-e4b-twi-tool --backend hf
  python training/train_sft.py ... --max-steps 20   # throughput pilot
  python training/merge_lora.py --base /mnt/volume_d2wey28/models/gemma-4-E4B-it \
      --adapter checkpoints/gemma4-e4b-twi-tool/adapter --out checkpoints/gemma4-e4b-twi-tool/merged
"""

import argparse
import json
import os
from typing import Dict, List

IGNORE = -100
GEMMA_LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
QWEN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TEMPLATE_KWARGS = {"enable_thinking": True}


def _clean(messages: List[Dict]) -> List[Dict]:
    """Arrow fills absent keys with None; drop them so the chat template sees clean dicts."""
    return [{k: v for k, v in m.items() if v is not None} for m in messages]


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def assistant_char_spans(tok, messages: List[Dict], tools, is_qwen: bool = False) -> (str, List[tuple]):
    """Render the conversation; return the text and char spans of every assistant turn."""
    if is_qwen:
        full = tok.apply_chat_template(messages, tools=tools, tokenize=False)
        asst_tag = "<|im_start|>assistant\n"
        end_tag = "<|im_end|>"
        spans = []
        pos = 0
        while True:
            start = full.find(asst_tag, pos)
            if start == -1:
                break
            content_start = start + len(asst_tag)
            end = full.find(end_tag, content_start)
            if end == -1:
                break
            spans.append((content_start, end + len(end_tag)))
            pos = end + len(end_tag)
        return full, spans
    else:
        render = lambda msgs, gen=False: tok.apply_chat_template(  # noqa: E731
            msgs, tools=tools, tokenize=False, add_generation_prompt=gen, **TEMPLATE_KWARGS)
        full = render(messages)
        spans = []
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            start = _common_prefix(render(messages[:i], gen=True), full)
            end = _common_prefix(render(messages[: i + 1]), full)
            if end > start:
                spans.append((start, end))
        return full, spans


def tokenize_example(ex: Dict, tok, max_len: int, is_qwen: bool = False) -> Dict:
    messages = _clean(ex["messages"])
    tools = ex["tools"]
    text, spans = assistant_char_spans(tok, messages, tools, is_qwen=is_qwen)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        return {"input_ids": [], "labels": [], "attention_mask": []}
    labels = [IGNORE] * len(ids)
    for j, (s, e) in enumerate(offsets):
        if e > s and any(a <= s < b for a, b in spans):
            labels[j] = ids[j]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def load_model(args):
    """Return a PEFT-wrapped model with LoRA on the linear layers."""
    import torch

    if args.backend == "unsloth":
        from unsloth import FastModel

        model, _ = FastModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_len, load_in_4bit=args.load_in_4bit,
            full_finetuning=False, dtype=torch.bfloat16)
        return FastModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    model.config.use_cache = False
    is_qwen = "qwen" in args.model.lower()
    targets = QWEN_LORA_TARGETS if is_qwen else GEMMA_LORA_TARGETS
    return get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=targets))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="checkpoints/gemma4-e4b-twi-tool")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=150)
    ap.add_argument("--backend", choices=["unsloth", "hf"], default="hf")
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA (unsloth backend)")
    ap.add_argument("--early-stopping-patience", type=int, default=3, help="Stop after N evals without eval_loss improvement (0 = disable)")
    ap.add_argument("--save-total-limit", type=int, default=None, help="Max checkpoints to keep (default: None, save all)")
    args = ap.parse_args()

    if args.backend == "unsloth":
        # Variable-length batches exhaust torch.compile's recompile limit within ~8 steps, and
        # Unsloth's compiled->eager switch then crashes on Gemma 4 (CUDA launch failure). Eager
        # from the start is stable; Unsloth's kernels and offloaded checkpointing still apply.
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        import unsloth  # noqa: F401  (must be imported before transformers)
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(args.model)
    is_qwen = "qwen" in args.model.lower()
    files = {s: os.path.join(args.data, f"{s}.jsonl") for s in ("train", "validation")}
    ds = load_dataset("json", data_files=files)
    keep = {"input_ids", "labels", "attention_mask"}
    ds = ds.map(lambda ex: tokenize_example(ex, tok, args.max_len, is_qwen=is_qwen), num_proc=16,
                remove_columns=[c for c in ds["train"].column_names if c not in keep])
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    print({s: len(d) for s, d in ds.items()}, flush=True)

    model = load_model(args)
    model.print_trainable_parameters()

    callbacks = []
    if args.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.001))

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        tf32=True,
        gradient_checkpointing=args.backend == "hf",  # unsloth applies its own
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.early_stopping_patience > 0,
        metric_for_best_model="eval_loss" if args.early_stopping_patience > 0 else None,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=IGNORE),
        callbacks=callbacks,
    )
    has_ckpt = os.path.isdir(args.out) and any(d.startswith("checkpoint-") for d in os.listdir(args.out))
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    print(trainer.evaluate(), flush=True)

    model.save_pretrained(os.path.join(args.out, "adapter"))
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved adapter to {os.path.join(args.out, 'adapter')}", flush=True)


if __name__ == "__main__":
    main()
