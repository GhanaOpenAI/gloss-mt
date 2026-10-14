"""LoRA fine-tune of TranslateGemma on the chunked English-Twi dataset
ghanaopenai/twi-translation-glossary-chunks, with the same inputs as train_seq2seq.py
(load_chunks: english_pretranslated_gold for training, english_pretranslated for validation),
rendered with TranslateGemma's own prompt (twi_translate.translategemma.render_prompt); the
loss covers only the Twi chunk and the closing <end_of_turn>. Single GPU, bf16 + TF32.

Usage (H200; --data is the HF dataset or a local save_to_disk copy of it):
  python training/train_translategemma.py --model google/translategemma-4b-it \\
      --out checkpoints/translategemma-4b-twi
  python training/train_translategemma.py ... --max-steps 30   # memory/throughput pilot
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_eval import batched_generate, sample_examples, with_gen_metrics  # noqa: E402
from train_seq2seq import DATASET, disk_guard, load_chunks  # noqa: E402
from twi_translate.translategemma import END_OF_TURN, load_translategemma, render_prompt  # noqa: E402

IGNORE = -100
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def tokenize(tok, text: str, tgt: str, max_len: int):
    """text is the (pretranslated) source chunk, given to the prompt as is."""
    prompt = tok(render_prompt(tok, text, None), add_special_tokens=False).input_ids
    answer = tok(tgt + END_OF_TURN + "\n", add_special_tokens=False).input_ids
    ids = prompt + answer
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [IGNORE] * len(prompt) + answer}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/translategemma-4b-it")
    ap.add_argument("--data", default=DATASET, help="HF dataset id or local save_to_disk directory")
    ap.add_argument("--out", default="checkpoints/translategemma-4b-twi")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="2 keeps the best and the most recent checkpoint")
    ap.add_argument("--min-free-gb", type=float, default=40, help="stop cleanly below this much free disk")
    ap.add_argument("--gen-eval-chunks", type=int, default=200,
                    help="validation chunks translated at each eval for BLEU/chrF2 and the "
                         "composite that selects the best checkpoint (0 = select on eval_loss only)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer,
                              TrainingArguments)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(args.model)
    ds, gen_examples = {}, []
    for split in ("train", "validation"):
        ex = load_chunks(args.data, split)
        enc = [tokenize(tok, e["input"], e["tgt"], args.max_len) for e in ex]
        kept = [e for e in enc if e]
        ds[split] = Dataset.from_list(kept)
        print(f"{split}: {len(ex)} chunks, {len(ex) - len(kept)} over length dropped",
              flush=True)
        if split == "validation" and args.gen_eval_chunks > 0:
            gen_examples = sample_examples(ex, args.gen_eval_chunks)

    model = load_translategemma(args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                device_map={"": 0})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="composite" if gen_examples else "eval_loss",
        greater_is_better=bool(gen_examples),
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids(END_OF_TURN)]

    def gen_fn(m, examples):
        prompts = [render_prompt(tok, e["input"], None) for e in examples]
        outs = batched_generate(m, tok, prompts, args.batch_size * 4, 1024, decoder_only=True,
                                eos_token_id=eos, pad_token_id=tok.pad_token_id)
        return [o.strip() for o in outs]

    trainer_cls = with_gen_metrics(Trainer)
    trainer_cls.gen_examples, trainer_cls.gen_fn = gen_examples, staticmethod(gen_fn)
    trainer = trainer_cls(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=IGNORE),
        callbacks=[disk_guard(args.out, args.min_free_gb)]
        + ([EarlyStoppingCallback(args.early_stopping_patience)] if args.early_stopping_patience > 0 else []),
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
