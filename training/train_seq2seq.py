"""Fine-tune umT5 or NLLB (full or LoRA) on the chunked English-Twi dataset
ghanaopenai/gloss-mt-parallel-en-twi (built by scripts/build_chunked_dataset.py).

Every chunk is one example -> its gold Twi chunk. Training inputs are english_pretranslated_gold:
the chunk's noun phrases replaced in place by "<Twi>" spans that Gemini aligned in the gold chunk,
with the same spans in <brackets> in the target (twi_bracketed), so the model learns to carry
bracketed terms over and translate the rest. Validation inputs are english_pretranslated:
dictionary terms in brackets, exactly what twi_translate.seq2seq gives the model at inference,
scored against the plain reference twi. Brackets are removed from generated text (clean_output). umT5 gets a "translate English to Twi: " prefix; NLLB (detected from its
tokenizer) gets eng_Latn -> twi_Latn language codes instead.

Full fine-tune: bf16 autocast over fp32 weights, Adafactor. --lora-r N instead trains LoRA
adapters (bf16 base, AdamW) on every attention/FFN projection plus the embedding row of the added
"\\n" token; checkpoints hold the adapter and final/ the merged model.

Usage (H200; --data is the HF dataset or a local save_to_disk copy of it):
  python training/train_seq2seq.py --model facebook/nllb-200-3.3B \\
      --out checkpoints/nllb-3.3b-twi-lora --lora-r 32 --lr 2e-4
  python training/train_seq2seq.py --model google/umt5-xl --out checkpoints/umt5-xl-twi
  python training/train_seq2seq.py ... --max-steps 20   # memory/throughput pilot
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twi_translate.seq2seq import (  # noqa: E402
    TRANSLATE_PREFIX, clean_output, load_seq2seq, nllb_langs,
    prepare_tokenizer)

DATASET = "ghanaopenai/gloss-mt-parallel-en-twi"
LORA_TARGETS = r".*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2|q|k|v|o|wi_0|wi_1|wo)$"  # NLLB and umT5


def load_chunks(data: str, split: str) -> List[Dict]:
    """The split's chunks as {"input", "tgt", "chunk"}: training reads english_pretranslated_gold and
    twi_bracketed (gold spans in <brackets> on both sides); validation/test read english_pretranslated
    (dictionary terms in brackets, the inference input) and the plain reference twi."""
    from datasets import load_dataset, load_from_disk

    rows = load_from_disk(data)[split] if os.path.isdir(data) else load_dataset(data, split=split)
    src, tgt = ("english_pretranslated_gold", "twi_bracketed") if split == "train" else ("english_pretranslated", "twi")
    return [{"input": r[src], "tgt": r[tgt], "chunk": r["english"], "ref": r["twi"]} for r in rows]


def disk_guard(path: str, min_free_gb: float):
    """Callback that saves and stops training cleanly when free disk space falls below min_free_gb
    (resume later from the last checkpoint once space is freed)."""
    from transformers import TrainerCallback

    class DiskGuard(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if control.should_save or state.global_step % 50:
                return
            free = shutil.disk_usage(path).free / 1e9
            if free < min_free_gb:
                print(f"DiskGuard: {free:.0f} GB free < {min_free_gb:.0f} GB; saving and stopping at step "
                      f"{state.global_step}", flush=True)
                control.should_save = True
                control.should_training_stop = True

    return DiskGuard()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/umt5-xl")
    ap.add_argument("--data", default=DATASET, help="HF dataset id or local save_to_disk directory")
    ap.add_argument("--out", default="checkpoints/umt5-xl-twi")
    ap.add_argument("--max-src", type=int, default=1024)
    ap.add_argument("--max-tgt", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank (0 = full fine-tune)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
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
    from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(args.model)
    added = prepare_tokenizer(tok)
    langs = nllb_langs(tok)  # NLLB: language codes replace the T5-style task prefix
    if langs:
        tok.src_lang, tok.tgt_lang = langs
    prefix = "" if langs else TRANSLATE_PREFIX

    ds, gen_examples = {}, []
    for split in ("train", "validation"):
        ex = load_chunks(args.data, split)
        for e in ex:
            e["src"] = prefix + e["input"]
        enc = tok([e["src"] for e in ex], text_target=[e["tgt"] for e in ex])
        keep = [i for i in range(len(ex))
                if len(enc["input_ids"][i]) <= args.max_src and len(enc["labels"][i]) <= args.max_tgt]
        ds[split] = Dataset.from_dict({k: [enc[k][i] for i in keep]
                                       for k in ("input_ids", "attention_mask", "labels")})
        print(f"{split}: {len(ex)} chunks, {len(ex) - len(keep)} over length dropped",
              flush=True)
        if split == "validation":
            gen_examples = sample_examples(ex, args.gen_eval_chunks) if args.gen_eval_chunks > 0 else []

    model = load_seq2seq(args.model, dtype=torch.bfloat16 if args.lora_r else torch.float32)
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    if args.lora_r:
        from peft import LoraConfig, get_peft_model

        new_ids = [tok.convert_tokens_to_ids("\n")] if added else []  # "\n" is new to the vocab: train its row
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="SEQ_2_SEQ_LM",
            target_modules=LORA_TARGETS, trainable_token_indices=new_ids or None))
        model.print_trainable_parameters()

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        optim="adamw_torch" if args.lora_r else "adafactor",
        lr_scheduler_type="linear",
        warmup_steps=200,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
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
    )
    forced = {"forced_bos_token_id": tok.convert_tokens_to_ids(langs[1])} if langs else {}

    def gen_fn(m, examples):
        outs = batched_generate(m, tok, [e["src"] for e in examples], args.batch_size * 2, args.max_tgt, **forced)
        return [clean_output(o) for o in outs]

    trainer_cls = with_gen_metrics(Seq2SeqTrainer)
    trainer_cls.gen_examples, trainer_cls.gen_fn = gen_examples, staticmethod(gen_fn)
    trainer = trainer_cls(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
        processing_class=tok,
        callbacks=[disk_guard(args.out, args.min_free_gb)]
        + ([EarlyStoppingCallback(args.early_stopping_patience)] if args.early_stopping_patience > 0 else []),
    )
    has_ckpt = os.path.isdir(args.out) and any(d.startswith("checkpoint-") for d in os.listdir(args.out))
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    print(trainer.evaluate(), flush=True)

    final = os.path.join(args.out, "final")
    if args.lora_r:
        trainer.model.save_pretrained(os.path.join(args.out, "adapter"))
        trainer.model.merge_and_unload().save_pretrained(final)
    else:
        trainer.save_model(final)
    tok.save_pretrained(final)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved model to {final}", flush=True)


if __name__ == "__main__":
    main()
