"""Merge a train_seq2seq.py --lora-r checkpoint (adapter + trained "\\n" token row) into its
base model and save a standalone model + tokenizer, e.g. to evaluate a chosen checkpoint
(train_seq2seq.py itself merges only the best checkpoint at the end of a completed run).

  python training/merge_seq2seq_lora.py \\
      --adapter checkpoints/nllb-3.3b-twi-lora/checkpoint-8000 \\
      --out checkpoints/nllb-3.3b-twi-lora/best-merged
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twi_translate.seq2seq import load_seq2seq, nllb_langs, prepare_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="checkpoint directory holding the LoRA adapter")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        base_id = json.load(f)["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(base_id)
    prepare_tokenizer(tok)  # the same added "\n" token as in training
    langs = nllb_langs(tok)
    if langs:
        tok.src_lang, tok.tgt_lang = langs
    model = load_seq2seq(base_id, dtype=torch.bfloat16)
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Merged {args.adapter} into {base_id} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
