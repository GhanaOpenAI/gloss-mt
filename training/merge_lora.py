"""Merge a trained LoRA adapter into the bf16 base model and save a standalone model.

The output has the same layout as the base checkpoint (plus the base tokenizer, processor,
chat template and generation config), so it can be served exactly like the base model:

  python training/merge_lora.py --base /mnt/volume_d2wey28/models/gemma-4-31B-it \\
      --adapter checkpoints/gemma4-31b-twi-tool/adapter --out checkpoints/gemma4-31b-twi-tool/merged
  vllm serve checkpoints/gemma4-31b-twi-tool/merged ...
"""

import argparse
import os
import shutil

COPY_FILES = ("chat_template.jinja", "processor_config.json", "preprocessor_config.json",
              "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "tokenizer.model")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="bf16 base model (path or HF id)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0", help="'cpu' works too, just slower")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": args.device})
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="10GB")

    base_dir = args.base if os.path.isdir(args.base) else snapshot_download(
        args.base, allow_patterns=list(COPY_FILES))
    for name in COPY_FILES:
        src = os.path.join(base_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
