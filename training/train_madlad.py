"""Fine-tune MADLAD-400 (google/madlad400-3b-mt, T5) on the chunked English-Twi dataset: the
training of train_seq2seq.py (same data, LoRA or full fine-tune, generation-based checkpoint
selection), with MADLAD's "<2ak> " target-language token as the input prefix in place of umT5's
"translate English to Twi: ". At inference twi_translate.seq2seq.task_prefix detects MADLAD's
tokenizer and adds the same prefix. All options are train_seq2seq.py's.

Usage (H200; --data is the HF dataset or a local save_to_disk copy of it):
  python training/train_madlad.py --model google/madlad400-3b-mt --out checkpoints/madlad-3b-twi --lora-r 32
  python training/train_madlad.py ... --max-steps 20   # memory/throughput pilot
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seq2seq  # noqa: E402
from twi_translate.seq2seq import MADLAD_PREFIX  # noqa: E402

if __name__ == "__main__":
    train_seq2seq.TRANSLATE_PREFIX = MADLAD_PREFIX  # read by train_seq2seq.main for non-NLLB models
    train_seq2seq.main()
