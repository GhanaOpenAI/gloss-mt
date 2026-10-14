"""Generation-based eval metrics during training, and a composite score for checkpoint selection.

At every eval, on top of the validation loss, the model translates a fixed (seeded) sample of
validation chunks and is scored with corpus BLEU and chrF2. The loss is mapped onto the same
0-100, higher-is-better scale as 100 * exp(-loss) (the mean per-token probability of the
reference, in percent), and

    composite = mean(bleu, chrf2, 100 * exp(-loss))

selects the best checkpoint and drives early stopping. GeminiScore is deliberately left out:
it is an embedding similarity that also rewards a correct translation into the wrong language
(zero-shot TranslateGemma wrote Tamil/Greek script and still scored 88), and it varies little.
"""

import math
import random
from typing import Callable, Dict, List, Optional


def sample_examples(examples: List[Dict], n: int, seed: int = 0) -> List[Dict]:
    """A fixed, order-preserving random sample of n examples (all of them if n >= len)."""
    if n <= 0 or n >= len(examples):
        return list(examples)
    idx = sorted(random.Random(seed).sample(range(len(examples)), n))
    return [examples[i] for i in idx]


def text_metrics(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    import sacrebleu

    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf2": round(sacrebleu.corpus_chrf(hyps, [refs], beta=2).score, 2)}


def composite(metrics: Dict[str, float], loss: float) -> Dict[str, float]:
    loss_score = round(100.0 * math.exp(-loss), 2)
    return {"loss_score": loss_score,
            "composite": round((metrics["bleu"] + metrics["chrf2"] + loss_score) / 3, 3)}


def with_gen_metrics(trainer_cls):
    """Subclass a Trainer so evaluation_loop also reports generation metrics + the composite.

    The trainer needs `gen_examples` (dicts with "tgt") and `gen_fn(model, examples) -> hyps`.
    """

    class GenEvalTrainer(trainer_cls):
        gen_examples: List[Dict] = []
        gen_fn: Optional[Callable] = None

        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                            metric_key_prefix="eval"):
            out = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys,
                                          metric_key_prefix)
            if self.gen_examples and self.gen_fn is not None:
                was_training = self.model.training
                self.model.eval()
                hyps = self.gen_fn(self.model, self.gen_examples)
                if was_training:
                    self.model.train()
                m = text_metrics(hyps, [e["tgt"] for e in self.gen_examples])
                m.update(composite(m, out.metrics[f"{metric_key_prefix}_loss"]))
                out.metrics.update({f"{metric_key_prefix}_{k}": v for k, v in m.items()})
                print(f"[gen_eval] {m}", flush=True)
            return out

    return GenEvalTrainer


def batched_generate(model, tok, inputs: List[str], batch_size: int, max_new_tokens: int,
                     decoder_only: bool = False, **gen_kwargs) -> List[str]:
    """Greedy generation in bf16 autocast, longest inputs first; returns decoded outputs in order."""
    import torch

    order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i]))
    out = [""] * len(inputs)
    side = tok.padding_side
    if decoder_only:
        tok.padding_side = "left"
    try:
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([inputs[i] for i in idx], return_tensors="pt", padding=True, truncation=not decoder_only,
                      max_length=None if decoder_only else 1024,
                      add_special_tokens=not decoder_only).to(model.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                     use_cache=True, **gen_kwargs)
            if decoder_only:
                gen = gen[:, enc["input_ids"].shape[1]:]
            for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t
    finally:
        tok.padding_side = side
    return out
