#!/usr/bin/env python
"""
train_qlora.py -- QLoRA fine-tune Qwen2.5-Coder-7B-Instruct with Unsloth.
Designed to run on Kaggle (T4 x2) for the real run and on Colab (single T4) for a
smoke test -- same script, smaller flags.

v2 (reasoning traces): reads the SFT file produced by
data/prepare_reasoning_traces.py (default data/reasoning_train.jsonl), where each
row is {prompt, response} and `response` is a DeepSeek-R1 reasoning trace
("<think>...</think>" + a fenced Python solution). Wraps each {prompt, response}
in Qwen's chat template, masks everything before the assistant turn, and trains a
LoRA adapter so loss is computed on the reasoning + code. The field names are
configurable (--prompt-field / --response-field) so the same script still handles
the v1 {prompt, solution} format.

Why the v2 defaults differ from v1 (see CLAUDE.md 3):
  * --max-seq-len 4096  -- reasoning traces are long (v1's 2048 truncates them),
    but Unsloth free uses only ONE T4 (~22 s/example at 8192 in trials), so 8192 x
    10k examples ~= 62h is infeasible. 4096 keeps most complete traces and fits a
    ~3000-example, ~1-epoch run in one ~9h Kaggle session. Pass --max-seq-len 8192
    (with fewer --max-samples) if you want the longest traces intact.
  * --batch-size 1 --grad-accum 8 -- validated config; keeps a 16 GB T4 off OOM
    while holding the effective batch at 8.
  * --lr 1e-4 (was 2e-4) -- lower LR + one pass to reduce catastrophic forgetting.

RULES (see CLAUDE.md 4): trains on whatever is in --data. That file must be the
firewalled OpenCodeReasoning train subset (or APPS-train). NEVER point --data at
APPS-test or LiveCodeBench.

--- Smoke test (Colab / Kaggle trial) ------------------------------------------
    python train_qlora.py --max-samples 20 --max-steps 10 --save-steps 5
--- Real run (Kaggle T4 x2) ----------------------------------------------------
    python train_qlora.py --epochs 1 --save-steps 50 --push --merge-16bit
--- Resume after a disconnect (auto-detects last checkpoint in --output-dir) ----
    python train_qlora.py --epochs 1 ... --resume
--------------------------------------------------------------------------------

Auth for --push: set env var HF_TOKEN to a *write* token. In a notebook:
    from huggingface_hub import login; login(token=HF_TOKEN)   # or os.environ
See setup.md (Training) for the exact secret-loading cell.
"""
from __future__ import annotations

import argparse
import os

DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
DEFAULT_HF_USERNAME = os.environ.get("HF_USERNAME", "Shaurya-saini")
# v2 uses a DISTINCT repo so the v1 model (…-apps-qlora) is never overwritten.
DEFAULT_HF_REPO = "qwen2.5-coder-7b-ocr-qlora"

# Qwen2.5 chat-template markers -- used to mask the prompt so loss is computed on
# the solution tokens only (train_on_responses_only).
QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / model
    p.add_argument("--data", default=os.path.join("data", "reasoning_train.jsonl"),
                   help="SFT JSONL. v2: prepare_reasoning_traces.py (OCR train "
                        "subset). v1 format also works via --response-field solution.")
    p.add_argument("--prompt-field", default="prompt",
                   help="JSONL field for the user turn (default: prompt).")
    p.add_argument("--response-field", default="response",
                   help="JSONL field for the assistant target. v2: 'response' "
                        "(reasoning+code); v1: 'solution' (code only).")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Match the data prep. 4096 fits a ~3000-example run in "
                        "one T4 session; 8192 keeps longer traces but ~2x slower.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap training rows (smoke test).")
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    # optimization
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If >0, overrides --epochs (use for smoke tests).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-device batch. 1 keeps 8192-token seqs on a 16 GB T4.")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Effective batch = batch-size * grad-accum * n_gpus.")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Lower than v1 (2e-4) to reduce catastrophic forgetting.")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no-mask-prompt", action="store_true",
                   help="Disable train_on_responses_only (train on full text).")
    # checkpointing / resume
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--resume", action="store_true",
                   help="Resume from the last checkpoint in --output-dir if present.")
    # hub push
    p.add_argument("--push", action="store_true",
                   help="Push to the HF Hub after training (needs HF_TOKEN).")
    p.add_argument("--merge-16bit", action="store_true",
                   help="Also push a merged 16-bit model (easier to eval) besides the adapter.")
    p.add_argument("--hf-username", default=DEFAULT_HF_USERNAME)
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Imports deferred so --help works without the GPU stack installed.
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTTrainer, SFTConfig

    # ---- 1. Base model in 4-bit + LoRA adapters -----------------------------
    print(f"Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ---- 2. Data: {prompt, response} -> chat-template "text" ----------------
    if not os.path.exists(args.data):
        raise SystemExit(f"Training data not found: {args.data}\n"
                         f"Run data/prepare_reasoning_traces.py first (v2).")
    ds = load_dataset("json", data_files=args.data, split="train")
    for field in (args.prompt_field, args.response_field):
        if field not in ds.column_names:
            raise SystemExit(f"--data is missing field '{field}'. "
                             f"Columns present: {ds.column_names}. "
                             f"Set --prompt-field/--response-field to match.")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Training rows: {len(ds)} "
          f"(prompt='{args.prompt_field}', response='{args.response_field}')")

    prompt_field, response_field = args.prompt_field, args.response_field

    def to_text(batch):
        texts = []
        for prompt, response in zip(batch[prompt_field], batch[response_field]):
            msgs = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
        return {"text": texts}

    ds = ds.map(to_text, batched=True, remove_columns=ds.column_names)

    # ---- 2b. Truncation guard (defense-in-depth) ----------------------------
    # If a rendered example is longer than max_seq_len, SFTTrainer silently
    # truncates its TAIL -- i.e. the </think> + code + <|im_end|> -- and the model
    # learns to reason without ever concluding. v2 shipped exactly this bug
    # (a broken length filter in data prep let median-7.5k-token traces through).
    # Refuse to train on a corpus that would be tail-truncated.
    _lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in ds["text"]]
    _over = [n for n in _lens if n > args.max_seq_len]
    _mx = max(_lens) if _lens else 0
    print(f"Token lengths: max={_mx}, over-limit={len(_over)}/{len(_lens)} "
          f"(max_seq_len={args.max_seq_len}).")
    if _over:
        frac = len(_over) / len(_lens)
        msg = (f"{len(_over)}/{len(_lens)} ({frac:.0%}) examples exceed "
               f"--max-seq-len {args.max_seq_len} (max {_mx}); their reasoning/code "
               f"tail would be TRUNCATED at train time. Regenerate the corpus with "
               f"the fixed length filter (data/prepare_reasoning_traces.py) or raise "
               f"--max-seq-len to >= {_mx}.")
        if frac > 0.02:
            raise SystemExit("ABORT: " + msg)
        print("WARNING: " + msg)

    # ---- 3. Trainer ---------------------------------------------------------
    # NOTE: recent TRL moved dataset_text_field / packing / sequence length OUT of
    # SFTTrainer kwargs and INTO SFTConfig (and SFTTrainer now takes them via the
    # args object). This is the current, working API on the Unsloth 2026.x stack.
    use_bf16 = torch.cuda.is_bf16_supported()
    sft_config = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
        # --- aggressive checkpointing (free sessions can die without warning) ---
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=not use_bf16,
        bf16=use_bf16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_config,
    )

    # Compute loss on the assistant's solution only (not the problem statement).
    if not args.no_mask_prompt:
        try:
            from unsloth.chat_templates import train_on_responses_only
            trainer = train_on_responses_only(
                trainer,
                instruction_part=QWEN_INSTRUCTION_PART,
                response_part=QWEN_RESPONSE_PART,
            )
            print("train_on_responses_only: enabled (loss on solution tokens only).")
        except Exception as e:  # noqa: BLE001 - version-tolerant, non-fatal
            print(f"train_on_responses_only unavailable ({e}); training on full text.")

    # ---- 4. Train (resume if asked and a checkpoint exists) -----------------
    resume_ckpt = None
    if args.resume and os.path.isdir(args.output_dir):
        resume_ckpt = get_last_checkpoint(args.output_dir)
        print(f"Resuming from checkpoint: {resume_ckpt}" if resume_ckpt
              else "No checkpoint found; starting fresh.")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ---- 5. Save adapter locally --------------------------------------------
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter -> {adapter_dir}")

    # ---- 6. Optional: push to the Hugging Face Hub --------------------------
    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--push requires env var HF_TOKEN (write role). "
                             "See setup.md (Training).")
        repo = f"{args.hf_username}/{args.hf_repo}"
        print(f"Pushing LoRA adapter -> {repo}-lora")
        # Unsloth's patched push_to_hub takes only the repo id positionally, so
        # the tokenizer must be pushed with its own call. (push_to_hub_merged
        # below is different -- it does accept the tokenizer positionally.)
        model.push_to_hub(f"{repo}-lora", token=token)
        tokenizer.push_to_hub(f"{repo}-lora", token=token)
        if args.merge_16bit:
            print(f"Pushing merged 16-bit model -> {repo}")
            model.push_to_hub_merged(repo, tokenizer,
                                     save_method="merged_16bit", token=token)
        print("Push complete.")

    print("\nTraining done.")


if __name__ == "__main__":
    main()
