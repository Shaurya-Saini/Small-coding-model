#!/usr/bin/env python
"""
fix_tokenizer_config.py -- one-time repair of a pushed model's tokenizer_config.json
so it loads under transformers 4.x (what bigcode-evaluation-harness needs).

Why this is needed
------------------
The training env (Unsloth + transformers 5.x) saved the tokenizer with the special
tokens under a top-level `extra_special_tokens` LIST -- the transformers-5 format.
transformers 4.x expects that field to be a dict and crashes while loading with:

    AttributeError: 'list' object has no attribute 'keys'
    (in tokenization_utils_base._set_model_specific_special_tokens)

The un-tuned base model (Qwen/Qwen2.5-Coder-7B-Instruct) stores the SAME 13 tokens
under `additional_special_tokens` (a list -- the 4.x-native format) and loads fine.
So the fix is simply to rename our field to match the base model. The tokens are
also already present in `added_tokens_decoder`, so tokenization is unchanged; this
is a pure metadata repair. The result loads on BOTH transformers 4.x and 5.x.

See: huggingface/transformers issue #45376 (v4/v5 extra_special_tokens format).

Usage (needs a WRITE token in env: HF_TOKEN)
    python eval/fix_tokenizer_config.py --repo Shaurya-saini/qwen2.5-coder-7b-apps-qlora
    python eval/fix_tokenizer_config.py --repo <id> --dry-run   # inspect, don't upload
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF model repo id to repair.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change; do not upload.")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, upload_file

    local = hf_hub_download(args.repo, "tokenizer_config.json")
    with open(local, encoding="utf-8") as fh:
        cfg = json.load(fh)

    extra = cfg.get("extra_special_tokens")
    if extra is None:
        print("Nothing to do: no `extra_special_tokens` key present "
              "(already 4.x-compatible).")
        return
    if not isinstance(extra, list):
        print(f"`extra_special_tokens` is a {type(extra).__name__}, not a list -- "
              "this script only handles the v5 list format. Leaving it unchanged.")
        return

    print(f"Found `extra_special_tokens` (list of {len(extra)}). "
          f"Renaming to `additional_special_tokens`.")
    cfg.pop("extra_special_tokens", None)
    # Don't clobber an existing additional_special_tokens; merge if somehow both exist.
    existing = cfg.get("additional_special_tokens") or []
    merged = existing + [t for t in extra if t not in existing]
    cfg["additional_special_tokens"] = merged

    out = os.path.join(os.path.dirname(local), "tokenizer_config.fixed.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)

    if args.dry_run:
        print(f"[dry-run] Wrote fixed file to {out}; NOT uploading.")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN (write) not set in env. Load your Kaggle Secret "
                         "into os.environ first, then re-run.")
    upload_file(
        path_or_fileobj=out,
        path_in_repo="tokenizer_config.json",
        repo_id=args.repo,
        token=token,
        commit_message="Fix tokenizer_config for transformers 4.x "
                       "(extra_special_tokens list -> additional_special_tokens)",
    )
    print(f"Uploaded repaired tokenizer_config.json to {args.repo}. "
          "It now loads on transformers 4.x and 5.x.")


if __name__ == "__main__":
    main()
