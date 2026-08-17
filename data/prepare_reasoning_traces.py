#!/usr/bin/env python
"""
prepare_reasoning_traces.py -- Build the v2 SFT corpus from OpenCodeReasoning.

v2 fixes root cause #1 of v1 (see CLAUDE.md 3): instead of imitating the single
*shortest* (golfed) APPS solution per problem, we imitate DeepSeek-R1 *reasoning
traces* -- a long chain-of-thought scratchpad followed by a clean solution. Same
supervised QLoRA stack as v1; only the DATA changes (clean v1->v2 ablation).

Source: nvidia/OpenCodeReasoning, config "split_0" (~568k rows). split_0 carries
the problem statement inline (`input`), R1's full response (`output`, which
already contains `<think>...reasoning...</think>` + a solution), and the extracted
code-only portion (`solution`). We deliberately do NOT use split_1 (it has no
inline question -- it must be joined back to APPS/TACO, which pulls raw benchmark
problems in and complicates the firewall).

=======================  NON-NEGOTIABLE FIREWALL (CLAUDE.md 4)  =================
Empirical fact (verified 2026-08-18 by a FULL 567,850-row scan): OCR **split_0's
`split=='train'` partition is effectively `code_contests` ONLY.** No apps/taco
(they're in split_1, needing an APPS/TACO join), and the other platforms the HF
statistics API hinted at (codeforces/atcoder/...) do NOT survive the train filter
here -- the ~39k non-code_contests rows are all *-valid/*-test and are correctly
dropped. So cross-platform balancing is impossible from split_0; we simply take
the target count from code_contests (a strong AGGREGATED competitive source:
DeepMind CodeContests = Codeforces/AtCoder/CodeChef/... problems). Consequences:

  * The APPS-*test* firewall is AUTOMATICALLY satisfied -- no APPS data in split_0.
  * `--per-source-frac` defaults to 1.0 (no balancing) because there is only one
    viable source. It stays a knob in case a future OCR revision adds platforms.
  * `EXCLUDE_SOURCES` (below) is the deny-list knob -- empty by default.
  * (Want true multi-platform diversity? That needs split_1 + the APPS/TACO join
    -- deferred to v2.1; see V2_PROGRESS.md.)

Two mandatory filters remain:
  1. split == "train"            -> drops any *-test / *-valid partition.
  2. dataset not in EXCLUDE_SOURCES.

LiveCodeBench caveat (for Phase 3): code_contests is a live-platform aggregate, so
its problems CAN overlap LiveCodeBench's recent window. This script does NOT try
to solve that; the LCB firewall is enforced at EVAL time -- pin the LCB version
window to dates that POST-DATE this corpus, or decontaminate LCB by problem id.
(LCB is never trained on regardless.)
================================================================================

Output (default): data/reasoning_train.jsonl -- one JSON object per example:
    id         : OpenCodeReasoning row id
    source     : originating dataset (in practice: code_contests -- see header)
    difficulty : raw source difficulty label (kept as-is; not normalized)
    prompt     : instruction + problem statement (user turn, pre chat-template)
    response   : assistant target -> "<think>\\n{reasoning}\\n</think>\\n\\n
                 ```python\\n{code}\\n```"  (reasoning trace + clean fenced code)
    n_tokens   : full rendered chat-template length (<= --max-seq-len)

training/train_qlora.py wraps {prompt, response} in Qwen's chat template and masks
everything before the assistant turn, so loss is computed on the reasoning + code.

Usage:
    python prepare_reasoning_traces.py                       # ~10k examples, 8192 tok
    python prepare_reasoning_traces.py --target 200 --max-seq-len 4096   # smoke
    python prepare_reasoning_traces.py --no-token-filter     # char-proxy (no tok)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

OCR_DATASET = "nvidia/OpenCodeReasoning"
OCR_CONFIG = "split_0"

# --- FIREWALL deny-list. split_0 has no apps/taco anyway (see header); empty by
# default so we train on ALL split_0 platforms, balanced via --per-source-frac.
# Add e.g. {"codeforces"} here to drop a specific platform. ---
EXCLUDE_SOURCES: set[str] = set()

# Instruction wrapper. MUST match the eval-time prompt so the model is on-
# distribution at test time (v1's biggest silent bug was a train/eval prompt
# mismatch -- see CLAUDE.md 8). Keep this string in sync with the eval generator.
INSTRUCTION = (
    "You are an expert competitive programmer. Solve the following problem.\n"
    "Reason step by step inside <think> </think> tags, then give the complete, "
    "correct solution as a single Python code block.\n\n"
)

TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_PYFENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def load_ocr_stream(seed: int, shuffle_buffer: int):
    """Stream OpenCodeReasoning split_0 (parquet-native; no loading script).

    Streaming avoids downloading the full ~28 GB split just to keep ~10k rows.
    A buffered shuffle de-correlates the take order from the on-disk ordering so
    the subset isn't dominated by whatever source happens to be stored first.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(OCR_DATASET, OCR_CONFIG, split=OCR_CONFIG, streaming=True)
    except (ValueError, KeyError):
        # Fall back if the single split isn't literally named "split_0".
        dsd = load_dataset(OCR_DATASET, OCR_CONFIG, streaming=True)
        ds = dsd[next(iter(dsd.keys()))]
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


def extract_reasoning(output: str) -> str | None:
    """Pull the chain-of-thought out of R1's `output` (the text inside <think>).

    Returns None if there is no usable reasoning -- we only want *reasoning*
    traces, so a row without a <think> block is skipped."""
    if not output:
        return None
    m = _THINK_RE.search(output)
    if m:
        reasoning = m.group(1).strip()
        return reasoning or None
    # Some rows have an unterminated think (truncated). If it opens <think> but
    # never closes, take everything after the tag up to the first code fence.
    lower = output.lower()
    if "<think>" in lower:
        after = output[lower.index("<think>") + len("<think>"):]
        after = _PYFENCE_RE.split(after)[0].strip()
        return after or None
    return None


def extract_code(solution: str, output: str) -> str | None:
    """Prefer the authoritative code-only `solution` field; fall back to the LAST
    ```python fence in `output` (the final solution after the reasoning)."""
    if solution and solution.strip():
        return solution.strip()
    if output:
        fences = _PYFENCE_RE.findall(output)
        if fences:
            return fences[-1].strip()
    return None


def firewall_source(row) -> str | None:
    """CHEAP gate (no regex/tokenize): return the lower-cased source if the row
    passes the firewall, else None. Kept separate so the per-source cap can be
    checked BEFORE the expensive parse -- split_0 is clustered by source, so we
    skip millions of over-cap rows without paying to parse their huge `output`."""
    if str(row.get("split", "")).strip().lower() != "train":
        return None
    source = str(row.get("dataset", "")).strip().lower()
    if not source or source in EXCLUDE_SOURCES:
        return None
    return source


def parse_row(row, source: str) -> dict | None:
    """EXPENSIVE step: extract the reasoning + code and build the SFT record."""
    question = (row.get("input") or "").strip()
    if not question:
        return None  # split_1-style rows (no inline question) are not used

    reasoning = extract_reasoning(row.get("output") or "")
    if not reasoning:
        return None
    code = extract_code(row.get("solution") or "", row.get("output") or "")
    if not code:
        return None

    prompt = INSTRUCTION + question
    response = f"<think>\n{reasoning}\n</think>\n\n```python\n{code}\n```"
    return {
        "id": row.get("id"),
        "source": source,
        "difficulty": str(row.get("difficulty", "")),
        "prompt": prompt,
        "response": response,
    }


def make_length_fn(max_seq_len: int, use_tokenizer: bool):
    """Return (n_tokens_of_full_chat, ok_flag). Uses the real Qwen tokenizer so
    the filter matches what training will truncate at; falls back to a char proxy
    if transformers/the tokenizer isn't available (with a loud warning)."""
    if use_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

            def n_tokens(prompt: str, response: str) -> int:
                msgs = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}]
                ids = tok.apply_chat_template(msgs, tokenize=True,
                                              add_generation_prompt=False)
                return len(ids)

            print(f"Length filter: exact tokens via {TOKENIZER_NAME} "
                  f"(max_seq_len={max_seq_len}).")
            return n_tokens
        except Exception as e:  # noqa: BLE001 - stay runnable in a minimal env
            print(f"WARNING: could not load {TOKENIZER_NAME} ({e}); "
                  f"falling back to a character-length proxy.")

    # Char proxy: ~3.3 chars/token is conservative for mixed English+code.
    def n_tokens_proxy(prompt: str, response: str) -> int:
        return int((len(prompt) + len(response)) / 3.3)

    print(f"Length filter: CHAR PROXY (~3.3 chars/token, max_seq_len={max_seq_len}).")
    return n_tokens_proxy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=os.path.join("data", "reasoning_train.jsonl"))
    p.add_argument("--target", type=int, default=3000,
                   help="How many examples to write (default: 3000 -- fits ~1 "
                        "T4 session at seq_len 4096).")
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Drop examples whose full chat length exceeds this "
                        "(default: 4096 -- matches training; 8192 keeps longer "
                        "traces but ~2x slower to train).")
    p.add_argument("--per-source-frac", type=float, default=1.0,
                   help="Cap any single source at this fraction of --target. "
                        "Default 1.0 = no balancing: split_0's train partition is "
                        "code_contests-only, so there is nothing to balance. Lower "
                        "it only if a future OCR revision adds platforms.")
    p.add_argument("--no-token-filter", action="store_true",
                   help="Skip loading the tokenizer; use a char-length proxy.")
    p.add_argument("--shuffle-buffer", type=int, default=10000,
                   help="Streaming shuffle buffer for WITHIN-source variety "
                        "(0 = no shuffle). Cross-source mixing comes from the "
                        "per-source cap + full scan, not this buffer.")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max-scan", type=int, default=1_500_000,
                   help="Safety cap on rows scanned. Default covers all of "
                        "split_0 (~568k) so late-in-stream platforms are reached.")
    args = p.parse_args()

    print("=" * 70)
    print(" OpenCodeReasoning split_0 -> v2 SFT corpus (reasoning traces).")
    excl = sorted(EXCLUDE_SOURCES) or "(none)"
    print(f" Firewall: split=='train'; excluded sources: {excl}.")
    print(f" Target {args.target} @ <= {args.max_seq_len} tok; "
          f"per-source cap {args.per_source_frac:g} of target.")
    print("=" * 70)

    length_of = make_length_fn(args.max_seq_len, use_tokenizer=not args.no_token_filter)
    per_source_cap = (int(args.target * args.per_source_frac)
                      if args.per_source_frac < 1.0 else args.target)

    ds = load_ocr_stream(seed=args.seed, shuffle_buffer=args.shuffle_buffer)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n_written = 0
    n_scanned = 0
    source_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()
    skipped = Counter()

    with open(args.output, "w", encoding="utf-8") as fh:
        for row in ds:
            n_scanned += 1
            if n_scanned > args.max_scan:
                print(f"Hit --max-scan={args.max_scan}; stopping early.")
                break

            # --- CHEAP checks first (no regex / no tokenizer) ---
            source = firewall_source(row)
            if source is None:
                skipped["firewall"] += 1
                continue
            if source_counts[source] >= per_source_cap:
                skipped["source_cap"] += 1
                continue

            # --- EXPENSIVE: parse reasoning/code, then tokenize-length filter ---
            rec = parse_row(row, source)
            if rec is None:
                skipped["no_reasoning_or_code"] += 1
                continue
            n_tok = length_of(rec["prompt"], rec["response"])
            if n_tok > args.max_seq_len:
                skipped["too_long"] += 1
                continue

            rec["n_tokens"] = n_tok
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            source_counts[source] += 1
            difficulty_counts[str(rec["difficulty"]).lower()] += 1

            if n_written % 500 == 0:
                print(f"  wrote {n_written}/{args.target} "
                      f"(scanned {n_scanned}) sources={dict(source_counts)}")
            if n_written >= args.target:
                break

    print("\nDone.")
    print(f"  examples written : {n_written}  -> {args.output}")
    print(f"  rows scanned     : {n_scanned}")
    print(f"  by source        : {dict(source_counts)}")
    print(f"  by difficulty    : {dict(difficulty_counts)}")
    print(f"  skipped          : {dict(skipped)}")
    if n_written < args.target:
        print(f"  NOTE: wrote fewer than --target ({n_written}<{args.target}). "
              f"Raise --max-scan or --per-source-frac, or lower --target.")
    print("\nFirewall reminder: APPS-*test* and all *-test/*-valid problems were "
          "excluded (split=='train' only). LiveCodeBench is never touched here.")


if __name__ == "__main__":
    main()
