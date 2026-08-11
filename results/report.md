# Results — difficulty-stratified evaluation

APPS held-out **test** split, 150 problems per difficulty tier, evaluated with bigcode-evaluation-harness. Both models: 4-bit, fp16 compute, single-sample (temp 0.2), prompted with the SAME Qwen chat template the fine-tune was trained in (fair comparison). Two metrics are reported: **pass@1** = strict accuracy (a problem counts only if ALL hidden tests pass), and **average test-case pass rate** = the mean fraction of individual hidden tests passed (partial credit, more resolution). The v1 QLoRA fine-tune (trained on the single *shortest* APPS solution per problem, 1 epoch) **underperformed the base on every tier and on both metrics** — see README.md for the analysis and the v2 plan. No frontier column: LiveCodeBench was not run in v1.

### APPS — pass@1 (strict accuracy)

| Difficulty | Base (Qwen2.5-Coder-7B-Instruct) | Fine-tuned v1 (QLoRA/APPS) |
|---|---|---|
| Easy / Introductory | pass@1 0.0 | pass@1 0.0 |
| Medium / Interview | pass@1 7.3 | pass@1 2.0 |
| Hard / Competition | pass@1 0.0 | pass@1 0.0 |

_Problems per tier — Easy / Introductory: 150, Medium / Interview: 150, Hard / Competition: 150._

![APPS — pass@1 (strict accuracy)](figures/apps.png)

### APPS — average test-case pass rate

| Difficulty | Base (Qwen2.5-Coder-7B-Instruct) | Fine-tuned v1 (QLoRA/APPS) |
|---|---|---|
| Easy / Introductory | 0.8 | 0.2 |
| Medium / Interview | 30.2 | 7.4 |
| Hard / Competition | 0.5 | 0.0 |

_Problems per tier — Easy / Introductory: 150, Medium / Interview: 150, Hard / Competition: 150._

![APPS — average test-case pass rate](figures/apps_avg.png)

---

**Honesty notes:**
- Only cells sourced from our own eval pipeline are unmarked; *(cited)* cells are published leaderboard numbers for the same split.
- The hard/competition-tier gap is reported as-is, not downplayed.
- Our 7B is evaluated **4-bit quantized** (base and fine-tuned identically), the realistic local-deploy setting; frontier numbers are full-precision API.
- pass@k settings (k, samples, temperature, precision) are recorded per run in the `results/apps/` and `results/livecodebench/` metrics files.
