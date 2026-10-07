# Understanding the evaluation

The baseline is **untouched ISTA GSQ-RCO IQ3_XXS**. Both models were tested locally with fixed runtime settings. The charts show this project's measurements, not ISTA's results on its own hardware.

## Speed: what was measured?

Generation speed measures how quickly the model produces output after reading a prompt.

| Measurement | ISTA baseline | This release |
|---|---:|---:|
| Average generation speed | 16.16 tokens/s | 21.13 tokens/s |
| Total model download | 75.84 GB | 83.14 GB |

That is about **31% faster generation** and a **9.6% larger download** in this workload.

- Hardware: i7-12700K, RTX 5070 Ti 16 GB, 64 GB RAM, NVMe storage.
- Runtime: frozen llama.cpp build 11199 / `86a24a182`.
- Settings: 12 threads, 43 CPU MoE layers, 999 GPU layers, lazy loading, Q8_0/Q5_1 conversation cache and flash attention.
- Workload: short prompts, 512 generated tokens per run, five runs per batch. Control batches before and after baseline testing differed by less than 1%.
- Context allocation: 16K; speed batch/microbatch: 2048/512.

**Allocating 16K context does not mean the prompts contained 16K tokens.** These runs do not establish populated 16K or long-conversation speed. Updated-engine throughput and speed on other hardware are unmeasured.

Exact sizes are 75,839,998,528 bytes for ISTA and 83,144,446,848 bytes for this release. Decimal GB includes the 28.80 GB lookup shard. Download size is not resident RAM usage; some runs left only 2–4 GiB free RAM.

## Quality: what was measured?

| Task | What it checks | Sample | ISTA | This release |
|---|---|---:|---:|---:|
| MMLU | Knowledge questions | 456 | 386 correct / 84.65% | 377 / 82.68% |
| GSM8K | Maths word problems | 256 | 226 correct / 88.28% | 222 / 86.72% |
| Strict IFEval | Following explicit instructions | 128 | 107 passed / 83.59% | 109 / 85.16% |
| Raw HumanEval | Completing programming tasks | 164 | 111 passed / 67.68% | 109 / 66.46% |

These historical answers use the earlier fixed local protocol, with quality batch/microbatch 512/128. They are **local task samples, not full public leaderboard runs**. Detailed generation provenance is incomplete for some older records.

Generated coding answers ran only in constrained offline containers. The primary coding score uses raw completions; a separate formatting diagnostic is excluded from the chart and scores above.

### How certain are these differences?

For the first three tasks, the average change was **−0.66 percentage points**. Resampling the paired answers gives a 95% uncertainty interval of **−2.79 to +1.41 points**. The coding change was **−1.22 points**, with an interval of **−10.37 to +7.93**.

These intervals include both a loss and a gain. The samples do not establish equal quality or a clear coding improvement. A percentage point is the direct difference between two percentages.

## What remains unknown?

- Long-context performance and memory headroom across machines.
- Comparison with a complete high-precision reference model.
- How much of each result comes from ablation versus compression: both changed together.
- Whether a smaller replacement can meet the same goals; compression trials did not establish one.

The released artifact is the preserved **83.14 GB experimental model**, not a claim that size, speed or quality have reached their limits.
