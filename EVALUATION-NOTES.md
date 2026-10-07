# Evaluation notes

All comparisons are against untouched ISTA GSQ-RCO IQ3_XXS. Charts show local measurements, not results reported by ISTA on its own hardware.

## Speed and size

Latest short averages: ISTA **16.16 tokens/s**, our model **21.13**. Control batches bracketed the baseline tests and differed by less than 1%; each batch used five 512-token generations. Hardware: i7-12700K, RTX 5070 Ti 16 GB, 64 GB RAM and NVMe; frozen llama.cpp 11199 / `86a24a182`.

Settings: 12 threads, 43 CPU MoE layers, GPU layers 999, lazy loading, Q8_0/Q5_1 KV cache, flash attention, 16K allocated context; speed batch/microbatch 2048/512. Prompts were short. This is not a populated 16K or long-conversation speed result. Updated-engine speed is unmeasured.

Total file sizes: ISTA **75,839,998,528 bytes**, ours **83,144,446,848 bytes**. Decimal GB includes the 28.80 GB lookup shard. File size is not resident RAM usage.

## Capability samples

| Task | Sample | ISTA correct | Ours correct |
|---|---:|---:|---:|
| MMLU | 456 | 386 | 377 |
| GSM8K | 256 | 226 | 222 |
| Strict IFEval | 128 | 107 | 109 |
| Raw HumanEval | 164 | 111 | 109 |

Historical answers use the earlier fixed local protocol, with quality batch/microbatch 512/128. These are sampled tasks, not full leaderboard runs; detailed generation provenance is incomplete for some older records. HumanEval outputs executed only in constrained offline containers. Raw completions are primary; the formatting diagnostic is excluded.

The MMLU/GSM8K/IFEval average difference is **−0.66 percentage points**, paired-bootstrap 95% interval **−2.79 to +1.41**. Coding difference: **−1.22 points**, interval **−10.37 to +7.93**. Results do not establish equal quality or a clear coding gain. Ablation and quantization both changed, so their effects cannot be separated here.

High precision reference and populated long-context validation remain unfinished. Compression trials did not establish a smaller replacement. This package preserves the 83.14 GB experimental model.
