# Experimental v0.1

Experimental research release v0.1. See the evaluation notes before use.

For research purposes. See README and EVALUATION-NOTES for the local comparison and its limitations.

## What is included

The preserved 83.14 GB mixed GGUF, source attribution, checksums and evaluation notes. The model package contains two shards. The code snapshot excludes weights, private data, raw logs, vendor source, local maintenance scripts and credentials. Exported tool paths are generic and may need configuration; this is not a complete portable rebuild pipeline. The requirements file covers earlier assembly tools, not every research dependency.

## How to run

Keep both GGUF shards together. With a compatible llama.cpp installed:

```text
llama-cli -m "Qwen3.8-Flash-Next-Abliterated-GSQ-RCO-IQ3_XXS-Calibrated-00001-of-00002.gguf" --lazy-mode on -ngl 999 -ncmoe 43 -t 12 -c 16384 -ctk q8_0 -ctv q5_1 -fa on -b 512 -ub 128 -rea off --reasoning-budget 0 -p "Explain how a rainbow forms." -n 256 --single-turn
```

The example uses 12 CPU threads and 43 CPU MoE layers from the measured profile, not a universal optimum. It allocates 16K context; this is not a claim about populated long-context performance. Allocate memory conservatively on other machines.

## Evidence and remaining limits

Latest short tests on the owner's machine: 21.13 tokens/s versus 16.16 for untouched ISTA, with a 9.6% larger model. Tests used the older frozen runtime. The updated engine passed a short generation check, but its throughput was not compared. Historical quality samples do not establish equivalence. Raw coding completions are primary; the formatting diagnostic is separate. Some runs left only 2–4 GiB free RAM. No vision or MTP weights are included.

Compression experiments did not establish a smaller replacement. No new model was assembled during release preparation. Treat this as an experimental release, not a fully accepted or maximally optimized model.

## Integrity

SHA256SUMS includes previously verified full model hashes and newly hashed small package files. Packaging checks sizes, modification times and hardlink identity. Local hardlinks share existing weights rather than duplicating 83 GB. Editing a linked GGUF affects every link; treat weights as immutable. A later upload will transfer the full logical model size.
