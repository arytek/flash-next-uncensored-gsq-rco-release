# Experimental release v0.1

For research purposes. The released model is available on [Hugging Face](https://huggingface.co/Methane/Qwen3.8-Flash-Next-Abliterated-GSQ-RCO-IQ3_XXS-Calibrated-GGUF).

## What is released?

- An **83.14 GB model in two GGUF shards**: 54.34 GB of main weights and a 28.80 GB lookup table. Both are required.
- Documentation, two comparison charts, source attribution and checksums.
- A separate GitHub research-code snapshot, with no model weights, private data, raw logs or credentials.

This release preserves the selected experimental model. Later compression trials did not establish a smaller replacement. No new weights were assembled during publication preparation.

## What changed from the baseline?

The model retains ISTA's existing GSQ-RCO compression for unchanged weights and freshly quantizes 146 edited tensors from an abliterated checkpoint. AtomicChat's recipe informed format and precision choices; its calibration text was used, but no AtomicChat weights are included.

It is **GSQ-RCO-based**, not a fresh full-model upstream GSQ/RCO run. See [source provenance](PROVENANCE.json).

## How to use it

Follow the [README download and launch instructions](README.md#use-the-released-model). Rebuilding from the research scripts is not required to run the downloaded model.

Text only; vision and MTP draft weights are not included. Use compatible llama.cpp and SSD storage, and leave memory for the conversation cache.

## Results and limits

Short local tests measured **21.13 tokens/s versus 16.16 for untouched ISTA**, with a **9.6% larger download**. The tested machine had an i7-12700K, RTX 5070 Ti 16 GB, 64 GB RAM and NVMe storage.

Historical quality samples do not prove equivalence. Long-conversation performance, updated-engine throughput and a full high-precision reference comparison remain unestablished. Some runs left only 2–4 GiB free RAM. See [evaluation notes](EVALUATION-NOTES.md).

The code is a research snapshot: intermediate data and external dependency trees are excluded, and the requirements file does not cover every research tool. A complete portable rebuild remains unfinished.

## Verify downloads

[SHA256SUMS](SHA256SUMS) in this GitHub repository covers its code and documentation. The model repository has its own SHA256SUMS, including both GGUF shards. Compare a downloaded file's SHA256 digest with the matching entry in the appropriate repository.

All model tensor bytes were checked before publication. Both model shards were fully hashed during upload, and the uploaded release files were verified against the audited inventory.
