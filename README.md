# Flash Next abliterated GGUF research tools

Research code for an experimental GSQ-RCO-based mixed GGUF. Read the [model card](MODEL_CARD.md), [evaluation notes](EVALUATION-NOTES.md) and [source provenance](PROVENANCE.json).

The preserved model is 83.14 GB. Latest short local tests measured 21.13 tokens/s versus 16.16 for untouched ISTA IQ3_XXS. These measurements do not establish equal quality, universal speed or long-context performance.

## Scope

Tools cover GGUF assembly, calibration, format trials, integrity checks and local evaluations. Exported paths are generic: place inputs below models/ and engines/, or configure the relevant tool. Full portable reproduction is unfinished; source pools, intermediate payloads, vendor dependencies, local maintenance scripts and raw experiment logs are excluded. The requirements file covers earlier assembly tools, not all research dependencies.

Keep both model shards together and use the direct llama-cli example in the model card. Model weights are separate from this code snapshot.

AtomicChat's recipe/corpus informed format choices; no AtomicChat weights are included. See [credits](ATTRIBUTION.md). Preserve the included base-model license. External tools retain their own licenses.

**For research purposes.** Independently evaluate outputs. Experimental release v0.1.

## Licensing

Original project code is available under the [MIT licence](CODE-LICENSE). Model weights and model derivatives retain the [Qwen Community License 1.0](LICENSE). Third-party tools retain their own licences; the code licence does not replace the model terms.
