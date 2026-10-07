# Flash Next · Abliterated GSQ-RCO-based GGUF

**An experimental Qwen3.8 Flash Next model for research purposes.** We combined ISTA's compressed model weights with freshly compressed weights from an abliterated checkpoint, using calibration and selective precision inspired by AtomicChat.

**[Download the model on Hugging Face](https://huggingface.co/Methane/Qwen3.8-Flash-Next-Abliterated-GSQ-RCO-IQ3_XXS-Calibrated-GGUF)** · [Model card](MODEL_CARD.md) · [Evaluation details](EVALUATION-NOTES.md) · [Credits](ATTRIBUTION.md)

This GitHub repository contains research code and documentation. The model files are hosted separately on Hugging Face. You do not need to rebuild the model to use it.

## What changed compared with ISTA?

| | Untouched ISTA IQ3_XXS | This release |
|---|---:|---:|
| Total download | 75.84 GB | 83.14 GB |
| Generation speed in short local tests | 16.16 tokens/s | 21.13 tokens/s |
| Weights from an abliterated checkpoint | Original ISTA weights | 146 edited tensors |

The measured trade-off is **about 31% faster generation with a 9.6% larger download**. These results apply to the tested machine and workload. Small benchmark samples do not prove equal quality.

![Local generation speed and total download size](assets/speed-size.png)

## How the model was built

1. **Keep most of ISTA's work.** Retain 1,078 tensors with their existing GSQ-RCO compression. A tensor is an array of model weights.
2. **Replace the edited weights.** Start from 146 tensors in an abliterated BF16 checkpoint and quantize them afresh. BF16 is the source's higher-precision weight format.
3. **Choose precision where it matters.** Use calibration text, format trials and targeted refinement to balance size and accuracy. AtomicChat's recipe inspired compression choices for the model's experts and selective precision; its calibration text was used, but no AtomicChat model weights were copied.
4. **Check the assembled model.** Correct the GGUF layout, verify all 1,224 tensors, then compare speed and capability samples against untouched ISTA IQ3_XXS.

**This is GSQ-RCO-based, not a new full-model run of the upstream GSQ/RCO method.** `IQ3_XXS` in the filenames identifies the retained source variant; the finished model uses several formats. Exact sources and format counts are in [PROVENANCE.json](PROVENANCE.json).

### Terms in plain language

| Term | Meaning here |
|---|---|
| Quantization | Store weights with fewer bits to reduce storage and memory traffic. |
| Mixed precision | Use different compression levels for different parts of the model. |
| Calibration | Use sample text to help decide which compression choices preserve useful outputs. |
| MoE / experts | A mixture-of-experts model uses selected groups of weights for each token. |
| Abliterated | Weights edited to alter response behaviour; the name does not guarantee unrestricted responses. |
| GGUF | The model file format used by llama.cpp. |

## Use the released model

1. Download **both** GGUF shards from the model page: the 54.34 GB main shard and the 28.80 GB lookup shard.
2. Keep them in the same folder on an SSD.
3. Use a compatible llama.cpp build supporting this architecture and lazy lookup loading. Open the first shard; the runtime loads the companion shard automatically.

```text
llama-cli -m "Qwen3.8-Flash-Next-Abliterated-GSQ-RCO-IQ3_XXS-Calibrated-00001-of-00002.gguf" -ngl 999 -ncmoe 43 --lazy-mode on -c 16384 -t 12 -ctk q8_0 -ctv q5_1 -fa on -b 512 -ub 128 -rea off --reasoning-budget 0 -p "Explain how a rainbow forms." -n 256 --single-turn
```

This example uses 12 CPU threads, 43 CPU expert layers and 16K allocated context. It was checked on an i7-12700K, RTX 5070 Ti 16 GB, 64 GB RAM and NVMe storage. Adjust the settings for other machines. **83.14 GB is download size, not a RAM requirement.** Some local tests left only 2–4 GiB of free RAM.

Text generation only: vision and MTP draft weights are not included.

## What the tests show

![Local knowledge, maths, instruction-following and coding samples](assets/quality.png)

Scores were slightly lower on knowledge, maths and coding samples, and slightly higher on instruction-following. The uncertainty is large enough that **quality equivalence remains unproven**. These are local samples, not full public leaderboard evaluations.

Speed measurements used short prompts and an older frozen llama.cpp build. Long-conversation speed and updated-engine speed were not established. See [evaluation notes](EVALUATION-NOTES.md) for settings, sample sizes and uncertainty.

## Explore the research code

| Area | Starting point |
|---|---|
| Compress the edited weights | [quantize_edited.py](tools/quantize_edited.py) |
| Assemble retained and edited weights | [assemble_calibrated.py](tools/assemble_calibrated.py) |
| Study targeted refinement | [refine_quantization.py](tools/refine_quantization.py) |
| Verify an assembled GGUF | [verify_calibrated.py](tools/verify_calibrated.py) |
| Run local evaluations | [evaluate_local.py](tools/evaluate_local.py) |

**This is a research snapshot, not a complete rebuild kit.** Some tools describe earlier or rejected trials. Intermediate data and external dependency trees are excluded; [requirements.txt](requirements.txt) covers only the earlier assembly tools. Paths and inputs may need configuration. Use the published model directly unless you intend to study the code.

## Licensing

Original project code uses the [MIT licence](CODE-LICENSE). Model weights and derivatives retain the [Qwen Community License 1.0](LICENSE); third-party tools retain their own licences. See [attribution](ATTRIBUTION.md).

For research purposes. Independently evaluate outputs before relying on them.
