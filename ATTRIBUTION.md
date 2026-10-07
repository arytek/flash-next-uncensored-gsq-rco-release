# Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): original model. Retain the included Qwen Community License 1.0.
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) and [GSQ authors](https://github.com/IST-DASLab/GSQ): retained compressed tensors and research method.
- [windowsxp811203](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated): ablated BF16 checkpoint used for 146 edited tensors.
- [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF): expert-down format and selective precision inspiration; [calibration text](https://huggingface.co/datasets/AtomicChat/calib-corpora). No AtomicChat model weights are included.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): GGUF tooling and runtime; external tooling retains its own licenses.

Source pins and scope are in [PROVENANCE.json](PROVENANCE.json). This package does not claim a full upstream GSQ/RCO rerun.
