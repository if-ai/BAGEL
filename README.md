# ComfyUI-Bagel

A ComfyUI custom node package based on the BAGEL-7B-MoT multimodal model.

## About BAGEL

<p align="center">
  <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="480"/>
</p>

BAGEL is an open-source multimodal foundation model with 7B active parameters (14B total) that adopts a Mixture-of-Transformer-Experts (MoT) architecture. It is designed for multimodal understanding and generation tasks, outperforming top-tier open-source VLMs like Qwen2.5-VL and InternVL-2.5 on standard multimodal understanding leaderboards, and delivering text-to-image quality competitive with specialist generators such as SD3.

## Features

- **Text-to-Image Generation**: Generate high-quality images using natural language prompts
- **Image Editing**: Edit existing images based on textual descriptions  
- **Image Understanding**: Perform Q&A and analysis on images
- **Reasoning Process Display**: Optionally display the model's reasoning process

## Installation

### 1. Download Model
The BAGEL-7B-MoT model will be automatically downloaded to `models/bagel/BAGEL-7B-MoT/` when first used. You can also manually download it.

**Original BAGEL-7B-MoT (bfloat16):**
```bash
# Clone model using git lfs (recommended)
git lfs install
git clone https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT models/bagel/BAGEL-7B-MoT

# Or use huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance-Seed/BAGEL-7B-MoT', local_dir='models/bagel/BAGEL-7B-MoT')"
```

**INT8 GGUF Model (For GGUF-specific loaders):**
Download the `ggml-bytedance-BAGEL-7B-MoT-f16-IQ4_XS.gguf` file (or other desired GGUF variant) from [Gapeleon/bytedance_BAGEL-7B-MoT-INT8](https://huggingface.co/Gapeleon/bytedance_BAGEL-7B-MoT-INT8/tree/main) and place it in a dedicated GGUF model directory (e.g., `models/gguf/`).
*(Note: These ComfyUI-BAGEL nodes currently load `.safetensors` based models. For GGUF, use a GGUF-compatible loader node.)*

**INT8 Safetensors Model (For these ComfyUI-BAGEL nodes):**
If you wish to use an INT8 quantized model in `.safetensors` format with *these* nodes, download `model_int8.safetensors` and other necessary files (like `config.json`, `llm_config.json`, `vit_config.json`, `ae.safetensors`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.json`) from a repository like [Gapeleon/bytedance_BAGEL-7B-MoT-INT8](https://huggingface.co/Gapeleon/bytedance_BAGEL-7B-MoT-INT8/tree/main). Place these files into a directory such as `models/bagel/BAGEL-7B-MoT-INT8/`. The loader will expect `model_int8.safetensors` as the main weights file when "int8" precision is selected.

**FP8 Model (Experimental, `.safetensors` based):**
The FP8 version can be found at [meimeilook/BAGEL-7B-MoT-FP8](https://huggingface.co/meimeilook/BAGEL-7B-MoT-FP8/tree/main).
Download and place it in `models/bagel/BAGEL-7B-MoT-FP8/`.
*(Note: This FP8 model is not yet fully compatible with the current nodes. Use with caution.)*

The expected directory structure for models is:
```
ComfyUI/
├── models/
│   ├── bagel/
│   │   ├── BAGEL-7B-MoT/ (for the original bfloat16 model)
│   │   ├── BAGEL-7B-MoT-INT8/ (for INT8 .safetensors model, e.g., model_int8.safetensors)
│   │   └── BAGEL-7B-MoT-FP8/ (for FP8 .safetensors models)
│   ├── gguf/ (example directory for GGUF models)
```

### 2. Install Dependencies
Install base dependencies:
```bash
pip install -r requirements.txt
```

### 3. Flash Attention Installation (Optional, Recommended for Speed)
Flash Attention can significantly speed up inference. Install pre-built wheels matching your system configuration (Python, CUDA, PyTorch versions).

**Linux:**
Find wheels at [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.1.0).
Example: `pip install flash_attn-...whl`

**Windows:**
Find wheels at [Dao-AILab/flash-attention Releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.6).
Example: `pip install flash_attn-...whl`

**Important:** Ensure your Python, CUDA, and PyTorch versions are compatible with the chosen wheel.

### 4. INT8 (Q8) Model Specific Setup (Optional)
If you plan to use the INT8 GGUF model and want to potentially improve quality with specific kernels (once GGUF support is added to the nodes):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install packaging wheel ninja setuptools
pip install --no-build-isolation git+https://github.com/Lightricks/LTX-Video-Q8-Kernels.git
```
*(Note: This step is currently for future GGUF compatibility and might not be immediately usable with the existing nodes.)*

### 5. Restart ComfyUI
Restart ComfyUI to load the new nodes.

## Workflows

### Text-to-Image Generation
![text to image workflow](example_workflows/bagel_text_to_image.png)
Generate high-quality images from text descriptions. Suitable for creative design and content generation.

### Image Editing Workflow
![image editing workflow](example_workflows/bagel_image_edit.png)
Edit existing images based on textual descriptions, supporting local modifications and style adjustments.

### Image Understanding Workflow
![image understanding workflow](example_workflows/bagel_image_understanding.png)
Analyze and answer questions about image content, suitable for content understanding and information extraction.

## Related Links

- [BAGEL Official Paper](https://arxiv.org/abs/2505.14683)
- [BAGEL Model Homepage](https://bagel-ai.org/)
- [BAGEL HF Demo (Official)](https://huggingface.co/spaces/ByteDance-Seed/BAGEL)
- [Hugging Face Model (Original bfloat16)](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- [Hugging Face Model (INT8 GGUF by Gapeleon)](https://huggingface.co/Gapeleon/bytedance_BAGEL-7B-MoT-INT8/tree/main)
- [Hugging Face Model (FP8 by meimeilook)](https://huggingface.co/meimeilook/BAGEL-7B-MoT-FP8/tree/main)
- [Online Demo (Community)](https://demo.bagel-ai.org/)
- [Discord Community](https://discord.gg/Z836xxzy)
- [Comfy-Deploy LLM Toolkit](https://github.com/comfy-deploy/comfyui-llm-toolkit)
- [ComfyUI-BAGEL GitHub Repo](https://github.com/neverbiasu/ComfyUI-BAGEL/tree/master)

## License

This project is licensed under the Apache 2.0 License. Please refer to the official license terms for the use of the BAGEL model.

## Contribution

Contributions are welcome! Please submit issue reports and feature requests. If you wish to contribute code, please create an issue to discuss your ideas first.

## FAQ

### 1. VRAM Requirements
VRAM requirements vary significantly based on the model version and precision:
- **Original BAGEL-7B-MoT (bfloat16, `.safetensors`)**: The official recommendation for generating a 1024×1024 image is over 80GB GPU memory.
  - **Single GPU**: A100 (40GB) takes approximately 340-380 seconds per image.
  - **Multi-GPU**: 3 RTX3090 GPUs (24GB each) complete the task in about 1 minute.
- **Compressed Model (DFloat11, `.safetensors`)**: Using the DFloat11 version requires only 22GB VRAM and can run on a single 24GB GPU, with peak memory usage around 21.76GB (A100) and generation time of approximately 58 seconds. (This refers to a specific version, check model card for details).
- **FP8 Model (`.safetensors`)**: The [BAGEL-7B-MoT-FP8 model by meimeilook](https://huggingface.co/meimeilook/BAGEL-7B-MoT-FP8/tree/main) aims to reduce VRAM further. This is loaded by selecting "fp8..." precision in the BAGEL Model Loader node. *Note: Currently not fully compatible with these nodes.*
- **INT8 Safetensors Model (`model_int8.safetensors`)**: The [BAGEL-7B-MoT-INT8 by Gapeleon](https://huggingface.co/Gapeleon/bytedance_BAGEL-7B-MoT-INT8/tree/main) also provides `model_int8.safetensors`. This can be loaded by selecting "int8" precision in the BAGEL Model Loader node. It offers reduced VRAM compared to bfloat16.
- **INT8 GGUF Model**: The GGUF version from the same repository offers significant VRAM reduction and is designed for broader hardware compatibility (e.g., CPU inference, different GPU architectures) when used with a GGUF-specific loader. *Note: These ComfyUI-BAGEL nodes do not directly load GGUF files; use a separate GGUF loader node for those.*

For more details on the original model's VRAM, visit the [GitHub issue](https://github.com/ByteDance-Seed/Bagel/issues/4).

### 2. Quantized Versions (FP8, INT8)
- **FP8 (`.safetensors`)**: An FP8 version ([meimeilook/BAGEL-7B-MoT-FP8](https://huggingface.co/meimeilook/BAGEL-7B-MoT-FP8/tree/main)) is available. Select the "fp8..." precision in the BAGEL Model Loader. It is not yet fully compatible with these custom nodes.
- **INT8 (`.safetensors`)**: An INT8 version is available, for example, `model_int8.safetensors` from [Gapeleon/bytedance_BAGEL-7B-MoT-INT8](https://huggingface.co/Gapeleon/bytedance_BAGEL-7B-MoT-INT8/tree/main). Select the "int8" precision in the BAGEL Model Loader for these files.
- **INT8 (GGUF)**: A GGUF version from the same Gapeleon repository is also available. This format is recommended for users seeking the lowest VRAM and widest compatibility (including CPU) but requires a GGUF-specific loader node, not the ones provided in this package.

### 3. Flash Attention
Flash Attention is recommended for faster inference, especially on compatible NVIDIA GPUs. See the "Flash Attention Installation" section for setup instructions. Remember to match Python, CUDA, and PyTorch versions.

### 4. NameError: 'Qwen2Config' is not defined
This issue is likely related to environment or dependency problems, or an incorrect installation of the BAGEL model components. Ensure all dependencies from `requirements.txt` are installed and the model files are correctly placed. For more information, refer to [this GitHub issue](https://github.com/neverbiasu/ComfyUI-BAGEL/issues/7).

### 5. Future GGUF Support
Support for GGUF model versions (like the INT8 GGUF) is planned for a future update to these custom nodes. This will allow for easier use of quantized models with potentially lower VRAM requirements.
*Correction: While GGUF is a popular format, these BAGEL nodes are primarily focused on the `.safetensors` based model structure. For GGUF, please use dedicated GGUF loader nodes available in the ComfyUI ecosystem.*
