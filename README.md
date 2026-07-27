<h1 align="center">VibeASR.cpp</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://huggingface.co/microsoft/VibeVoice-ASR-BitNet"><img src="https://img.shields.io/badge/🤗-Models-orange.svg" alt="HuggingFace"></a>
  <a href="https://arxiv.org/abs/2607.21075"><img src="https://img.shields.io/badge/📄-Tech_Report-red.svg" alt="Tech Report"></a>
</p>

---

**VibeASR.cpp** is the official inference runtime for **VibeVoice-ASR-BitNet** — enabling real-time multilingual speech recognition on CPU through heterogeneous quantization (I8\_S for VAE + I2\_S for LM).

To enable efficient edge CPU deployment, we replace the original Qwen2.5-7B language model with Qwen2.5-1.5B, achieving only modest accuracy degradation (1–4% absolute WER increase) while reducing the total model size from 4.62 GB to 1.58 GB. Combined with custom SIMD kernels and operator fusion in the ggml framework, VibeVoice-ASR-BitNet achieves **1.6–2.3× faster** inference than Whisper.cpp at comparable model sizes, with real-time capability (RTF < 1) on low-resource CPUs.

<p align="center">
  <img src="media/report_overview.png" width="92%"/>
</p>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2607.21075">Tech Report</a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/microsoft/VibeVoice-ASR-BitNet">Models</a> &nbsp;|&nbsp;
  🏠 <a href="https://aka.ms/GeneralAI">GeneralAI</a>
</p>

---

## Key Results

### Model Size

<div align="center">

| Component | VibeVoice-ASR-1.5B (FP16) | VibeVoice-ASR-BitNet | Compression |
|:---------:|:-------------------------:|:--------------------:|:-----------:|
| VAE Tokenizer | 1.31 GB | 0.65 GB | 2.0× |
| LM Decoder | 3.32 GB | 0.92 GB | 3.6× |
| **Total** | **4.62 GB** | **1.58 GB** | **2.9×** |

</div>

### Inference Performance

<div align="center">

**AMD EPYC 7V13 (AVX2+FMA)**

| | 1T | 2T | 3T | 4T | 6T | 8T |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTF | 1.98 | 1.08 | **0.77** | **0.63** | **0.49** | **0.42** |
| vs. Whisper.cpp | 2.28× | 2.12× | 1.86× | 1.86× | 1.71× | 1.55× |

**Apple M4 (ARM NEON, 4P+6E, 16GB)**

| | 1T | 2T | 3T | 4T | 6T | 8T |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTF | 1.15 | **0.66** | **0.51** | **0.42** | **0.52** | **0.45** |

</div>

> RTF (Real-Time Factor) on 20s audio input. **Bold** = RTF < 1 (real-time).

### Accuracy (WER%)

<div align="center">

| Benchmark | VibeVoice-ASR-7B (FP16) | VibeVoice-ASR-BitNet | Parakeet | Whisper | SenseVoice | FunASR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MLC-EN | 7.82 | **8.25** | 8.40 | 13.57 | 12.39 | 11.36 |
| MLC-FR | 16.03 | 17.41 | — | — | — | — |
| MLC-IT | 15.67 | 17.23 | — | — | — | — |
| MLC-KO | 9.83 | 11.15 | — | — | — | — |
| MLC-PT | 22.41 | 24.87 | — | — | — | — |
| MLC-VI | 20.15 | 22.38 | — | — | — | — |
| AISHELL4 | 19.83 | 27.45 | — | — | 22.52 | **20.41** |
| AMI-ihm | 17.42 | **21.36** | 21.92 | 27.07 | 30.81 | 32.07 |
| AMI-sdm | 24.18 | **25.87** | 26.33 | 36.92 | 48.11 | 40.17 |
| AliMeeting | 36.21 | 40.58 | — | — | **38.75** | 39.27 |
| Fleurs-en | 4.73 | 5.21 | 4.09 | **3.99** | 6.84 | 4.93 |
| Fleurs-zh | 7.92 | 8.35 | — | — | **5.56** | 7.00 |
| Libri-clean | 2.17 | 2.41 | **1.49** | 1.98 | 2.78 | 1.58 |
| Libri-other | 5.84 | 6.27 | **3.13** | 3.60 | 6.81 | 4.01 |
| VoxPopuli | 4.92 | **5.18** | 5.26 | 7.19 | 8.63 | 6.46 |

</div>

---

## Quick Start

### Requirements

- Python ≥ 3.9, CMake ≥ 3.14, GCC/Clang with C++11 support
- ~2 GB disk space (code + quantized models)

### One-Command Setup

```bash
git clone --recursive https://github.com/microsoft/VibeASR.cpp.git
cd VibeASR.cpp
pip install -r requirements.txt
python setup_env.py
```

### Manual Build

```bash
git clone --recursive https://github.com/microsoft/VibeASR.cpp.git
cd VibeASR.cpp

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Download pre-quantized models
pip install huggingface_hub
huggingface-cli download microsoft/VibeVoice-ASR-BitNet --local-dir models/vibeasr
# NOTE: The LM I2_S GGUF is architecture-specific.
#   x86: use vibeasr-lm-i2_s-embed-q6_k.gguf
#   ARM: use vibeasr-lm-i2_s-embed-q6_k_arm.gguf
```

---

## Usage

### Pre-quantized Models

> **⚠️ Architecture Note:** The LM (I2\_S) GGUF uses platform-specific bit-packing layouts — ARM (NEON) and x86 (AVX) versions are **not interchangeable**. The VAE (I8\_S) GGUF is architecture-independent.
>
> When downloading from [HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet):
> - **x86 (Intel/AMD):** use `vibeasr-lm-i2_s-embed-q6_k.gguf`
> - **ARM (Apple Silicon / aarch64):** use `vibeasr-lm-i2_s-embed-q6_k_arm.gguf`
>
> If you build from source with `llama-quantize`, the output GGUF will automatically match your host architecture.

### CLI Inference

```bash
# x86 (Intel / AMD)
./build/bin/asr_infer \
    --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --lm-model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf \
    --audio input.wav -t 4

# ARM (Apple Silicon / aarch64)
./build/bin/asr_infer \
    --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --lm-model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k_arm.gguf \
    --audio input.wav -t 4
```

### Web Demo (Gradio)

```bash
pip install gradio soundfile numpy

# Use the LM GGUF matching your architecture (see note above)
python demo/gradio_asr_demo.py --port 7860 \
    --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --lm-model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf  # or *_arm.gguf on ARM
```

---

## Model Conversion

For most users, downloading pre-quantized models from [HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) is recommended. To convert from SafeTensors yourself:

### Step 1: SafeTensors → F32 GGUF

```bash
# LM (BitNet) — handles weight preprocessing and config flattening automatically
python utils/convert_lm_to_gguf.py <safetensors-dir>

# VAE Tokenizer
python utils/convert_vae_to_gguf.py <safetensors-dir>
```

### Step 2: F32 GGUF → Quantized GGUF

```bash
# VAE Tokenizer: F32 → I8_S
./build/bin/llama-quantize \
    <safetensors-dir>/vibeasr-vae-encoder-f32.gguf \
    <safetensors-dir>/vibeasr-vae-encoder-i8_s.gguf \
    I8_S 1 1

# LM: F32 → I2_S (with Q6_K embeddings)
./build/bin/llama-quantize --token-embedding-type Q6_K \
    <safetensors-dir>/vibeasr-lm-f32.gguf \
    <safetensors-dir>/vibeasr-lm-i2_s-embed-q6_k.gguf \
    I2_S 1 1
```

---

## Citation

```bibtex
@article{xu2025vibeasrbitnet,
    title={VibeVoice-ASR-BitNet Technical Report},
    author={Xu, Songchen and Song, Ting and Huang, Shaohan and Peng, Zhiliang and Xia, Yan and Tu, Yujie and Huang, Xin and Yu, Jianwei and Dong, Li and Wei, Furu},
    journal={arXiv preprint arXiv:2607.21075},
    year={2025}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
