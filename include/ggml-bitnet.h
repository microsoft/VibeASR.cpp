// ggml-bitnet.h - BitNet LUT kernel declarations
// Used by llama.cpp for TL1/TL2 quantization path (optional).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

// Transform tensor for BitNet LUT inference (TL1/TL2 path only)
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
