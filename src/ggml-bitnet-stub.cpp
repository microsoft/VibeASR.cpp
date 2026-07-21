// Stub implementation for ggml_bitnet_transform_tensor
// Only used when GGML_BITNET_ARM_TL1 or GGML_BITNET_X86_TL2 is enabled.
// VibeASR uses the I2_S MAD path and does not need LUT transforms.

#include "ggml-bitnet.h"
#include "ggml.h"

void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    (void)tensor;
    // No-op: LUT transform not needed for I2_S MAD path
}
