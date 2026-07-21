#ifndef VAE_H
#define VAE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for VAE model
typedef struct vae_model vae_model_t;
typedef struct vae_context vae_context_t;

// Model parameters
struct vae_model_params {
    int32_t n_threads;
    bool use_gpu;
};

// Context parameters
struct vae_context_params {
    int32_t n_threads;
};

// Get default model parameters
struct vae_model_params vae_model_default_params(void);

// Get default context parameters
struct vae_context_params vae_context_default_params(void);

// Load VAE model from GGUF file
// Returns NULL on failure
vae_model_t* vae_load_model_from_file(
    const char* model_path,
    struct vae_model_params params);

// Free model
void vae_free_model(vae_model_t* model);

// Create context from model
vae_context_t* vae_new_context_with_model(
    vae_model_t* model,
    struct vae_context_params params);

// Free context
void vae_free(vae_context_t* ctx);

// Get model info
int32_t vae_model_acoustic_dim(const vae_model_t* model);
int32_t vae_model_semantic_dim(const vae_model_t* model);

// Encode audio to acoustic features
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * acoustic_dim], caller must allocate
// Returns number of frames on success, -1 on error
int32_t vae_encode_acoustic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output);

// Encode audio to semantic features
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * semantic_dim], caller must allocate
// Returns number of frames on success, -1 on error
int32_t vae_encode_semantic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output);

// Encode audio to acoustic features with timing
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * acoustic_dim], caller must allocate
// inference_time_ms: output parameter for inference time in milliseconds
// Returns number of frames on success, -1 on error
int32_t vae_encode_acoustic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms);

// Encode audio to semantic features with timing
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * semantic_dim], caller must allocate
// inference_time_ms: output parameter for inference time in milliseconds
// Returns number of frames on success, -1 on error
int32_t vae_encode_semantic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms);

#ifdef __cplusplus
}
#endif

#endif // VAE_H
