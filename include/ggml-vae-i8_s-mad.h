#pragma once

#include "ggml.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


// INT8 × INT8 vec_dot implementation (output is int32 to avoid overflow)
void ggml_vec_dot_i8_i8(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

// Optimized INT8 × INT8 vec_dot for n=4 (process 8 columns simultaneously)
void ggml_vec_dot_i8_i8_n4_col8(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=8 (process 4 columns simultaneously)
void ggml_vec_dot_i8_i8_n8_col4(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=16 (process 2 columns simultaneously)
void ggml_vec_dot_i8_i8_n16_col2(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=2 (process 16 columns simultaneously, AVX only)
void ggml_vec_dot_i8_i8_n2_col16(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=4 (process 2 columns simultaneously, ARM only)
void ggml_vec_dot_i8_i8_n4_col2(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 vec_dot for n=2 (process 4 columns simultaneously, ARM only)
void ggml_vec_dot_i8_i8_n2_col4(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc);

// Optimized INT8 × INT8 depthwise convolution kernel for n=8
// Typical shape: [8,1,32] x [8,579200,32]
// Process 4 columns at a time to fill 256-bit register
void ggml_vec_dot_i8_i8_batch_n8(
    int32_t * dst_data,
    const int8_t * weight_data,
    const int8_t * input_data,
    int64_t ne00,  // 8
    int64_t ne01,  // 1
    int64_t ne02,  // batch (32)
    int64_t ne10,  // 8
    int64_t ne11  // cols (579200)
    );

#ifdef __cplusplus
}
#endif
