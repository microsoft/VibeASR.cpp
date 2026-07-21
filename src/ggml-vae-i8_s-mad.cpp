#include <vector>
#include <type_traits>
#include <assert.h>
#include <cmath>
#include <cstring>
#include "ggml-vae-i8_s-mad.h"
#include "ggml-cpu-impl.h"
#include "lm-config.h"
#include "vae-config.h"

#define QK_I8_S 32

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#include <immintrin.h>
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#endif

void ggml_vec_dot_i8_i8_1x1(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__) || defined(__AVX__)
    const int8_t * x = (int8_t *)vx;
    const int8_t * y = (int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {

        __m256i accu = _mm256_setzero_si256();
        const int8_t * x_row = x + row * bx;

        for (int i = 0; i < group32_num; i++) {
            const int8_t * px = x_row + i * 1024;
            const int8_t * py = y + i * 1024;
            __m256i accu32 = _mm256_setzero_si256();

            for (int j = 0; j < 32; j++) {
                __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                __m256i dot = _mm256_maddubs_epi16(ax, sy);

                accu32 = _mm256_add_epi16(accu32, dot);

                px += 32;
                py += 32;
            }
            accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, one16), accu);
        }

        for (int i = 0; i < groupla_num; i++) {
            __m256i accula = _mm256_setzero_si256();
            const int8_t * px = x_row + group32_num * 1024;
            const int8_t * py = y + group32_num * 1024;

            for (int j = 0; j < la_num; j++) {
                __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                __m256i dot = _mm256_maddubs_epi16(ax, sy);

                accula = _mm256_add_epi16(accula, dot);

                px += 32;
                py += 32;
            }
            accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, one16));
        }

        int sumi = hsum_i32_8(accu);
        s[row] = sumi;
    }
#endif
}

void ggml_vec_dot_i8_i8_1xN(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__) || defined(__AVX__)
    const int8_t * x = (int8_t *)vx;
    const int8_t * y = (int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row += VAE_PARALLEL_SIZE) {
        __m256i accu[VAE_PARALLEL_SIZE];
        const int8_t * x_row[VAE_PARALLEL_SIZE];
        for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            accu[rb] = _mm256_setzero_si256();
            x_row[rb] = x + (row + rb) * bx;
        }

        for (int i = 0; i < group32_num; i++) {
            const int8_t * px[VAE_PARALLEL_SIZE];
            __m256i accu32[VAE_PARALLEL_SIZE];

            for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + i * 1024;
                accu32[rb] = _mm256_setzero_si256();
            }

            const int8_t * py = y + i * 1024;

            for (int j = 0; j < 32; j++) {

                __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++)
                {
                    __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px[rb]));

                    const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                    const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                    __m256i dot = _mm256_maddubs_epi16(ax, sy);

                    accu32[rb] = _mm256_add_epi16(accu32[rb], dot);

                    px[rb] += 32;
                }
                py += 32;
            }
            for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[rb], one16), accu[rb]);
            }
        }

        for (int i = 0; i < groupla_num; i++) {

            const int8_t * py = y + group32_num * 1024;
            const int8_t * px[VAE_PARALLEL_SIZE];
            __m256i accula[VAE_PARALLEL_SIZE];

            for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + group32_num * 1024;
                accula[rb] = _mm256_setzero_si256();
            }

            for (int j = 0; j < la_num; j++) {

                __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {

                    __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px[rb]));

                    const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                    const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                    __m256i dot = _mm256_maddubs_epi16(ax, sy);

                    accula[rb] = _mm256_add_epi16(accula[rb], dot);

                    px[rb] += 32;
                }
                py += 32;
            }
            for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(accula[rb], one16));
            }
        }

        for(int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            int sumi = hsum_i32_8(accu[rb]);
            s[row + rb] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_Nx1(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__) || defined(__AVX__)
    const int8_t * x = (int8_t *)vx;
    const int8_t * y = (int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i one16 = _mm256_set1_epi16(1);

    for (int col = 0; col < nrc; col += VAE_PARALLEL_SIZE) {

        __m256i accu[VAE_PARALLEL_SIZE];

        for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            accu[cb] = _mm256_setzero_si256();
        }

        for (int i = 0; i < group32_num; i++) {

            __m256i accu32[VAE_PARALLEL_SIZE];

            for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu32[cb] = _mm256_setzero_si256();
            }

            for (int j = 0; j < 32; j++) {

                const int8_t * px = x + (i * 32 + j) * 32;

                __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px));

                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {

                    const int8_t * py = y + (col + cb) * by + (i * 32 + j) * 32;

                    __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                    const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                    const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                    __m256i dot = _mm256_maddubs_epi16(ax, sy);

                    accu32[cb] = _mm256_add_epi16(accu32[cb], dot);
                }
            }

            for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[cb], one16), accu[cb]);
            }
        }

        for (int i = 0; i < groupla_num; i++) {

            __m256i accula[VAE_PARALLEL_SIZE];

            for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accula[cb] = _mm256_setzero_si256();
            }

            for (int j = 0; j < la_num; j++) {

                const int8_t * px = x + (group32_num * 32 + j) * 32;

                __m256i xq8 = _mm256_loadu_si256((const __m256i*)(px));

                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {

                    const int8_t * py = y + (col + cb) * by + (group32_num * 32 + j) * 32;

                    __m256i yq8 = _mm256_loadu_si256((const __m256i*)(py));

                    const __m256i ax = _mm256_sign_epi8(xq8, xq8);
                    const __m256i sy = _mm256_sign_epi8(yq8, xq8);
                    __m256i dot = _mm256_maddubs_epi16(ax, sy);

                    accula[cb] = _mm256_add_epi16(accula[cb], dot);
                }
            }

            for(int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = _mm256_add_epi32(accu[cb], _mm256_madd_epi16(accula[cb], one16));
            }
        }

        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            int sumi = hsum_i32_8(accu[cb]);
            s[(col + cb) * bs] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    if (nrc % VAE_PARALLEL_SIZE == 0) {
#if defined(VAE_ACT_PARALLEL)
        ggml_vec_dot_i8_i8_Nx1(n, s, bs, vx, bx, vy, by, nrc);
#else
        ggml_vec_dot_i8_i8_1xN(n, s, bs, vx, bx, vy, by, nrc);
#endif
    } else {
        ggml_vec_dot_i8_i8_1x1(n, s, bs, vx, bx, vy, by, nrc);
    }
}

void ggml_vec_dot_i8_i8_n4_col8(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {

        const int8_t * vy_row = vy + row * 4;
        const int8_t * vx_row = vx;
        
        __m256i qx = _mm256_loadu_si256((const __m256i *)vx_row);
        
        uint32_t vy_32;
        memcpy(&vy_32, vy_row, sizeof(uint32_t));
        __m256i qy = _mm256_set1_epi32(vy_32);
        
        __m256i acc_i32;
        
#if __AVXVNNIINT8__
        acc_i32 = _mm256_setzero_si256();
        acc_i32 = _mm256_dpbssd_epi32(acc_i32, qx, qy);
#else
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i dot = _mm256_maddubs_epi16(ax, sy);
        acc_i32 = _mm256_madd_epi16(dot, one16);
#endif
        
        int32_t sums[8];
        _mm256_storeu_si256((__m256i *)sums, acc_i32);
        
        for (int i = 0; i < 8; i++) {
            s[row * bs + i] = sums[i];
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n8_col4(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 8;
        const int8_t * vx_row = vx;
        
        __m256i qx = _mm256_loadu_si256((const __m256i *)vx_row);
        
        uint64_t vy_64;
        memcpy(&vy_64, vy_row, sizeof(uint64_t));
        __m256i qy = _mm256_set_epi64x(vy_64, vy_64, vy_64, vy_64);
        
        __m256i acc_i32;
        
#if __AVXVNNIINT8__
        acc_i32 = _mm256_setzero_si256();
        acc_i32 = _mm256_dpbssd_epi32(acc_i32, qx, qy);
#else
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i dot = _mm256_maddubs_epi16(ax, sy);
        acc_i32 = _mm256_madd_epi16(dot, one16);
#endif
        
        __m256i sum_h1 = _mm256_hadd_epi32(acc_i32, acc_i32);
        
        int32_t sums[8];
        _mm256_storeu_si256((__m256i *)sums, sum_h1);
        
        s[row * bs + 0] = sums[0];
        s[row * bs + 1] = sums[1];
        s[row * bs + 2] = sums[4];
        s[row * bs + 3] = sums[5];
    }
#endif
}

void ggml_vec_dot_i8_i8_n16_col2(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {
    
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 16;
        const int8_t * vx_row = vx;
        
        __m256i qx = _mm256_loadu_si256((const __m256i *)vx_row);
        
        __m128i vy_128 = _mm_loadu_si128((const __m128i *)vy_row);
        __m256i qy = _mm256_set_m128i(vy_128, vy_128);
        
        __m256i acc_i32;
        
#if __AVXVNNIINT8__
        acc_i32 = _mm256_setzero_si256();
        acc_i32 = _mm256_dpbssd_epi32(acc_i32, qx, qy);
#else
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i dot = _mm256_maddubs_epi16(ax, sy);
        acc_i32 = _mm256_madd_epi16(dot, one16);
#endif
        
        __m256i sum_h1 = _mm256_hadd_epi32(acc_i32, acc_i32);
        __m256i sum_h2 = _mm256_hadd_epi32(sum_h1, sum_h1);
        
        int32_t sums[8];
        _mm256_storeu_si256((__m256i *)sums, sum_h2);
        
        s[row * bs + 0] = sums[0];
        s[row * bs + 1] = sums[4];
    }
#endif
}

void ggml_vec_dot_i8_i8_batch_n8(
    int32_t * dst_data,
    const int8_t * weight_data,
    const int8_t * input_data,
    int64_t ne00,
    int64_t ne01,
    int64_t ne02,
    int64_t ne10,
    int64_t ne11) {
    
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i one16 = _mm256_set1_epi16(1);
    
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input = input_data + batch * ne10 * ne11;
        int32_t * output = dst_data + batch * ne11;

        int64_t weight_i64;
        memcpy(&weight_i64, weight, sizeof(int64_t));
        __m256i w_vec = _mm256_set_epi64x(weight_i64, weight_i64, weight_i64, weight_i64);
        
        int64_t col;
        for (col = 0; col + 3 < ne11; col += 4) {

            __m256i i_vec = _mm256_set_epi64x(
                *(int64_t *)(input + (col + 3) * ne10),
                *(int64_t *)(input + (col + 2) * ne10),
                *(int64_t *)(input + (col + 1) * ne10),
                *(int64_t *)(input + (col + 0) * ne10)
            );

            __m256i acc_i32;
#if __AVXVNNIINT8__
            acc_i32 = _mm256_setzero_si256();
            acc_i32 = _mm256_dpbssd_epi32(acc_i32, i_vec, w_vec);
#else
            const __m256i ax = _mm256_sign_epi8(i_vec, i_vec);
            const __m256i sy = _mm256_sign_epi8(w_vec, i_vec);
            __m256i dot = _mm256_maddubs_epi16(ax, sy);
            acc_i32 = _mm256_madd_epi16(dot, one16);
#endif
            __m256i sum_h1 = _mm256_hadd_epi32(acc_i32, acc_i32);
            
            int32_t sums[8];
            _mm256_storeu_si256((__m256i *)sums, sum_h1);
            
            output[col + 0] = sums[0];
            output[col + 1] = sums[1];
            output[col + 2] = sums[4];
            output[col + 3] = sums[5];
        }
        
        for (; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#endif
}
