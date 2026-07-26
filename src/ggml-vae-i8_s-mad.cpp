#include <vector>
#include <type_traits>
#include <assert.h>
#include <cmath>
#include <cstring>
#include "ggml-vae-i8_s-mad.h"
#include "ggml-cpu-impl.h"
#include "lm-config.h"
#include "vae-config.h"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I8_S 32
#elif defined(__ARM_NEON)
#define QK_I8_S 8
#else
#define QK_I8_S 32
#endif

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
#elif defined(__ARM_NEON)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = la_num != 0 ? 1 : 0;

    for (int row = 0; row < nrc; row++) {
        int32x4_t accu = vdupq_n_s32(0);
        const int8_t * x_row = x + row * bx;

        for (int i = 0; i < group32_num; i++) {
            const int8_t * px = x_row + i * 32 * QK_I8_S;
            const int8_t * py = y + i * 32 * QK_I8_S;
#if defined(__ARM_FEATURE_DOTPROD)
            for (int j = 0; j < 32; j++) {
                int8x8_t xv = vld1_s8(px);
                int8x8_t yv = vld1_s8(py);
                int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                accu = vcombine_s32(vadd_s32(vget_low_s32(accu), d), vget_high_s32(accu));
                px += QK_I8_S;
                py += QK_I8_S;
            }
#else
            int16x8_t accu16 = vdupq_n_s16(0);
            for (int j = 0; j < 32; j++) {
                int8x8_t xv = vld1_s8(px);
                int8x8_t yv = vld1_s8(py);
                accu16 = vmlal_s8(accu16, xv, yv);
                px += QK_I8_S;
                py += QK_I8_S;
            }
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu16)));
            accu = vaddq_s32(accu, vmovl_high_s16(accu16));
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
            const int8_t * px = x_row + group32_num * 32 * QK_I8_S;
            const int8_t * py = y + group32_num * 32 * QK_I8_S;
#if defined(__ARM_FEATURE_DOTPROD)
            for (int j = 0; j < la_num; j++) {
                int8x8_t xv = vld1_s8(px);
                int8x8_t yv = vld1_s8(py);
                int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                accu = vcombine_s32(vadd_s32(vget_low_s32(accu), d), vget_high_s32(accu));
                px += QK_I8_S;
                py += QK_I8_S;
            }
#else
            int16x8_t accu16la = vdupq_n_s16(0);
            for (int j = 0; j < la_num; j++) {
                int8x8_t xv = vld1_s8(px);
                int8x8_t yv = vld1_s8(py);
                accu16la = vmlal_s8(accu16la, xv, yv);
                px += QK_I8_S;
                py += QK_I8_S;
            }
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu16la)));
            accu = vaddq_s32(accu, vmovl_high_s16(accu16la));
#endif
        }

        s[row] = vaddlvq_s32(accu);
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int row = 0; row < nrc; row++) {
        const int8_t * x_row = x + row * bx;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x_row[k] * (int32_t)y[k];
        }
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
#elif defined(__ARM_NEON)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = la_num != 0 ? 1 : 0;

    for (int row = 0; row < nrc; row += VAE_PARALLEL_SIZE) {
        int32x4_t accu[VAE_PARALLEL_SIZE];
        const int8_t * x_row[VAE_PARALLEL_SIZE];
        for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            accu[rb] = vdupq_n_s32(0);
            x_row[rb] = x + (row + rb) * bx;
        }

        for (int i = 0; i < group32_num; i++) {
            const int8_t * py = y + i * 32 * QK_I8_S;
#if defined(__ARM_FEATURE_DOTPROD)
            const int8_t * px[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + i * 32 * QK_I8_S;
            }
            for (int j = 0; j < 32; j++) {
                int8x8_t yv = vld1_s8(py);
                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                    int8x8_t xv = vld1_s8(px[rb]);
                    int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                    accu[rb] = vcombine_s32(vadd_s32(vget_low_s32(accu[rb]), d), vget_high_s32(accu[rb]));
                    px[rb] += QK_I8_S;
                }
                py += QK_I8_S;
            }
#else
            int16x8_t accu16[VAE_PARALLEL_SIZE];
            const int8_t * px[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu16[rb] = vdupq_n_s16(0);
                px[rb] = x_row[rb] + i * 32 * QK_I8_S;
            }
            for (int j = 0; j < 32; j++) {
                int8x8_t yv = vld1_s8(py);
                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                    int8x8_t xv = vld1_s8(px[rb]);
                    accu16[rb] = vmlal_s8(accu16[rb], xv, yv);
                    px[rb] += QK_I8_S;
                }
                py += QK_I8_S;
            }
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accu16[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accu16[rb]));
            }
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
            const int8_t * py = y + group32_num * 32 * QK_I8_S;
#if defined(__ARM_FEATURE_DOTPROD)
            const int8_t * px[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + group32_num * 32 * QK_I8_S;
            }
            for (int j = 0; j < la_num; j++) {
                int8x8_t yv = vld1_s8(py);
                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                    int8x8_t xv = vld1_s8(px[rb]);
                    int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                    accu[rb] = vcombine_s32(vadd_s32(vget_low_s32(accu[rb]), d), vget_high_s32(accu[rb]));
                    px[rb] += QK_I8_S;
                }
                py += QK_I8_S;
            }
#else
            int16x8_t accu16la[VAE_PARALLEL_SIZE];
            const int8_t * px[VAE_PARALLEL_SIZE];
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu16la[rb] = vdupq_n_s16(0);
                px[rb] = x_row[rb] + group32_num * 32 * QK_I8_S;
            }
            for (int j = 0; j < la_num; j++) {
                int8x8_t yv = vld1_s8(py);
                for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                    int8x8_t xv = vld1_s8(px[rb]);
                    accu16la[rb] = vmlal_s8(accu16la[rb], xv, yv);
                    px[rb] += QK_I8_S;
                }
                py += QK_I8_S;
            }
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accu16la[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accu16la[rb]));
            }
#endif
        }

        for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            s[row + rb] = vaddlvq_s32(accu[rb]);
        }
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int row = 0; row < nrc; row++) {
        const int8_t * x_row = x + row * bx;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x_row[k] * (int32_t)y[k];
        }
        s[row] = sumi;
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
#elif defined(__ARM_NEON)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb = n / QK_I8_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = la_num != 0 ? 1 : 0;

    for (int col = 0; col < nrc; col += VAE_PARALLEL_SIZE) {
        int32x4_t accu[VAE_PARALLEL_SIZE];
        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            accu[cb] = vdupq_n_s32(0);
        }

        for (int i = 0; i < group32_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)
            for (int j = 0; j < 32; j++) {
                const int8_t * px = x + (i * 32 + j) * QK_I8_S;
                int8x8_t xv = vld1_s8(px);
                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                    const int8_t * py = y + (col + cb) * by + (i * 32 + j) * QK_I8_S;
                    int8x8_t yv = vld1_s8(py);
                    int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                    accu[cb] = vcombine_s32(vadd_s32(vget_low_s32(accu[cb]), d), vget_high_s32(accu[cb]));
                }
            }
#else
            int16x8_t accu16[VAE_PARALLEL_SIZE];
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu16[cb] = vdupq_n_s16(0);
            }
            for (int j = 0; j < 32; j++) {
                const int8_t * px = x + (i * 32 + j) * QK_I8_S;
                int8x8_t xv = vld1_s8(px);
                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                    const int8_t * py = y + (col + cb) * by + (i * 32 + j) * QK_I8_S;
                    int8x8_t yv = vld1_s8(py);
                    accu16[cb] = vmlal_s8(accu16[cb], xv, yv);
                }
            }
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = vaddq_s32(accu[cb], vmovl_s16(vget_low_s16(accu16[cb])));
                accu[cb] = vaddq_s32(accu[cb], vmovl_high_s16(accu16[cb]));
            }
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)
            for (int j = 0; j < la_num; j++) {
                const int8_t * px = x + (group32_num * 32 + j) * QK_I8_S;
                int8x8_t xv = vld1_s8(px);
                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                    const int8_t * py = y + (col + cb) * by + (group32_num * 32 + j) * QK_I8_S;
                    int8x8_t yv = vld1_s8(py);
                    int32x2_t d = vdot_s32(vdup_n_s32(0), xv, yv);
                    accu[cb] = vcombine_s32(vadd_s32(vget_low_s32(accu[cb]), d), vget_high_s32(accu[cb]));
                }
            }
#else
            int16x8_t accu16la[VAE_PARALLEL_SIZE];
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu16la[cb] = vdupq_n_s16(0);
            }
            for (int j = 0; j < la_num; j++) {
                const int8_t * px = x + (group32_num * 32 + j) * QK_I8_S;
                int8x8_t xv = vld1_s8(px);
                for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                    const int8_t * py = y + (col + cb) * by + (group32_num * 32 + j) * QK_I8_S;
                    int8x8_t yv = vld1_s8(py);
                    accu16la[cb] = vmlal_s8(accu16la[cb], xv, yv);
                }
            }
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = vaddq_s32(accu[cb], vmovl_s16(vget_low_s16(accu16la[cb])));
                accu[cb] = vaddq_s32(accu[cb], vmovl_high_s16(accu16la[cb]));
            }
#endif
        }

        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            s[(col + cb) * bs] = vaddlvq_s32(accu[cb]);
        }
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int col = 0; col < nrc; col++) {
        const int8_t * y_col = y + col * by;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x[k] * (int32_t)y_col[k];
        }
        s[col * bs] = sumi;
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
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 4;
        for (int col = 0; col < 8; col++) {
            const int8_t * vx_col = vx + col * 4;
            int32_t sumi = 0;
            for (int k = 0; k < 4; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
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
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 8;
        for (int col = 0; col < 4; col++) {
            const int8_t * vx_col = vx + col * 8;
            int32_t sumi = 0;
            for (int k = 0; k < 8; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
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
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 16;
        for (int col = 0; col < 2; col++) {
            const int8_t * vx_col = vx + col * 16;
            int32_t sumi = 0;
            for (int k = 0; k < 16; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n2_col16(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {

#if defined(__AVX2__) || defined(__AVX__)
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;

        __m256i qx = _mm256_loadu_si256((const __m256i *)vx);

        uint16_t vy_16;
        memcpy(&vy_16, vy_row, sizeof(uint16_t));
        __m256i qy = _mm256_set1_epi16(vy_16);

        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i dot = _mm256_maddubs_epi16(ax, sy);

        __m128i dot_lo = _mm256_castsi256_si128(dot);
        __m128i dot_hi = _mm256_extracti128_si256(dot, 1);
        __m256i ext_lo = _mm256_cvtepi16_epi32(dot_lo);
        __m256i ext_hi = _mm256_cvtepi16_epi32(dot_hi);

        _mm256_storeu_si256((__m256i *)(s + row * bs), ext_lo);
        _mm256_storeu_si256((__m256i *)(s + row * bs + 8), ext_hi);
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;
        for (int col = 0; col < 16; col++) {
            const int8_t * vx_col = vx + col * 2;
            int32_t sumi = 0;
            for (int k = 0; k < 2; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n4_col2(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {

#if defined(__ARM_NEON)
    int8x8_t vx_vec = vld1_s8(vx);

    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 4;
        int32_t vy_32;
        memcpy(&vy_32, vy_row, sizeof(int32_t));
        int8x8_t vy_vec = vreinterpret_s8_s32(vdup_n_s32(vy_32));

#if defined(__ARM_FEATURE_DOTPROD)
        int32x2_t dot = vdot_s32(vdup_n_s32(0), vx_vec, vy_vec);
        s[row * bs + 0] = vget_lane_s32(dot, 0);
        s[row * bs + 1] = vget_lane_s32(dot, 1);
#else
        int16x8_t prod = vmull_s8(vx_vec, vy_vec);
        int16x4_t lo = vget_low_s16(prod);
        int16x4_t hi = vget_high_s16(prod);
        s[row * bs + 0] = vaddlv_s16(lo);
        s[row * bs + 1] = vaddlv_s16(hi);
#endif
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 4;
        for (int col = 0; col < 2; col++) {
            const int8_t * vx_col = vx + col * 4;
            int32_t sumi = 0;
            for (int k = 0; k < 4; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n2_col4(
    int32_t * s, size_t bs,
    const int8_t * vx, size_t bx,
    const int8_t * vy,
    int nrc) {

#if defined(__ARM_NEON)
    int8x8_t vx_vec = vld1_s8(vx);

    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;
        int16_t vy_16;
        memcpy(&vy_16, vy_row, sizeof(int16_t));
        int8x8_t vy_vec = vreinterpret_s8_s16(vdup_n_s16(vy_16));

        int16x8_t prod = vmull_s8(vx_vec, vy_vec);
        int32x4_t psum = vpaddlq_s16(prod);
        s[row * bs + 0] = vgetq_lane_s32(psum, 0);
        s[row * bs + 1] = vgetq_lane_s32(psum, 1);
        s[row * bs + 2] = vgetq_lane_s32(psum, 2);
        s[row * bs + 3] = vgetq_lane_s32(psum, 3);
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;
        for (int col = 0; col < 4; col++) {
            const int8_t * vx_col = vx + col * 2;
            int32_t sumi = 0;
            for (int k = 0; k < 2; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
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
#elif defined(__ARM_NEON)
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input  = input_data + batch * ne10 * ne11;
        int32_t * output      = dst_data + batch * ne11;

        int8x8_t wv = vld1_s8(weight);

        int64_t col;
        for (col = 0; col + 3 < ne11; col += 4) {
            int8x8_t iv0 = vld1_s8(input + (col + 0) * ne10);
            int8x8_t iv1 = vld1_s8(input + (col + 1) * ne10);
            int8x8_t iv2 = vld1_s8(input + (col + 2) * ne10);
            int8x8_t iv3 = vld1_s8(input + (col + 3) * ne10);
#if defined(__ARM_FEATURE_DOTPROD)
            int32x2_t d0 = vdot_s32(vdup_n_s32(0), iv0, wv);
            int32x2_t d1 = vdot_s32(vdup_n_s32(0), iv1, wv);
            int32x2_t d2 = vdot_s32(vdup_n_s32(0), iv2, wv);
            int32x2_t d3 = vdot_s32(vdup_n_s32(0), iv3, wv);
            output[col + 0] = vget_lane_s32(d0, 0) + vget_lane_s32(d0, 1);
            output[col + 1] = vget_lane_s32(d1, 0) + vget_lane_s32(d1, 1);
            output[col + 2] = vget_lane_s32(d2, 0) + vget_lane_s32(d2, 1);
            output[col + 3] = vget_lane_s32(d3, 0) + vget_lane_s32(d3, 1);
#else
            int16x8_t p0 = vmull_s8(iv0, wv);
            int16x8_t p1 = vmull_s8(iv1, wv);
            int16x8_t p2 = vmull_s8(iv2, wv);
            int16x8_t p3 = vmull_s8(iv3, wv);
            output[col + 0] = vaddlvq_s16(p0);
            output[col + 1] = vaddlvq_s16(p1);
            output[col + 2] = vaddlvq_s16(p2);
            output[col + 3] = vaddlvq_s16(p3);
#endif
        }

        for (; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#else
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input  = input_data + batch * ne10 * ne11;
        int32_t * output      = dst_data + batch * ne11;

        for (int64_t col = 0; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#endif
}
