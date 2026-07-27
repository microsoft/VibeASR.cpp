// Differential test for the I2_S (BitNet ternary) LM kernels.
//
// The GGUF file layout is produced by quantize_i2_s() in src/ggml-lm-mad.cpp,
// which is compiled ONLY for x86 and uses QK_I2_S = 128:
//
//   element e = blk*128 + j,  group = j/32,  pos = j%32
//   byte[blk*32 + pos], bits (6-2*group) .. (7-2*group)
//
// This test packs weights in exactly that layout and checks whether the kernels
// on this machine decode them the same way.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>

extern "C" {
void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_gemv_i2_i8_s(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_i2_i8_s(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
}
// C++ linkage (not extern "C" in ggml-lm-mad.cpp)
void ggml_vec_dot_i2_i8_s_1x4_32W(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static std::mt19937 rng(7);

// pack one row of n ternary codes (values 0/1/2) using the x86 file layout
static void pack_row_x86(const uint8_t * q, int n, uint8_t * out) {
    memset(out, 0, n / 4);
    for (int blk = 0; blk < n / 128; blk++)
        for (int j = 0; j < 128; j++) {
            int g = j / 32, pos = j % 32;
            out[blk * 32 + pos] |= (uint8_t)(q[blk * 128 + j] << (6 - 2 * g));
        }
}

// what the ARM kernel currently assumes: 64-element blocks, 16-byte units
static void pack_row_arm64blk(const uint8_t * q, int n, uint8_t * out) {
    memset(out, 0, n / 4);
    for (int blk = 0; blk < n / 64; blk++)
        for (int j = 0; j < 64; j++) {
            int g = j / 16, pos = j % 16;
            out[blk * 16 + pos] |= (uint8_t)(q[blk * 64 + j] << (6 - 2 * g));
        }
}

static int32_t ref_dot(const uint8_t * q, const int8_t * y, int n) {
    int32_t s = 0;
    for (int k = 0; k < n; k++) s += (int32_t)q[k] * (int32_t)y[k];
    return s;
}

int main() {
    const int n = 1536;                 // qwen2 n_embd; also 8960 for the FFN
    const int nc = 8;                   // output columns
    printf("n = %d, nc = %d\n\n", n, nc);

    std::vector<uint8_t> q((size_t)nc * n);
    for (auto & v : q) v = (uint8_t)(rng() % 3);        // ternary codes 0,1,2
    std::vector<int8_t> y(n);
    for (auto & v : y) v = (int8_t)((int)(rng() % 255) - 127);

    std::vector<uint8_t> packed_x86((size_t)nc * n / 4 + 64);
    std::vector<uint8_t> packed_arm((size_t)nc * n / 4 + 64);
    for (int c = 0; c < nc; c++) {
        pack_row_x86(&q[(size_t)c * n], n, &packed_x86[(size_t)c * n / 4]);
        pack_row_arm64blk(&q[(size_t)c * n], n, &packed_arm[(size_t)c * n / 4]);
    }

    std::vector<int32_t> ref(nc);
    for (int c = 0; c < nc; c++) ref[c] = ref_dot(&q[(size_t)c * n], y.data(), n);

    std::vector<float> got_x86(nc, -1e30f), got_arm(nc, -1e30f);
    ggml_gemv_i2_i8_s(n, got_x86.data(), 1, packed_x86.data(), y.data(), 1, nc);
    ggml_gemv_i2_i8_s(n, got_arm.data(), 1, packed_arm.data(), y.data(), 1, nc);

    int bad_x86 = 0, bad_arm = 0;
    printf("%-4s %14s %20s %20s\n", "col", "reference", "kernel(x86 layout)", "kernel(64-blk layout)");
    for (int c = 0; c < nc; c++) {
        printf("%-4d %14d %20.0f %20.0f\n", c, ref[c], got_x86[c], got_arm[c]);
        if ((int32_t)got_x86[c] != ref[c]) bad_x86++;
        if ((int32_t)got_arm[c] != ref[c]) bad_arm++;
    }
    printf("\nfed the REAL GGUF layout (128-element blocks): %d / %d columns WRONG\n", bad_x86, nc);
    printf("fed a 64-element-block layout instead:        %d / %d columns wrong\n", bad_arm, nc);
    // ---- gemm path (batched prefill) : s[r*bs + c] = dot(col c, row r of y)
    {
        const int nr = 4;
        std::vector<int8_t> ym((size_t)nr * n);
        for (auto & v : ym) v = (int8_t)((int)(rng() % 255) - 127);
        std::vector<float> g((size_t)nr * nc, -1e30f);
        ggml_gemm_i2_i8_s(n, g.data(), (size_t)nc, packed_x86.data(), ym.data(), nr, nc);
        int bad = 0;
        printf("\n-- ggml_gemm_i2_i8_s (nr=%d) --\n", nr);
        for (int r = 0; r < nr; r++)
            for (int c = 0; c < nc; c++) {
                int32_t want = ref_dot(&q[(size_t)c * n], &ym[(size_t)r * n], n);
                if ((int32_t)g[(size_t)r * nc + c] != want) {
                    if (bad < 6) printf("   r=%d c=%d  got %10.0f  ref %10d\n", r, c, g[(size_t)r*nc+c], want);
                    bad++;
                }
            }
        printf("   %d / %d entries wrong\n", bad, nr * nc);
        bad_x86 += bad;
    }

    // ---- _1x4_32W : row semantics, s[row] = dot(row, y); currently unreachable
    {
        std::vector<float> g4(nc, -1e30f);
        ggml_vec_dot_i2_i8_s_1x4_32W(n, g4.data(), 1, packed_x86.data(), (size_t)n, y.data(), 0, nc);
        int bad = 0;
        for (int c = 0; c < nc; c++) if ((int32_t)g4[c] != ref[c]) bad++;
        printf("\n-- ggml_vec_dot_i2_i8_s_1x4_32W --\n   %d / %d wrong\n", bad, nc);
        bad_x86 += bad;
    }

    printf("\n%s\n", bad_x86 == 0
        ? "kernel agrees with the shipped file layout."
        : "*** kernel does NOT decode the shipped file layout ***");
    return bad_x86 ? 1 : 0;
}
