// Differential test: I8_S kernels vs unambiguous scalar reference.
// Build: see tests/build_difftest.sh

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>

extern "C" {
void ggml_vec_dot_i8_i8(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_gemv_i8_i8(int n, int32_t * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_i8_i8(int n, int32_t * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_vec_dot_i8_i8_batch_n8(int32_t * dst, const int8_t * w, const int8_t * in,
                                 int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne10, int64_t ne11);
}

static std::mt19937 rng(1234);
static int8_t r8() { return (int8_t)((int)(rng() % 255) - 127); }

static int fail_count = 0;

static void report(const char * what, int n, int nr, int nc,
                   const std::vector<int32_t> & got,
                   const std::vector<int32_t> & ref) {
    int bad = 0;
    int first = -1;
    for (size_t i = 0; i < ref.size(); i++) {
        if (got[i] != ref[i]) { if (first < 0) first = (int)i; bad++; }
    }
    if (bad == 0) {
        printf("  PASS %-28s n=%-6d nr=%-5d nc=%-5d\n", what, n, nr, nc);
        return;
    }
    fail_count++;
    printf("  FAIL %-28s n=%-6d nr=%-5d nc=%-5d  %d/%zu mismatched, first at idx %d: got %d ref %d\n",
           what, n, nr, nc, bad, ref.size(), first, got[first], ref[first]);
    // print a few
    int shown = 0;
    for (size_t i = 0; i < ref.size() && shown < 6; i++) {
        if (got[i] != ref[i]) { printf("       idx %4zu: got %8d  ref %8d  diff %8d\n", i, got[i], ref[i], got[i]-ref[i]); shown++; }
    }
}

// ---- ggml_gemm_i8_i8: s[r*bs + c] = sum_k vx[c*n+k] * vy[r*n+k]
static void test_gemm(int n, int nr, int nc) {
    std::vector<int8_t> x((size_t)nc*n), y((size_t)nr*n);
    for (auto & v : x) v = r8();
    for (auto & v : y) v = r8();
    std::vector<int32_t> got((size_t)nr*nc, 0x7ec0ffee), ref((size_t)nr*nc, 0);
    for (int r = 0; r < nr; r++)
        for (int c = 0; c < nc; c++) {
            int32_t s = 0;
            for (int k = 0; k < n; k++) s += (int32_t)x[(size_t)c*n+k] * (int32_t)y[(size_t)r*n+k];
            ref[(size_t)r*nc + c] = s;
        }
    ggml_gemm_i8_i8(n, got.data(), (size_t)nc, x.data(), y.data(), nr, nc);
    report("ggml_gemm_i8_i8", n, nr, nc, got, ref);
}

// ---- ggml_gemv_i8_i8: s[c] = sum_k vx[c*n+k] * vy[k]
static void test_gemv(int n, int nc) {
    std::vector<int8_t> x((size_t)nc*n), y((size_t)n);
    for (auto & v : x) v = r8();
    for (auto & v : y) v = r8();
    std::vector<int32_t> got((size_t)nc, 0x7ec0ffee), ref((size_t)nc, 0);
    for (int c = 0; c < nc; c++) {
        int32_t s = 0;
        for (int k = 0; k < n; k++) s += (int32_t)x[(size_t)c*n+k] * (int32_t)y[k];
        ref[c] = s;
    }
    ggml_gemv_i8_i8(n, got.data(), (size_t)nc, x.data(), y.data(), 1, nc);
    report("ggml_gemv_i8_i8", n, 1, nc, got, ref);
}

// ---- ggml_vec_dot_i8_i8 in ACT_PARALLEL (column) semantics:
//      s[col*bs] = sum_k x[k] * y[col*by + k]
static void test_vecdot_col(int n, int nrc) {
    std::vector<int8_t> x((size_t)n), y((size_t)nrc*n);
    for (auto & v : x) v = r8();
    for (auto & v : y) v = r8();
    const size_t bs = 3;   // non-trivial stride, like the real caller (bs = OC)
    std::vector<int32_t> got((size_t)nrc*bs, 0x7ec0ffee), ref((size_t)nrc*bs, 0x7ec0ffee);
    for (int c = 0; c < nrc; c++) {
        int32_t s = 0;
        for (int k = 0; k < n; k++) s += (int32_t)x[k] * (int32_t)y[(size_t)c*n+k];
        ref[(size_t)c*bs] = s;
    }
    ggml_vec_dot_i8_i8(n, got.data(), bs, x.data(), n, y.data(), n, nrc);
    report("ggml_vec_dot_i8_i8(col)", n, 1, nrc, got, ref);
}

// ---- batch_n8: out[b*ne11 + col] = sum_k w[b*ne00+k] * in[b*ne10*ne11 + col*ne10 + k]
static void test_batch_n8(int ne00, int ne02, int ne11) {
    const int ne10 = ne00;
    std::vector<int8_t> w((size_t)ne02*ne00), in((size_t)ne02*ne10*ne11);
    for (auto & v : w) v = r8();
    for (auto & v : in) v = r8();
    std::vector<int32_t> got((size_t)ne02*ne11, 0x7ec0ffee), ref((size_t)ne02*ne11, 0);
    for (int b = 0; b < ne02; b++)
        for (int c = 0; c < ne11; c++) {
            int32_t s = 0;
            for (int k = 0; k < ne00; k++)
                s += (int32_t)w[(size_t)b*ne00+k] * (int32_t)in[(size_t)b*ne10*ne11 + (size_t)c*ne10 + k];
            ref[(size_t)b*ne11 + c] = s;
        }
    ggml_vec_dot_i8_i8_batch_n8(got.data(), w.data(), in.data(), ne00, 1, ne02, ne10, ne11);
    report("batch_n8(dw conv)", ne00, ne02, ne11, got, ref);
}

int main() {
    printf("== ggml_vec_dot_i8_i8 (column semantics, VAE_ACT_PARALLEL) ==\n");
    // NOTE: only nrc == 1 or a multiple of VAE_PARALLEL_SIZE (4) is tested here.
    // For any other nrc the dispatcher falls into ggml_vec_dot_i8_i8_1x1, which
    // uses ROW semantics instead of COLUMN semantics. No live caller does that,
    // but it is a latent landmine -- see the notes.
    for (int n : {8, 16, 32, 40, 64, 128, 256, 1024, 1536, 4096, 8192, 16384})
        for (int nrc : {1, 4, 8, 16})
            test_vecdot_col(n, nrc);

    printf("\n== ggml_gemv_i8_i8 ==\n");
    for (int n : {8, 32, 64, 128, 256, 1024, 1536, 4096, 8192, 16384})
        test_gemv(n, 32);

    printf("\n== ggml_gemm_i8_i8 (real VAE shapes) ==\n");
    // n = KW*IC per downsample layer, nc = OC
    struct { int n, nc; const char * tag; } shapes[] = {
        {    8,   32, "downsample0 8x1 ->32"   },
        {  128,   64, "downsample1 4x32 ->64"  },
        {  256,  128, "downsample2 4x64 ->128" },
        { 1024,  256, "downsample3 8x128->256" },
        { 4096,  512, "downsample4 16x256->512"},
        { 8192, 1024, "downsample5 16x512->1024"},
        {16384, 2048, "downsample6 16x1024->2048"},
        {16384,   64, "acoustic head 8x2048->64"},
        {   64, 1536, "connector fc1 64->1536" },
        { 1536, 1536, "connector fc2 1536->1536"},
    };
    for (auto & s : shapes) {
        printf("  [%s]\n", s.tag);
        for (int nr : {1, 2, 3, 4, 5, 7, 8, 16, 17, 64})
            test_gemm(s.n, nr, s.nc);
    }

    printf("\n== batch_n8 (depthwise, ne00=8) ==\n");
    for (int C : {32, 64, 2048})
        for (int cols : {1, 3, 4, 5, 16, 17, 64})
            test_batch_n8(8, C, cols);

    printf("\n%s  (%d failing cases)\n", fail_count ? "*** FAILURES ***" : "ALL PASS", fail_count);
    return fail_count ? 1 : 0;
}
