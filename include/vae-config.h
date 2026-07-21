#define VAE_ACT_PARALLEL
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#if defined(VAE_ACT_PARALLEL)
    #define VAE_ROW_BLOCK_SIZE 4
    #define VAE_COL_BLOCK_SIZE 16
    #define VAE_PARALLEL_SIZE 4
#else
    #define VAE_ROW_BLOCK_SIZE 16
    #define VAE_COL_BLOCK_SIZE 4
    #define VAE_PARALLEL_SIZE 4
#endif
#endif
