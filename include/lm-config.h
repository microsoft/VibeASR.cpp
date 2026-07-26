#define ACT_PARALLEL
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#if defined(ACT_PARALLEL)
    #define ROW_BLOCK_SIZE 4
    #define COL_BLOCK_SIZE 32
    #define PARALLEL_SIZE 4
#else
    #define ROW_BLOCK_SIZE 32
    #define COL_BLOCK_SIZE 4
    #define PARALLEL_SIZE 4
#endif
#elif defined(__ARM_NEON)
#if defined(ACT_PARALLEL)
    #define ROW_BLOCK_SIZE 4
    #define COL_BLOCK_SIZE 32
    #define PARALLEL_SIZE 4
#else
    #define ROW_BLOCK_SIZE 32
    #define COL_BLOCK_SIZE 4
    #define PARALLEL_SIZE 4
#endif
#endif

