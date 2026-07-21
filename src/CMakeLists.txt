set(GGML_HEADERS_VAE ../include/ggml-vae-i8_s-mad.h)
set(GGML_SOURCES_LM ggml-lm-mad.cpp)
set(GGML_SOURCES_VAE ggml-vae-i8_s-mad.cpp)

include_directories(3rdparty/llama.cpp/ggml/include)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for VibeASR.cpp compilation")
endif()
