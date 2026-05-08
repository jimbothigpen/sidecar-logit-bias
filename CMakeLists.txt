cmake_minimum_required(VERSION 3.18)
project(sidecar_logit_bias CXX)

#
# Standalone out-of-tree sidecar handler plugin: token-level additive logit
# bias. Builds against an installed frankenturbo2 engine; produces
# libsidecar_logit_bias.so loadable via:
#
#     llama-cli --sidecar-load-plugin /path/to/libsidecar_logit_bias.so \
#               --sidecar-vectors /path/to/your.bias.gguf
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Hidden-by-default visibility matches the engine; the
# LLAMA_SIDECAR_PLUGIN_INIT_DECL macro in <llama-sidecar-plugin.h> re-marks
# the plugin entry point as default-visible.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(LLAMA_INSTALL_PREFIX "/opt/llama-frankenturbo2-vulkan"
    CACHE PATH "frankenturbo2 install prefix (contains include/ and lib/)")

add_library(sidecar_logit_bias SHARED logit_bias.cpp)

target_include_directories(sidecar_logit_bias PRIVATE
    "${LLAMA_INSTALL_PREFIX}/include")

find_library(LLAMA_LIB      llama      PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB       ggml       PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB  ggml-base  PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

target_link_libraries(sidecar_logit_bias PRIVATE
    ${LLAMA_LIB}
    ${GGML_LIB}
    ${GGML_BASE_LIB})
