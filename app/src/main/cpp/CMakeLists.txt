# Builds libllama_jni.so: llama.cpp linked statically into one shared object.
#
# llama.cpp is fetched at configure time rather than vendored. The pin is a tag, not a
# branch: the public C API has renamed functions twice in the releases around this one
# (`llama_load_model_from_file` -> `llama_model_load_from_file`, and the matching context
# call), so llama_jni.cpp only compiles against a known point. Moving the pin means
# re-reading include/llama.h, not just bumping the number.

cmake_minimum_required(VERSION 3.22.1)
project("llama_jni" CXX)

include(FetchContent)

# Everything below is set before MakeAvailable so it wins over llama.cpp's own option()
# defaults. CMP0077 has been NEW since 3.13 and llama.cpp asks for 3.14, so a plain set()
# would be honoured too -- the cache form is here because it also survives a reconfigure.
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
# llama and ggml build as static archives that then go inside libllama_jni.so, so their
# objects have to be relocatable. The NDK compiles with -fPIC by default and would have let
# this pass unnoticed; a host link of the same CMakeLists is what surfaced it, and being
# explicit costs nothing.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
# The app downloads models itself, over the same HTTP client the rest of it uses. Pulling
# libcurl into the NDK build to duplicate that would be a second network stack in the APK.
set(LLAMA_CURL         OFF CACHE BOOL "" FORCE)

# Both of these are upstream's documented Android settings (docs/android.md). OpenMP ships
# in the NDK but has to be installed as a CMake dependency, which is not supported; and
# llamafile's tinyBLAS does not support Android at all.
set(GGML_OPENMP        OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE     OFF CACHE BOOL "" FORCE)
# We are cross-compiling. -mcpu=native would describe the build machine.
set(GGML_NATIVE        OFF CACHE BOOL "" FORCE)

# The instruction set the quantised kernels are compiled for.
#
# This matters more than it looks. With no -march at all -- which is what ggml does when
# cross-compiling and neither this nor GGML_CPU_ALL_VARIANTS is set -- the NDK default is
# plain armv8-a, so the int8 dot-product kernels that carry Q4/Q8 matmuls compile down to
# scalar NEON and decoding is several times slower than it needs to be.
#
# The default here is the widest baseline that still gets those kernels: dotprod and fp16
# arithmetic are armv8.2, which every arm64 device that shipped with API 31 has. i8mm is
# faster again on this phone's Dimensity 9500s, but it is armv8.6 and would SIGILL on the
# older devices this APK also installs on, so it is opt-in:
#
#     ./gradlew :app:assembleRelease -PllamaArmArch=armv9.2-a+i8mm+dotprod+fp16+sve2
#
# ggml has runtime feature detection, but it only chooses between kernels the compiler was
# allowed to emit -- it cannot recover an instruction that -march excluded.
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)

FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        b6100
    GIT_SHALLOW    TRUE
    # llama.cpp has no submodules; asking for none skips a pointless network round trip.
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(llama)

add_library(llama_jni SHARED llama_jni.cpp)
target_compile_features(llama_jni PRIVATE cxx_std_17)

target_link_libraries(llama_jni PRIVATE llama)

# liblog is Android's, and ggml calls into it for its own diagnostics. Guarded because
# src/test/native builds this same file for the host, where there is no such library --
# that host build is what runs the JNI layer against a real model, so keeping it possible
# is worth one conditional.
if (ANDROID)
    target_link_libraries(llama_jni PRIVATE log)
endif()
