enable_language(CUDA CXX)

set(target_suffix chuvashev)
set(target_name "softmax_cuda_${target_suffix}")

add_executable(${target_name} 
    src/kernel.cu 
    src/main.cpp
    src/utils.cpp
    src/simd_utils.cpp
    include/utils.h
    include/simd_utils.h
)

source_group("Source Files" FILES 
    src/kernel.cu 
    src/main.cpp
    src/utils.cpp
    src/simd_utils.cpp
)
source_group("Header Files" FILES 
    include/utils.h
    include/simd_utils.h
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(${target_name} PRIVATE -mavx2)
else()
    message(WARNING "AVX2 not supported, performance may be degraded")
endif()

target_include_directories(${target_name} PRIVATE include/)

find_package(OpenMP REQUIRED)
target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target_name} PRIVATE CUDA::cudart)

target_compile_features(${target_name} PRIVATE cxx_std_17)

set_target_properties(${target_name}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)