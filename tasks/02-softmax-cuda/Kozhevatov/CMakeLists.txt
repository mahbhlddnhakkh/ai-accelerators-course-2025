cmake_minimum_required(VERSION 3.18)
project(softmax_cuda LANGUAGES CXX CUDA)

enable_language(CUDA)

set(target_suffix Kozhevatov)
set(target_name "softmax_cuda_${target_suffix}")

add_executable(${target_name} main.cu)

target_compile_features(${target_name} PRIVATE cxx_std_17)

set_target_properties(${target_name}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

# Оптимизации для релизной сборки
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(${target_name} 
        PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target_name} PRIVATE CUDA::cudart)

message(STATUS "Building target: ${target_name}")
