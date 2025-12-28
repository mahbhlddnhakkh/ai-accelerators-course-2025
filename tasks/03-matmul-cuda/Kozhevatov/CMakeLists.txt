cmake_minimum_required(VERSION 3.18)
project(softmax_cpu_Kozhevatov)

enable_language(CUDA)

set(target_suffix Kozhevatov)
set(target_name "softmax_cpu_${target_suffix}")

add_executable(${target_name} main.cu)

target_compile_features(${target_name} PRIVATE cxx_std_17)

set_target_properties(${target_name}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "75"
)

# Добавляем флаги CUDA
target_compile_options(${target_name} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)

find_package(OpenMP REQUIRED)
target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target_name} PRIVATE CUDA::cudart)

# CUTLASS
if(CUTLASS_ROOT)
    message(STATUS "Using user-provided CUTLASS_ROOT: ${CUTLASS_ROOT}")
    if(EXISTS "${CUTLASS_ROOT}/include/cutlass/gemm/device/gemm.h")
        target_include_directories(${target_name} PRIVATE "${CUTLASS_ROOT}/include")
        message(STATUS "CUTLASS found in provided path")
    else()
        message(WARNING "CUTLASS not found in ${CUTLASS_ROOT}/include. Will search in standard locations.")
        set(CUTLASS_ROOT "")  # Сбрасываем, чтобы искать в стандартных путях
    endif()
endif()

# Если CUTLASS_ROOT не задан или не найден в указанном пути, ищем в стандартных
if(NOT CUTLASS_ROOT)
    # Список стандартных путей для поиска CUTLASS
    set(CUTLASS_SEARCH_PATHS
        "/opt/cutlass"
        "/usr/local/cutlass" 
        "/usr/include/cutlass"
        "${CMAKE_SOURCE_DIR}/cutlass"
        "${CMAKE_SOURCE_DIR}/vendor/cutlass"
        "${CMAKE_SOURCE_DIR}/../cutlass"
    )
    
    # Ищем заголовочный файл CUTLASS
    find_path(CUTLASS_INCLUDE_DIR
        NAMES cutlass/gemm/device/gemm.h
        PATHS ${CUTLASS_SEARCH_PATHS}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH  # Не искать в системных путях по умолчанию
    )
    
    if(CUTLASS_INCLUDE_DIR)
        target_include_directories(${target_name} PRIVATE "${CUTLASS_INCLUDE_DIR}")
        message(STATUS "Found CUTLASS: ${CUTLASS_INCLUDE_DIR}")
    else()
        # Последняя попытка: системный поиск
        find_path(CUTLASS_INCLUDE_DIR cutlass/gemm/device/gemm.h)
        if(CUTLASS_INCLUDE_DIR)
            target_include_directories(${target_name} PRIVATE "${CUTLASS_INCLUDE_DIR}")
            message(STATUS "Found CUTLASS in system: ${CUTLASS_INCLUDE_DIR}")
        else()
            message(WARNING "CUTLASS not found. CUTLASS functionality will be disabled.")
            # Добавляем определение для условной компиляции
            target_compile_definitions(${target_name} PRIVATE -DNO_CUTLASS)
        endif()
    endif()
endif()
