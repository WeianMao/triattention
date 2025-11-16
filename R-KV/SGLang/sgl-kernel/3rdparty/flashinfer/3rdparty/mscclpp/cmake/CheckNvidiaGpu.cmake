# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(NVIDIA_FOUND "FALSE")

find_package(CUDAToolkit)

if(NOT CUDAToolkit_FOUND)
    return()
endif()

set(CMAKE_CUDA_ARCHITECTURES "60")
if(NOT CMAKE_CUDA_COMPILER)
    # In case the CUDA Toolkit directory is not in the PATH
    find_program(CUDA_COMPILER
                 NAMES nvcc
                 PATHS ${CUDAToolkit_BIN_DIR})
    if(NOT CUDA_COMPILER)
        message(WARNING "Could not find nvcc in ${CUDAToolkit_BIN_DIR}")
        unset(CMAKE_CUDA_ARCHITECTURES)
        return()
    endif()
    set(CMAKE_CUDA_COMPILER "${CUDA_COMPILER}")
endif()
enable_language(CUDA)

set(CHECK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_nvidia_gpu.cu")

try_run(RUN_RESULT COMPILE_SUCCESS SOURCES ${CHECK_SRC})

if(COMPILE_SUCCESS AND RUN_RESULT EQUAL 0)
    set(NVIDIA_FOUND "TRUE")
else()
    unset(CMAKE_CUDA_ARCHITECTURES)
    unset(CMAKE_CUDA_COMPILER)
endif()
