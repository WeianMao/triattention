# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(AMD_FOUND "FALSE")

set(CMAKE_PREFIX_PATH "/opt/rocm;${CMAKE_PREFIX_PATH}")
# Temporal fix for rocm5.6
set(ENV{amd_comgr_DIR} "/opt/rocm/lib/cmake/amd_comgr")
set(ENV{AMDDeviceLibs_DIR} "/opt/rocm/lib/cmake/AMDDeviceLibs")

find_package(hip QUIET)

if(NOT hip_FOUND)
    return()
endif()

enable_language(HIP)

set(CHECK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_amd_gpu.hip")

try_run(RUN_RESULT COMPILE_SUCCESS SOURCES ${CHECK_SRC})

if(COMPILE_SUCCESS AND RUN_RESULT EQUAL 0)
    set(AMD_FOUND "TRUE")
endif()
