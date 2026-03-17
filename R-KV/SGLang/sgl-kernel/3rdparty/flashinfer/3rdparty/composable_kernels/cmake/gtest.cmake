include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

FetchContent_Declare(
    GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571
)

# Suppress ROCMChecks WARNING on GoogleTests
set(ROCM_DISABLE_CHECKS FALSE)
macro(rocm_check_toolchain_var var access value list_file)
    if(NOT ROCM_DISABLE_CHECKS)
        _rocm_check_toolchain_var("${var}" "${access}" "${value}" "${list_file}")
    endif()
endmacro()

if(WIN32)
    set(gtest_force_shared_crt ON CACHE_INTERNAL "")
endif()

set(BUILD_GMOCK OFF CACHE INTERNAL "")
set(INSTALL_GTEST OFF CACHE INTERNAL "")

# Store the current value of BUILD_SHARED_LIBS
set(__build_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

set(ROCM_DISABLE_CHECKS TRUE)
FetchContent_MakeAvailable(GTest)
set(ROCM_DISABLE_CHECKS FALSE)

# Restore the old value of BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "Type of libraries to build" FORCE)

set(BUILD_GMOCK OFF CACHE INTERNAL "")
set(INSTALL_GTEST OFF CACHE INTERNAL "")

set(GTEST_CXX_FLAGS
     -Wno-undef
     -Wno-reserved-identifier
     -Wno-global-constructors
     -Wno-missing-noreturn
     -Wno-disabled-macro-expansion
     -Wno-used-but-marked-unused
     -Wno-switch-enum
     -Wno-zero-as-null-pointer-constant
     -Wno-unused-member-function
     -Wno-comma
     -Wno-old-style-cast
     -Wno-deprecated
     -Wno-unsafe-buffer-usage
     -Wno-float-equal
)

if(WIN32)
    list(APPEND GTEST_CXX_FLAGS
            -Wno-suggest-destructor-override
            -Wno-suggest-override
            -Wno-nonportable-system-include-path
            -Wno-language-extension-token)
endif()

target_compile_options(gtest PRIVATE ${GTEST_CXX_FLAGS})
target_compile_options(gtest_main PRIVATE ${GTEST_CXX_FLAGS})
