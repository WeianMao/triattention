# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Find the numa libraries
#
# The following variables are optionally searched for defaults
#  NUMA_ROOT_DIR: Base directory where all numa components are found
#  NUMA_INCLUDE_DIR: Directory where numa headers are found
#  NUMA_LIB_DIR: Directory where numa libraries are found

# The following are set after configuration is done:
#  NUMA_FOUND
#  NUMA_INCLUDE_DIRS
#  NUMA_LIBRARIES

# An imported target MSCCLPP::numa is created if the library is found.

find_path(NUMA_INCLUDE_DIRS
  NAMES numa.h
  HINTS
  ${NUMA_INCLUDE_DIR}
  ${NUMA_ROOT_DIR}
  ${NUMA_ROOT_DIR}/include)

find_library(NUMA_LIBRARIES
  NAMES numa
  HINTS
  ${NUMA_LIB_DIR}
  ${NUMA_ROOT_DIR}
  ${NUMA_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_INCLUDE_DIRS NUMA_LIBRARIES)
mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARIES)

if(NUMA_FOUND)
  if(NOT TARGET MSCCLPP::numa)
    add_library(MSCCLPP::numa UNKNOWN IMPORTED)
  endif()
  set_target_properties(MSCCLPP::numa PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${NUMA_LIBRARIES}")
endif()