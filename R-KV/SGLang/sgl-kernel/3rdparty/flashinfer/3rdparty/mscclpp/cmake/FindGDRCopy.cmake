# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Find the GDRCopy libraries
#
# The following variables are optionally searched for defaults
#  GDRCOPY_ROOT_DIR: Base directory where all GDRCopy components are found
#  GDRCOPY_INCLUDE_DIR: Directory where GDRCopy headers are found
#  GDRCOPY_LIB_DIR: Directory where GDRCopy libraries are found

# The following are set after configuration is done:
#  GDRCOPY_FOUND
#  GDRCOPY_INCLUDE_DIRS
#  GDRCOPY_LIBRARIES

# An imported target MSCCLPP::gdrcopy is created if the library is found.

find_path(GDRCOPY_INCLUDE_DIRS
  NAMES gdrapi.h
  HINTS
  ${GDRCOPY_INCLUDE_DIR}
  ${GDRCOPY_ROOT_DIR}
  ${GDRCOPY_ROOT_DIR}/include)

find_library(GDRCOPY_LIBRARIES
  NAMES gdrapi
  HINTS
  ${GDRCOPY_LIB_DIR}
  ${GDRCOPY_ROOT_DIR}
  ${GDRCOPY_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GDRCopy DEFAULT_MSG GDRCOPY_INCLUDE_DIRS GDRCOPY_LIBRARIES)
mark_as_advanced(GDRCOPY_INCLUDE_DIR GDRCOPY_LIBRARIES)

if(GDRCOPY_FOUND)
  if(NOT TARGET MSCCLPP::gdrcopy)
    add_library(MSCCLPP::gdrcopy UNKNOWN IMPORTED)
  endif()
  set_target_properties(MSCCLPP::gdrcopy PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GDRCOPY_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${GDRCOPY_LIBRARIES}")
endif()