# Find the JPEGTurbo libraries
#
# The following variables are optionally searched for defaults
#  JPEGTurbo_ROOT_DIR:    Base directory where all JPEGTurbo components are found
#
# The following are set after configuration is done:
#  JPEGTurbo_FOUND
#  JPEGTurbo_INCLUDE_DIR
#  JPEGTurbo_LIBRARIES

find_path(JPEGTurbo_INCLUDE_DIR NAMES turbojpeg.h
                             PATHS /usr/include ${JPEGTurbo_ROOT_DIR} ${JPEGTurbo_ROOT_DIR}/include)

find_library(JPEGTurbo_LIBRARIES SHARED NAMES libturbojpeg.so.0
                              PATHS  ${JPEGTurbo_ROOT_DIR} ${JPEGTurbo_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JPEGTurbo DEFAULT_MSG JPEGTurbo_INCLUDE_DIR JPEGTurbo_LIBRARIES)

if(JPEGTurbo_FOUND)
  message(STATUS "Found JPEGTurbo  (include: ${JPEGTurbo_INCLUDE_DIR}, library: ${JPEGTurbo_LIBRARIES})")
  mark_as_advanced(JPEGTurbo_INCLUDE_DIR JPEGTurbo_LIBRARIES)
endif()

