#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "onnxruntime::onnxruntime" for configuration "Release"
set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libonnxruntime.so.1.21.0"
  IMPORTED_SONAME_RELEASE "libonnxruntime.so.1"
  )

list(APPEND _cmake_import_check_targets onnxruntime::onnxruntime )
list(APPEND _cmake_import_check_files_for_onnxruntime::onnxruntime "${_IMPORT_PREFIX}/lib64/libonnxruntime.so.1.21.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
