cmake_minimum_required(VERSION 3.13)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)


get_filename_component(TFINF_DIR "." REALPATH)
set(tfpath ${TFINF_DIR}/../Deps/tf)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
  include_directories(${tfpath}/include_x86_64)
  find_library(TENSORFLOW_LIB
               NAMES libtensorflow_cc.so.2
               PATHS ${tfpath}/lib/x86_64
               NO_DEFAULT_PATH)

  find_library(TENSORFLOW_FW_LIB
               NAMES libtensorflow_framework.so.2
               PATHS ${tfpath}/lib/x86_64
               NO_DEFAULT_PATH)
else()
  include_directories(${tfpath}/include_aarch64)
  find_library(TENSORFLOW_LIB
               NAMES libtensorflow_cc.so.2
               PATHS ${tfpath}/lib/aarch64
               NO_DEFAULT_PATH)

  find_library(TENSORFLOW_FW_LIB
               NAMES libtensorflow_framework.so.2
               PATHS ${tfpath}/lib/aarch64
               NO_DEFAULT_PATH)
endif()


if(NOT TENSORFLOW_LIB OR NOT TENSORFLOW_FW_LIB)
    message(WARNING "tensorflow library not found. Please install it if using TfInfer")
    return()
endif()


file(GLOB local_src
    "*.c"
    "*.cpp"
	)

set(TFINFER_DYNAMIC_LIB "tfinfer")
set(TFINFER_STATIC_LIB "tfinfer_static")

add_library( ${TFINFER_STATIC_LIB} STATIC TfInfer.cpp)
target_link_libraries( ${TFINFER_STATIC_LIB} ${TENSORFLOW_LIB} ${TENSORFLOW_FW_LIB} mx_accl)

add_library(${TFINFER_DYNAMIC_LIB} SHARED  TfInfer.cpp)
target_link_libraries(${TFINFER_DYNAMIC_LIB} ${TENSORFLOW_LIB} ${TENSORFLOW_FW_LIB} mx_accl)

list(APPEND ALL_STATIC_UTILS ${TFINFER_STATIC_LIB})
set(ALL_STATIC_UTILS ${ALL_STATIC_UTILS} PARENT_SCOPE)
