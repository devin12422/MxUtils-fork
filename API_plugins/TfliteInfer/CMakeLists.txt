cmake_minimum_required(VERSION 3.13)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)


get_filename_component(TFLINF_DIR "." REALPATH)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
  set(tflpath ${TFLINF_DIR}/../Deps/tflite/x86_64)
else()
  set(tflpath ${TFLINF_DIR}/../Deps/tflite/aarch64)
endif()


find_library(TFLITE NAMES tensorflow-lite  HINTS ${tflpath}/lib)

if(NOT TFLITE)
    message(WARNING "tensorflow-lite library not found. Please install it if using TfliteInfer")
    return()
endif()

include_directories(${tflpath}/include)

file(GLOB local_src
    "*.c"
    "*.cpp"
	)

set(TFLITEINFER_DYNAMIC_LIB "tfliteinfer")
set(TFLITEINFER_STATIC_LIB "tfliteinfer_static")

add_library(${TFLITEINFER_DYNAMIC_LIB} SHARED  ${local_src})
target_link_libraries(${TFLITEINFER_DYNAMIC_LIB} ${TFLITE} mx_accl)

add_library(${TFLITEINFER_STATIC_LIB} STATIC ${local_src})
target_link_libraries(${TFLITEINFER_STATIC_LIB} ${TFLITE} mx_accl)

list(APPEND ALL_STATIC_UTILS ${TFLITEINFER_STATIC_LIB})
set(ALL_STATIC_UTILS ${ALL_STATIC_UTILS} PARENT_SCOPE)
