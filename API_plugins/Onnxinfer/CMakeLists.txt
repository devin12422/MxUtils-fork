cmake_minimum_required(VERSION 3.13)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)


get_filename_component(ONNXINF_DIR "." REALPATH)
include_directories(${ONNXINF_DIR}/../Deps/ort/include)

file(GLOB local_src
    "*.c"
    "*.cpp"
	)

set(ONNXINFER_DYNAMIC_LIB "onnxinfer")
set(ONNXINFER_STATIC_LIB "onnxinfer_static")


add_library(${ONNXINFER_STATIC_LIB} STATIC OnnxInfer.cpp)
add_library(${ONNXINFER_DYNAMIC_LIB} SHARED  OnnxInfer.cpp)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
  target_link_libraries(${ONNXINFER_DYNAMIC_LIB} ${ONNXINF_DIR}/../Deps/ort/lib/x86_64/libonnxruntime.so.1.18.1 mx_accl)
  target_link_libraries(${ONNXINFER_STATIC_LIB}  ${ONNXINF_DIR}/../Deps/ort/lib/x86_64/libonnxruntime.so.1.18.1 mx_accl)
else()
  target_link_libraries(${ONNXINFER_DYNAMIC_LIB} ${ONNXINF_DIR}/../Deps/ort/lib/aarch64/libonnxruntime.so.1.18.1 mx_accl)
  target_link_libraries(${ONNXINFER_STATIC_LIB}  ${ONNXINF_DIR}/../Deps/ort/lib/aarch64/libonnxruntime.so.1.18.1 mx_accl)
endif()

list(APPEND ALL_STATIC_UTILS ${ONNXINFER_STATIC_LIB})
set(ALL_STATIC_UTILS ${ALL_STATIC_UTILS} PARENT_SCOPE)
