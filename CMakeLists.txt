cmake_minimum_required(VERSION 3.13)

project(MX_UTILS)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_X86_FLAGS_BASE "-mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mfxsr -mcx16 -msahf -mpclmul")
#set(CMAKE_X86_FLAGS_AVX2 "-mavx -mavx2 -mfma -mbmi -mbmi2 -maes -mf16c -mfsgsbase -mlzcnt -mmovbe -mxsave")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
  set(
      CMAKE_C_FLAGS
      "${CMAKE_C_FLAGS} -Wall -Wextra -pipe -fPIC -O3 ${CMAKE_X86_FLAGS_BASE} -mtune=generic"
    )
  set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 ${CMAKE_X86_FLAGS_BASE} -mtune=generic"
  )
else()
  set(
      CMAKE_C_FLAGS
      "${CMAKE_C_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=armv8-a+simd"
    )
  set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=armv8-a+simd"
  )
endif()



if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_definitions(OS_LINUX)
endif()

set(ALL_STATIC_UTILS)

message(STATUS "MX_API_HOME_DIR  set to " $ENV{MX_API_HOME})
set(MX_API_DIR $ENV{MX_API_HOME}/mx_accl)
message(STATUS "MX_API_DIR  set to " ${MX_API_DIR})

#define SUBDIR macro
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

add_subdirectory(API_plugins)
add_subdirectory(mxutils_gui)

set(ALL_STATIC_FILE_PATHS)
FOREACH(util ${ALL_STATIC_UTILS})
  list(APPEND ALL_STATIC_FILE_PATHS $<TARGET_FILE:${util}>)
ENDFOREACH()
