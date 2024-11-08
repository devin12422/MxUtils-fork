
cmake_minimum_required(VERSION 3.13)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_X86_FLAGS_BASE "-mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mfxsr -mcx16 -msahf -mpclmul")
#set(CMAKE_X86_FLAGS_AVX2 "-mavx -mavx2 -mfma -mbmi -mbmi2 -maes -mf16c -mfsgsbase -mlzcnt -mmovbe -mxsave")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
  set(
    CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -Wall -Wextra -pipe -fPIC -O3 ${CMAKE_X86_FLAGS_BASE} -mtune=generic -fopenmp"
  )
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 ${CMAKE_X86_FLAGS_BASE} -mtune=generic -fopenmp"
  )
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(AARCH64)|(arm64)|(ARM64)")
  set(
    CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=armv8-a+simd -fopenmp"
  )
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=armv8-a+simd -fopenmp"
  )
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(riscv64)|(RISCV64)|(RV64)|(rv64gc)")
  set(
    CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=rv64gc -fopenmp"
  )
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=rv64gc -fopenmp"
  )
endif()


include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB local_src
    "*.c"
    "*.cpp"
	)

set(MXUTIL_GUI_DYNAMIC_LIB "mxutils_gui")
set(MXUTIL_GUI_STATIC_LIB "mxutils_gui_static")


add_library(${MXUTIL_GUI_DYNAMIC_LIB} SHARED ${local_src})
add_library(${MXUTIL_GUI_STATIC_LIB} STATIC ${local_src})
target_link_libraries(${MXUTIL_GUI_DYNAMIC_LIB} PUBLIC  ${OpenCV_LIBS} Qt5::Widgets  Qt5::Core  Qt5::Gui)
target_link_libraries(${MXUTIL_GUI_STATIC_LIB} PUBLIC  ${OpenCV_LIBS} Qt5::Widgets  Qt5::Core  Qt5::Gui)


list(APPEND ALL_STATIC_UTILS ${MXUTIL_GUI_STATIC_LIB})
set(ALL_STATIC_UTILS ${ALL_STATIC_UTILS} PARENT_SCOPE)
