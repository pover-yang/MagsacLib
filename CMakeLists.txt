cmake_minimum_required(VERSION 3.20)
project (MagsacLib)

set(CMAKE_CXX_STANDARD 14)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if the sample project should be generated
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" ON)

SET (CMAKE_BUILD_TYPE "DEBUG")

IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-fpermissive")
# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV)
if (OPENCV_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
    add_definitions(-DUSE_OPENMP)
    set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (OPENCV_FOUND)

# ==============================================================================
# Find GFlags
# ==============================================================================
find_package(gflags REQUIRED)
if(DEFINED gflags_VERSION_STRING OR DEFINED GFLAGS_VERSION_STRING)
    # different cmake variable with lower or upper case exist
    set(GFLAGS_INCLUDE_DIR ${GFLAGS_INCLUDE_DIR})
    set(GFLAGS_LIBRARIES ${GFLAGS_LIBRARIES})
    message( "Found Gflags ${GFLAGS_VERSION_STRING}" )
    message( STATUS "> GFLAGS_INCLUDE_DIR:   ${GFLAGS_INCLUDE_DIR}" )
    message( STATUS "> GFLAGS_LIBRARIES:   ${GFLAGS_LIBRARIES}" )
endif()

# ==============================================================================
# Find GLOG
# ==============================================================================
find_package(Glog REQUIRED)
if(NOT Glog_FOUND)
    message(FATAL_ERROR "Glog not found.")
else()
    message("Glog library found.")
endif()

if(IS_MSVC)
    # Some fixes for the Glog library.
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()


# ==============================================================================
# Find Eigen
# ==============================================================================
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# ==============================================================================
# Find OpenMP
# ==============================================================================
if(APPLE)
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
    set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    set(OpenMP_C "${CMAKE_C_COMPILER}")
    set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
    set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
endif()

find_package(OpenMP)
if (USE_OPENMP)
    if(NOT OPENMP_FOUND)
        message(FATAL_ERROR "OPENMP not found.")
    endif()
    add_definitions(-DUSE_OPENMP)
    set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (USE_OPENMP)

# ==============================================================================
# Includes
# ==============================================================================
SET(SOURCE_DIR "magsac")

include_directories (
        ${PROJECT_SOURCE_DIR}/graph-cut-ransac/src/pygcransac/include
)


# ==============================================================================
# Structure: MAGSAC Library
# ==============================================================================

# Tell cmake that headers are in alse in source_dir
include_directories (${SOURCE_DIR}/include)

# Set header files for the library
file(GLOB_RECURSE HDRS_MAGSAC
        "${SOURCE_DIR}/include/*.h"
        )

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_MAGSAC
        "${SOURCE_DIR}/include/*.cpp"
        )

add_library(
        ${PROJECT_NAME} STATIC
        ${HDRS_MAGSAC}
        ${SRCS_MAGSAC})

target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        glog::glog
        gflags)

add_executable(SampleProject cpp_example.cpp)
target_link_libraries(SampleProject PUBLIC
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        glog::glog
        gflags
        )