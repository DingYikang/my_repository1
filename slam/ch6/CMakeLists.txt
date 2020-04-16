cmake_minimum_required(VERSION 2.8)
project(ch6)

set(CMAKE_BUILD_TYPE Release)

#设置使用C++11编译
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
#set(Ceres_DIR /usr/local/include/Ceres)
#find_package(Ceres REQUIRED)
#include_directories(${CERES_INCLUDE_DIRS})

# g2o
#list(APPEND CMAKE_MODULE_PATH  /home/yikang/g2o/cmake_modules)
#set(G2O_DIR /home/yikang/g2o)
#find_package(G2O REQUIRED)
#include_directories(${G2O_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(self_gaussionNewton self_gaussionNewton.cpp)
target_link_libraries(self_gaussionNewton ${OpenCV_LIBS})

#add_executable(ceresCurveFitting ceresCurveFitting.cpp)
#target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

#add_executable(g2oCurveFitting g2oCurveFitting.cpp)
#target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} g2o_core g2o_stuff)