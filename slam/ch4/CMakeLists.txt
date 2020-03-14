cmake_minimum_required( VERSION 2.7 )

project(useSophus)

# 为使用Sophus，使用find_package命令找到它
find_package(Sophus REQUIRED)
include_directories(${Sophus_INLCUDE_DIRS})

#Eigen
include_directories("/usr/inlcude/eigen3")

add_executable(useSophus useSophus.cpp)

target_link_libraries( useSophus ${Sophus_LIBRARIES} )