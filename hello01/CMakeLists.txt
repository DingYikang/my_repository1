# 声明要求的cmake最低版本
cmake_minimum_required( VERSION 2.8 )

# 声明一个cmake工程
project( HelloWorld )

# 添加一个可执行程序
# 语法为： add_executable( 程序名  源代码文件 ）
add_executable(Helloworld HelloWorld.cpp)

# 添加一个库文件
add_library(hello libHelloWorld.cpp)

# 添加一个共享库文件
add_library(hello_shared SHARED libHelloWorld.cpp)

# 再添加一个useHello文件
add_executable( useHello useHello.cpp )

# 建立起useHello与库文件的链接
target_link_libraries( useHello hello )

# 设置编译模式
set( CMAKE_BUILD_TYPE "Debug" )
