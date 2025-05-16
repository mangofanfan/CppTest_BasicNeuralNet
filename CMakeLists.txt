cmake_minimum_required(VERSION 3.10.0)
project(CppTest10 VERSION 0.1.0 LANGUAGES C CXX)

file(GLOB_RECURSE neural_net_SRC
    "neuralnet/*.cpp"
    "neuralnet/*.h")

add_library(neural_net ${neural_net_SRC})

target_include_directories(neural_net
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/neuralnet>
        $<INSTALL_INTERFACE:include>)

add_executable(CppTest10 main.cpp)

target_link_libraries(CppTest10
    PRIVATE
        neural_net)

