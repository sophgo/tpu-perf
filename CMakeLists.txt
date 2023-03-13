cmake_minimum_required(VERSION 3.10)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/python/tpu_perf CACHE PATH "Install prefix")

project(tpu-perf)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(sg REQUIRED)
file(GLOB pipeline_srcs pipeline/*.cpp)
add_library(pipeline SHARED ${pipeline_srcs} ${tool_srcs})
target_include_directories(
    pipeline PUBLIC
    ${CMAKE_SOURCE_DIR}/pipeline)
target_compile_features(pipeline PUBLIC cxx_std_11)
target_link_libraries(pipeline PUBLIC ${sg_LIBRARIES})

find_package(Threads REQUIRED)
file(GLOB benchmark_srcs benchmark/*.cpp)
add_library(benchmark SHARED ${benchmark_srcs} ${tool_srcs})
target_include_directories(
    benchmark PUBLIC
    ${CMAKE_SOURCE_DIR}/benchmark)
target_compile_features(benchmark PUBLIC cxx_std_11)
target_link_libraries(benchmark PUBLIC ${sg_LIBRARIES} Threads::Threads)

set(proto_srcs
    ${CMAKE_BINARY_DIR}/blob_pb2.py
    ${CMAKE_BINARY_DIR}/blob.pb.cc)
find_program(PROTOC protoc)
if (NOT PROTOC)
    message(FATAL_ERROR
        "\nprotoc not found.\nInstall by `sudo apt-get install -y protobuf-compiler`")
endif()
add_custom_command(
    OUTPUT ${proto_srcs}
    COMMAND ${PROTOC}
        -I=${CMAKE_SOURCE_DIR}
        --cpp_out=${CMAKE_BINARY_DIR}
        --python_out=${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/blob.proto
    DEPENDS ${CMAKE_SOURCE_DIR}/blob.proto
    VERBATIM)
add_custom_target(proto ALL DEPENDS ${proto_srcs})

configure_file(cmake/bdist.sh.in bdist.sh)

install(TARGETS pipeline LIBRARY DESTINATION .)
install(FILES ${proto_srcs} DESTINATION .)
install(SCRIPT ${CMAKE_SOURCE_DIR}/cmake/postinst.cmake)
