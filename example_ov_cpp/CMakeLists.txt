cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME "qwen2vl_app_cpp")

project($TARGET_NAME)

find_package(OpenVINO REQUIRED)
find_package(OpenVINOGenAI REQUIRED)

add_executable(${TARGET_NAME} main.cpp load_image.cpp llm_lookup.cpp)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOUCE_DIR}" "${CMAKE_BINARY_DIR}")

target_link_libraries(${TARGET_NAME} PRIVATE openvino::genai)