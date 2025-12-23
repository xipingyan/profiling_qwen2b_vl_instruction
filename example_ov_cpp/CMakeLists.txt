cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME "qwen2vl_app_cpp")

project($TARGET_NAME)

find_package(OpenVINO REQUIRED)
find_package(OpenVINOGenAI REQUIRED)
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found! Check your installation and environment.")
endif()

file(GLOB SOURCES "*.cpp")
add_executable(${TARGET_NAME} ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOUCE_DIR}" "${CMAKE_BINARY_DIR}", "${OpenCV_INCLUDE_DIRS}")

target_link_libraries(${TARGET_NAME} PRIVATE openvino::genai ${OpenCV_LIBS})