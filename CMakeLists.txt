cmake_minimum_required(VERSION 3.12)
project(ndarray CXX)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include_directories(libs)
include_directories(ndarray)

add_subdirectory(ndarray)

option(TESTING "Enable testing" ON)
if (TESTING)
    add_subdirectory(test)
endif (TESTING)


add_library(${PROJECT_NAME}_c INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME}_c ALIAS ${PROJECT_NAME}_c)

target_include_directories(${PROJECT_NAME}_c INTERFACE .)

