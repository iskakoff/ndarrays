cmake_minimum_required(VERSION 3.12)
project(Ndarray)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include_directories(libs)
include_directories(src)

add_subdirectory(src)

option(TESTING "Enable testing" ON)
if (TESTING)
    add_subdirectory(test)
endif (TESTING)
