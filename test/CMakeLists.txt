project(TensorTests)
add_definitions(-DTEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(runtests
    tests_main.cpp test_init.cpp)

target_link_libraries(runtests)
