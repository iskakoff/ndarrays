project(TensorTests)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(runUnitTests tests_main.cpp test_init.cpp ndarray_test.cpp)

target_link_libraries(runUnitTests GTest::gtest GTest::gtest_main)

add_test(NAME runUnitTests COMMAND runUnitTests)