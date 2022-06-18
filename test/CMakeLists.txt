project(TensorTests)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(runUnitTests tests_main.cpp ndarray_test.cpp ndarray_math_test.cpp)

target_link_libraries(runUnitTests gtest_main)

include(GoogleTest)
gtest_discover_tests(runUnitTests)