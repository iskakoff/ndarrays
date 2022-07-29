project(Common CXX)

#add_custom_target(tensor-lib SOURCES ndarray.h ndarray_math.h string_utils.h)

add_library(tensor-lib INTERFACE)
target_include_directories(tensor-lib INTERFACE .)
#target_link_libraries(h5pp-lib INTERFACE ${ALPSCore_LIBRARIES})