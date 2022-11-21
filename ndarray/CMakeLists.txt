project(ndarray-lib CXX)

#add_custom_target(tensor-lib SOURCES ndarray.h ndarray_math.h string_utils.h)

add_library(ndarray-lib_c INTERFACE)
target_include_directories(ndarray-lib_c INTERFACE .)
#target_link_libraries(h5pp-lib INTERFACE ${ALPSCore_LIBRARIES})