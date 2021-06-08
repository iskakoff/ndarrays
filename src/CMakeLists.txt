project(Common CXX)

add_custom_target(tensor-lib SOURCES index_sequence.h ndarray.h ndarray_math.h)

#target_link_libraries(h5pp-lib INTERFACE ${ALPSCore_LIBRARIES})