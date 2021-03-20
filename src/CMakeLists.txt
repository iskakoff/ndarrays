project(Common CXX)

add_custom_target(tensor-lib SOURCES tensor.h index_sequence.h)

#target_link_libraries(h5pp-lib INTERFACE ${ALPSCore_LIBRARIES})