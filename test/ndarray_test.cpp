/*
 * Copyright (c) 2021 University of Michigan.
 *
 */


#include <catch2/catch.hpp>
#include <ndarray.h>


TEST_CASE("InitNDArrayTest") {
  ndarray<double> array(1, 2, 3, 4, 5);
  REQUIRE(array.size() == 1 * 2 * 3 * 4 * 5);
}