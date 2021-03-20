/*
 * Copyright (c) 2020 University of Michigan.
 *
 */

#include <catch2/catch.hpp>
#include <iostream>
#include <array>
#include <tensor.h>

TEST_CASE("InitTensorTest") {
  alps::tensor<double> x(3,4,5);
  REQUIRE(x.shape().size() == 3);
  REQUIRE(x.dim() == 3);
  REQUIRE(x.strides()[0] == 20);
  REQUIRE(x.strides()[1] == 5);
  REQUIRE(x.strides()[2] == 1);
  REQUIRE(x.size() == 3*4*5);
}

TEST_CASE("AssgnmentTest") {
  alps::tensor<double> x(3,4,5);
  REQUIRE_NOTHROW(x(1,2,3) = 4);
  REQUIRE(x(1,2,3) == 4);
}

alps::tensor<double> get_tensor() {
  alps::tensor<double> x(1,2,3,4);
  x(0,0,0,1) = 5;
  auto ll = x[{0,0,0}];
  ll(0) = 3;
  return std::move(ll);
}

TEST_CASE("SlicesTest") {
  alps::tensor<double> x(3,4,5,6);
  x(1,2,3,0) = 14;
  alps::tensor<double> y = x [ {1, 2, 3} ];
  alps::tensor<double> y2 = y;

  REQUIRE_THROWS(x[{1,2,3,4}]);
  REQUIRE(y(0) == x(1,2,3,0));
  REQUIRE_NOTHROW(y(3) = 4);
  REQUIRE(y(3) == x(1,2,3,3));

  alps::tensor<double> z  =  get_tensor();
  REQUIRE(z(1) == 5);
  REQUIRE(z(0) == 3);
}