/*
 * Copyright (c) 2021 University of Michigan.
 *
 */

#include <random>


#include <catch2/catch.hpp>
#include <ndarray_t.h>
#include <complex>

void initialize_array(ndarray::ndarray_t<double> &array) {
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
  std::uniform_int_distribution<double> dist{0.0, 10.0};

  std::generate(array.data().get(), array.data().get() + array.size(), [&dist, &mersenne_engine]() -> double {
                  return dist(mersenne_engine);
                }
  );
}


TEST_CASE("InitNDArrayTest") {
  ndarray::ndarray_t<double> array(1, 2, 3, 4, 5);
  REQUIRE(array.size() == 1 * 2 * 3 * 4 * 5);
  REQUIRE(array.strides()[0] == 120);
  REQUIRE(array.strides()[1] == 60);
  REQUIRE(array.strides()[4] == 1);
  REQUIRE(array.shape()[0] == 1);
  REQUIRE(array.shape()[1] == 2);
  REQUIRE(array.shape()[3] == 4);
}

TEST_CASE("SliceNDArrayTest") {
  ndarray::ndarray_t<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  ndarray::ndarray_t<double> array2(array, 0, 1);
  REQUIRE(array2.size() == 3 * 4 * 5);
  REQUIRE(array2.strides()[0] == 20);
  REQUIRE(array2.strides()[1] == 5);
  REQUIRE(array2.strides()[2] == 1);
  REQUIRE(array2.shape()[0] == 3);
  REQUIRE(array2.shape()[1] == 4);
  REQUIRE(array2.shape()[2] == 5);

  ndarray::ndarray_t<double> array3 = array2(2);

  REQUIRE(array3.size() == 4 * 5);
  REQUIRE(array3.strides()[0] == 5);
  REQUIRE(array3.strides()[1] == 1);
  REQUIRE(array3.shape()[0] == 4);
  REQUIRE(array3.shape()[1] == 5);


}

TEST_CASE("ScalarNDArrayTest") {
  ndarray::ndarray_t<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  float value = array(0, 1, 2, 3, 4);
  std::complex<double> value2 = array(0, 1, 2, 3, 4);
  REQUIRE((value - value2.real()) == Approx(0.0));

  // take reference to an element
  double &val = array(0, 1, 2, 3, 4);

  // take a slice
  ndarray::ndarray_t<double> slice = array(0, 1);
  // check that value in the slice points to the same data
  REQUIRE((val - slice(2, 3, 4)) == Approx(0.0));
  // change value at the reference point
  val = 3.0;
  // check that value of the slice has been correctly changed
  REQUIRE((val - slice(2, 3, 4)) == Approx(0.0));

  double v;
  REQUIRE_THROWS((v = array(0, 1)));

  array(0, 1, 1, 1, 1) = 33.0;
  REQUIRE((33.0 - slice(1, 1, 1)) == Approx(0.0));

}


