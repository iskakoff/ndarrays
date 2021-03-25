/*
 * Copyright (c) 2021 University of Michigan.
 *
 */

#include <random>

#include <gtest/gtest.h>
#include <ndarray_t.h>
#include <complex>

void initialize_array(ndarray::ndarray_t<double> &array) {
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine(1);  // Generates pseudo-random integers
  std::uniform_int_distribution<double> dist{0.0, 10.0};

  std::generate(array.data().get(), array.data().get() + array.size(), [&dist, &mersenne_engine]() -> double {
                  return dist(mersenne_engine);
                }
  );
}


TEST(NDArrayTest, Init) {
  ndarray::ndarray_t<double> array(1, 2, 3, 4, 5);
  ASSERT_EQ(array.size(), 1 * 2 * 3 * 4 * 5);
  ASSERT_EQ(array.strides()[0], 120);
  ASSERT_EQ(array.strides()[1], 60);
  ASSERT_EQ(array.strides()[4], 1);
  ASSERT_EQ(array.shape()[0], 1);
  ASSERT_EQ(array.shape()[1], 2);
  ASSERT_EQ(array.shape()[3], 4);
}

TEST(NDArrayTest, Slice) {
  ndarray::ndarray_t<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  ndarray::ndarray_t<double> array2(array, 0, 1);
  ASSERT_EQ(array2.size(), 3 * 4 * 5);
  ASSERT_EQ(array2.strides()[0], 20);
  ASSERT_EQ(array2.strides()[1], 5);
  ASSERT_EQ(array2.strides()[2], 1);
  ASSERT_EQ(array2.shape()[0], 3);
  ASSERT_EQ(array2.shape()[1], 4);
  ASSERT_EQ(array2.shape()[2], 5);

  ndarray::ndarray_t<double> array3 = array2(2);

  ASSERT_EQ(array3.size(), 4 * 5);
  ASSERT_EQ(array3.strides()[0], 5);
  ASSERT_EQ(array3.strides()[1], 1);
  ASSERT_EQ(array3.shape()[0], 4);
  ASSERT_EQ(array3.shape()[1], 5);
}

TEST(NDArrayTest, Scalar) {
  ndarray::ndarray_t<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  float value = array(0, 1, 2, 3, 4);
  std::complex<double> value2 = array(0, 1, 2, 3, 4);
  ASSERT_NEAR(value, value2.real(), 1e-12);

  // take reference to an element
  double &val = array(0, 1, 2, 3, 4);

  // take a slice
  ndarray::ndarray_t<double> slice = array(0, 1);
  // check that value in the slice points to the same data
  ASSERT_NEAR(val, slice(2, 3, 4), 1e-12);
  // change value at the reference point
  val = 3.0;
  // check that value of the slice has been correctly changed
  ASSERT_NEAR(val, slice(2, 3, 4), 1e-12);

  double v;
  EXPECT_ANY_THROW((v = array(0, 1)));

  array(0, 1, 1, 1, 1) = 33.0;
  ASSERT_NEAR(33.0, slice(1, 1, 1), 1e-12);

}

TEST(NDArrayTest, WrongDimensions) {
  ndarray::ndarray_t<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  // throw if number of indices is larger than dimension
  EXPECT_ANY_THROW(array(0, 0, 0, 0, 0, 0));
  // throw if index value is larger than size of corresponding dimension
  EXPECT_ANY_THROW(array(5, 5));
  // the same for constructors
  EXPECT_ANY_THROW(ndarray::ndarray_t<double>(array, 1, 2, 3, 4, 5));
  EXPECT_ANY_THROW(ndarray::ndarray_t<double>(array, 0, 0, 0, 0, 0, 0));
}

void test_const_array(ndarray::ndarray_t<double> &arr1, const ndarray::ndarray_t<double> &arr2) {
  ndarray::ndarray_t<const double> slice = arr2(0, 1, 2);
  ndarray::ndarray_t<const double> slice2 = slice(0, 0);
  ASSERT_NEAR(arr1(0, 1, 2, 0, 0), slice2, 1e-12);
}

TEST(NDArrayTest, ConstArray) {
  ndarray::ndarray_t<double> arr1(1, 2, 3, 4, 5);
  ndarray::ndarray_t<double> arr2(1, 2, 3, 4, 5);
  initialize_array(arr1);
  arr2 = arr1;
  test_const_array(arr1, arr2);

}

TEST(NDArrayTest, Copy) {
  ndarray::ndarray_t<double> arr1(1, 2, 3, 4, 5);
  initialize_array(arr1);
  // create const array
  const ndarray::ndarray_t<double> arr2 = arr1.copy();
  // make copy of const array to array of consts
  ndarray::ndarray_t<const double> arr3 = arr2.copy();
  // copy of array of consts to array of consts
  ndarray::ndarray_t<const double> arr4 = arr3.copy();
  // copy of array of consts to non-const array
  ndarray::ndarray_t<double> arr5 = arr3.copy();
  ASSERT_NEAR(arr1(0, 1, 2, 0, 0), arr2(0, 1, 2, 0, 0), 1e-12);
  ASSERT_NEAR(arr1(0, 1, 2, 0, 0), arr3(0, 1, 2, 0, 0), 1e-12);
  ASSERT_NEAR(arr1(0, 1, 2, 0, 0), arr4(0, 1, 2, 0, 0), 1e-12);
  ASSERT_NEAR(arr1(0, 1, 2, 0, 0), arr5(0, 1, 2, 0, 0), 1e-12);
  // change value in origin
  arr1(0, 1, 2, 0, 0) = -5;
  ASSERT_FALSE(std::abs(double(arr1(0, 1, 2, 0, 0)) - double(arr2(0, 1, 2, 0, 0))) < 1e-9);
  ASSERT_FALSE(std::abs(double(arr1(0, 1, 2, 0, 0)) - double(arr3(0, 1, 2, 0, 0))) < 1e-9);
  ASSERT_FALSE(std::abs(double(arr1(0, 1, 2, 0, 0)) - double(arr4(0, 1, 2, 0, 0))) < 1e-9);
  ASSERT_FALSE(std::abs(double(arr1(0, 1, 2, 0, 0)) - double(arr5(0, 1, 2, 0, 0))) < 1e-9);
}

