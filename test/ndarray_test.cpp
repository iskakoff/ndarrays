/*
 * Copyright (c) 2021-2022 Sergei Iskakov
 *
 */

#include <gtest/gtest.h>

#include <complex>
#include <random>

#include <ndarray.h>

#include "common.h"


TEST(NDArrayTest, Init) {
  ndarray::ndarray<double> array(1, 2, 3, 4, 5);
  ASSERT_EQ(array.size(), 1 * 2 * 3 * 4 * 5);
  ASSERT_EQ(array.strides()[0], 120);
  ASSERT_EQ(array.strides()[1], 60);
  ASSERT_EQ(array.strides()[4], 1);
  ASSERT_EQ(array.shape()[0], 1);
  ASSERT_EQ(array.shape()[1], 2);
  ASSERT_EQ(array.shape()[3], 4);
}

TEST(NDArrayTest, Slice) {
  ndarray::ndarray<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  ndarray::ndarray<double> array2(array, 0, 1);
  ASSERT_EQ(array2.size(), 3 * 4 * 5);
  ASSERT_EQ(array2.strides()[0], 20);
  ASSERT_EQ(array2.strides()[1], 5);
  ASSERT_EQ(array2.strides()[2], 1);
  ASSERT_EQ(array2.shape()[0], 3);
  ASSERT_EQ(array2.shape()[1], 4);
  ASSERT_EQ(array2.shape()[2], 5);

  ndarray::ndarray<double> array3 = array2(2);

  ASSERT_EQ(array3.size(), 4 * 5);
  ASSERT_EQ(array3.strides()[0], 5);
  ASSERT_EQ(array3.strides()[1], 1);
  ASSERT_EQ(array3.shape()[0], 4);
  ASSERT_EQ(array3.shape()[1], 5);
}

TEST(NDArrayTest, Scalar) {
  ndarray::ndarray<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  float value = array(0, 1, 2, 3, 4);
  std::complex<double> value2 = array(0, 1, 2, 3, 4);
  ASSERT_NEAR(value, value2.real(), 1e-8);

  // take reference to an element
  double &val = array(0, 1, 2, 3, 4);

  // take a slice
  ndarray::ndarray<double> slice = array(0, 1);
  // check that value in the slice points to the same data
  ASSERT_NEAR(val, slice(2, 3, 4), 1e-12);
  // change value at the reference point
  val = 3.0;
  // check that value of the slice has been correctly changed
  ASSERT_NEAR(val, slice(2, 3, 4), 1e-12);

  double v;
#ifndef NDEBUG
  EXPECT_ANY_THROW((v = array(0, 1)));
#endif
  array(0, 1, 1, 1, 1) = 33.0;
  ASSERT_NEAR(33.0, slice(1, 1, 1), 1e-12);

}

#ifndef NDEBUG
TEST(NDArrayTest, WrongDimensions) {
  ndarray::ndarray<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  // throw if number of indices is larger than dimension
  EXPECT_ANY_THROW(array(0, 0, 0, 0, 0, 0));
  // throw if index value is larger than size of corresponding dimension
  EXPECT_ANY_THROW(array(5, 5));
  // the same for constructors
  EXPECT_ANY_THROW(ndarray::ndarray<double>(array, 1, 2, 3, 4, 5));
  EXPECT_ANY_THROW(ndarray::ndarray<double>(array, 0, 0, 0, 0, 0, 0));
}
#endif

void test_const_array(ndarray::ndarray<double> &arr1, const ndarray::ndarray<double> &arr2) {
  ndarray::ndarray<const double> slice = arr2(0, 1, 2);
  ndarray::ndarray<const double> slice2 = slice(0, 0);
  ASSERT_NEAR(arr1(0, 1, 2, 0, 0), slice2, 1e-12);
}

TEST(NDArrayTest, ConstArray) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4, 5);
  ndarray::ndarray<double> arr2(1, 2, 3, 4, 5);
  initialize_array(arr1);
  arr2 = arr1;
  test_const_array(arr1, arr2);

}

TEST(NDArrayTest, Copy) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4, 5);
  initialize_array(arr1);
  // create const array
  const ndarray::ndarray<double> arr2 = arr1.copy();
  // make copy of const array to array of consts
  ndarray::ndarray<const double> arr3 = arr2.copy();
  // copy of array of consts to array of consts
  ndarray::ndarray<const double> arr4 = arr3.copy();
  // copy of array of consts to non-const array
  ndarray::ndarray<double> arr5 = arr3.copy();
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

TEST(NDArrayTest, CopyOfSlice) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4, 5);
  initialize_array(arr1);
  // create const array
  ndarray::ndarray<double> arr2 = arr1(0, 1);
  // make copy of const array to array of consts
  ndarray::ndarray<double> arr3 = arr2.copy();
  ASSERT_NEAR(arr1(0, 1, 2, 0, 0), arr2(2, 0, 0), 1e-12);
  ASSERT_NEAR(arr1(0, 1, 2, 0, 0), arr3(2, 0, 0), 1e-12);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        ASSERT_NEAR(arr2(i, j, k), arr3(i, j, k), 1e-12);
      }
    }
  }
  // change value in origin
  arr1(0, 1, 2, 2, 2) = -5;
  ASSERT_FALSE(std::abs(double(arr2(2, 2, 2)) - double(arr3(2, 2, 2))) < 1e-12);
}


TEST(NDArrayTest, SetValue) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4, 5);
  initialize_array(arr1);
  double value = arr1(0,0,0,0,0);
  arr1.set_value(value + 2.0);
  ASSERT_TRUE(std::all_of(arr1.begin(), arr1.end(),
                          [&](double x) {return std::abs(x-(value + 2.0))<1e-12;})
                          );
  arr1.set_zero();
  ASSERT_TRUE(std::all_of(arr1.begin(), arr1.end(),
                          [&](double x) {return std::abs(x)<1e-12;})
  );
}

TEST(NDArrayTest, Reshape) {
  ndarray::ndarray<double> array(1, 2, 3, 4, 5);
  initialize_array(array);
  std::vector<size_t> shape{1, 2, 30, 2};
  std::vector<size_t> strides{120, 60, 2, 1};
  ndarray::ndarray<double> reshaped_array = array.reshape(shape);
  ASSERT_EQ(reshaped_array.shape(), shape);
  ASSERT_EQ(reshaped_array.strides(), strides);
}

TEST(NDArrayTest, RangeLoop) {
  ndarray::ndarray<double> array(50, 20, 3, 4);
  array.set_value(2.0);
  for(auto v : array) {
    ASSERT_NEAR(v, 2.0, 1e-12);
  }
}


TEST(NDArrayTest, ElementAccess) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4, 5);
  initialize_array(arr1);
  ndarray::ndarray<double> arr2 = arr1(0,1,2);
  ASSERT_TRUE(arr1.at(0,1,2,1,1) == arr2(1,1));
}
