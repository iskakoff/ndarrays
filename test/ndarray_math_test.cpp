/*
 * Copyright (c) 2021-2022 Sergei Iskakov
 *
 */

#include <gtest/gtest.h>

#include <ndarray_math.h>

#include "common.h"

TEST(NDArrayMathTest, MathAddSub) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4);
  initialize_array(arr1);
  ndarray::ndarray<double> arr2(1, 2, 3, 4);
  initialize_array(arr2);
  ndarray::ndarray<double> arr3 = arr1 + arr2;
  ASSERT_NEAR(double(arr1(0, 1, 2, 0) + arr2(0, 1, 2, 0)), arr3(0, 1, 2, 0), 1e-12);
  ndarray::ndarray<double> arr4 = arr1.copy();
  arr1 += arr2;
  ASSERT_NEAR(arr1(0, 1, 2, 0), arr3(0, 1, 2, 0), 1e-12);
  arr1 -= arr2;
  ASSERT_NEAR(arr1(0, 1, 0, 2), arr4(0, 1, 0, 2), 1e-12);
}

TEST(NDArrayMathTest, InplaceMathAddSub) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4);
  initialize_array(arr1);
  ndarray::ndarray<double> arr2(1, 2, 3, 4);
  initialize_array(arr2);
  ndarray::ndarray<double> arr3 = arr1(0, 1);
  ndarray::ndarray<double> arr4 = arr2(0, 0);

  ndarray::ndarray<double> arr5 = arr3.copy();
  ndarray::ndarray<double> arr6 = arr4.copy();


  arr3+=arr4;
  arr5+=arr6;

  ASSERT_NEAR(arr3(0, 1), arr5(0, 1), 1e-12);
  arr3 -= arr4;
  arr5 -= arr6;
  ASSERT_NEAR(arr3(1, 2), arr5(1, 2), 1e-12);
}

TEST(NDArrayMathTest, MathAddSubConversion) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4);
  initialize_array(arr1);
  ndarray::ndarray<std::complex<double> > arr2(1, 2, 3, 4);
  initialize_array(arr2);
  ndarray::ndarray<std::complex<double> > arr3 = arr1 + arr2;
  ndarray::ndarray<std::complex<double> > arr4 = arr3 - arr1;
  std::complex<double> a1 = arr1(0, 1, 0, 2);
  std::complex<double> a2 = arr2(0, 1, 0, 2);
  std::complex<double> a3 = arr3(0, 1, 0, 2);
  std::complex<double> a4 = arr4(0, 1, 0, 2);

  std::complex<double> a12 = a1 + a2;
  ASSERT_NEAR(a12.real(), a3.real(), 1e-12);
  ASSERT_NEAR(a2.real(), a4.real(), 1e-12);
}


TEST(NDArrayMathTest, MathScalarAddSub) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4);
  initialize_array(arr1);
  double shift = 15.0;
  ndarray::ndarray<double> arr2 = arr1 + shift;
  ASSERT_NEAR(arr1(0, 1, 2, 2) + 15.0, arr2(0, 1, 2, 2), 1e-12);
  ndarray::ndarray<double> arr3 = arr2 - shift;
  ASSERT_NEAR(arr1(0, 1, 2, 0), arr3(0, 1, 2, 0), 1e-12);
  ndarray::ndarray<double> arr4 = shift + arr1;
  ASSERT_NEAR(arr4(0, 1, 0, 2), arr2(0, 1, 0, 2), 1e-12);
}

TEST(NDArrayMathTest, UnaryOp) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4);
  initialize_array(arr1);
  ndarray::ndarray<double> arr2 = -arr1;
  ASSERT_TRUE(std::equal(arr1.begin(), arr1.end(), arr2.begin(),
                         [&](double a, double b) {return std::abs(a+b)<1e-12;})
                         );
}

TEST(NDArrayMathTest, Comparison) {
  ndarray::ndarray<double> arr1(1, 2, 3, 4);
  initialize_array(arr1);
  ndarray::ndarray<double> arr2(1, 2, 3, 4);
  arr2 += arr1;

  ndarray::ndarray<double> arr3(1, 2, 3, 4);
  initialize_array(arr3);
  ndarray::ndarray<std::complex<double>> arr4(1, 2, 3, 4);
  arr4 += arr3;

  ndarray::ndarray<int> arr5(1, 2, 3, 4);
  initialize_array(arr3);
  ndarray::ndarray<double> arr6(1, 2, 3, 4);
  arr6 += arr5;


  ASSERT_TRUE(arr1 == arr2);
  ASSERT_TRUE(arr3 == arr4);
  ASSERT_TRUE(arr5 == arr6);
}
