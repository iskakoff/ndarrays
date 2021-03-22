/*
 * Copyright (c) 2020 University of Michigan.
 *
 */

#include <gtest/gtest.h>
#include <array>
#include <tensor.h>

TEST(TensorTest, InitTensorTest) {
  alps::tensor<double> x(3, 4, 5);
  ASSERT_EQ(x.shape().size(), 3);
  ASSERT_EQ(x.dim(), 3);
  ASSERT_EQ(x.strides()[0], 20);
  ASSERT_EQ(x.strides()[1], 5);
  ASSERT_EQ(x.strides()[2], 1);
  ASSERT_EQ(x.size(), 3 * 4 * 5);
}

TEST(TensorTest, AssgnmentTest) {
  alps::tensor<double> x(3, 4, 5);
  EXPECT_NO_THROW((x(1, 2, 3) = 4));
  ASSERT_EQ(x(1, 2, 3), 4);
}

alps::tensor<double> get_tensor() {
  alps::tensor<double> x(1, 2, 3, 4);
  x(0, 0, 0, 1) = 5;
  auto ll = x[{0, 0, 0}];
  ll(0) = 3;
  return std::move(ll);
}

TEST(TensorTest, SlicesTest) {
  alps::tensor<double> x(3, 4, 5, 6);
  x(1, 2, 3, 0) = 14;
  alps::tensor<double> y = x[{1, 2, 3}];
  alps::tensor<double> y2 = y;

  EXPECT_ANY_THROW((x[{1, 2, 3, 4}]));
  ASSERT_EQ(y(0), x(1, 2, 3, 0));
  EXPECT_NO_THROW((y(3) = 4));
  ASSERT_EQ(y(3), x(1, 2, 3, 3));

  alps::tensor<double> z = get_tensor();
  ASSERT_EQ(z(1), 5);
  ASSERT_EQ(z(0), 3);
}