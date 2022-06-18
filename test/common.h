/*
 * Copyright (c) 2021-2022 Sergei Iskakov
 *
 */

#ifndef NDARRAY_COMMON_H
#define NDARRAY_COMMON_H

#include <ndarray.h>
#include <random>

template<typename T>
inline void initialize_array(ndarray::ndarray<T> &array) {
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine(1);  // Generates pseudo-random integers
  std::uniform_real_distribution<double> dist{0.0, 10.0};

  std::generate(array.data().get(), array.data().get() + array.size(), [&dist, &mersenne_engine]() -> T {
                  return T(dist(mersenne_engine));
                }
  );
}

#endif //NDARRAY_COMMON_H
