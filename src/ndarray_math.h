/*
 * Copyright (c) 2021 University of Michigan.
 *
 */

#ifndef TENSORS_NDARRAY_MATH_H
#define TENSORS_NDARRAY_MATH_H

#include "ndarray_t.h"

//namespace ndarray {

template<typename T1, typename T2>
ndarray::ndarray_t<decltype(T1{} + T2{})> operator+(const ndarray::ndarray_t<T1> &first, const ndarray::ndarray_t<T2> &second) {
  if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
    throw std::runtime_error("Arrays size is miss matched.");
  }
  ndarray::ndarray_t<decltype(T1{} + T2{})> result(first.shape());
  std::transform(first.data().get() + first.offset(), first.data().get() + first.offset() + first.size(),
                 second.data().get() + second.offset(),
                 result.data().get(), [&](const T1 f, const T2 s) {
        return decltype(T1{} + T2{})(f + s);
      });
  return result;
};
//}


#endif //TENSORS_NDARRAY_MATH_H
