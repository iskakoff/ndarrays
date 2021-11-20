/*
 * Copyright (c) 2021 University of Michigan.
 *
 */

#ifndef ALPS_NDARRAY_MATH_H
#define ALPS_NDARRAY_MATH_H

#include "ndarray.h"

namespace ndarray {

  // Arithmetic operations on tensors

  // inplace operators

  template<typename T1, typename T2>
  typename std::enable_if<std::is_convertible<T2, T1>::value, ndarray < T1> >::type &
  operator+=(ndarray <T1> &first, const ndarray <T2> &second) {
    using result_t = decltype(T1{} + T2{});
#ifndef NDEBUG
    if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    std::transform(first.data().get() + first.offset(), first.data().get() + first.offset() + first.size(),
                   second.data().get() + second.offset(),
                   first.data().get(), [&](const T1 f, const T2 s) {
          return result_t(f) + result_t(s);
        });
    return first;
  }

  template<typename T1, typename T2>
  typename std::enable_if<std::is_convertible<T2, T1>::value, ndarray < T1> >::type &
  operator-=(ndarray <T1> &first, const ndarray <T2> &second) {
    using result_t = decltype(T1{} - T2{});
#ifndef NDEBUG
    if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    std::transform(first.data().get() + first.offset(), first.data().get() + first.offset() + first.size(),
                   second.data().get() + second.offset(),
                   first.data().get(), [&](const T1 f, const T2 s) {
          return result_t(f) - result_t(s);
        });
    return first;
  }

  // Binary operations with tensors
  template<typename T1, typename T2>
  ndarray<decltype(T1{} + T2{})> operator+(const ndarray <T1> &first, const ndarray <T2> &second) {
    using result_t = decltype(T1{} + T2{});
#ifndef NDEBUG
    if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    ndarray<result_t> result(first.shape());
    std::transform(first.data().get() + first.offset(), first.data().get() + first.offset() + first.size(),
                   second.data().get() + second.offset(),
                   result.data().get(), [&](const T1 f, const T2 s) {
          return result_t(f) + result_t(s);
        });
    return result;
  };


  template<typename T1, typename T2>
  ndarray<decltype(T1{} - T2{})> operator-(const ndarray <T1> &first, const ndarray <T2> &second) {
    using result_t = decltype(T1{} - T2{});
#ifndef NDEBUG
    if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    ndarray<result_t> result(first.shape());
    std::transform(first.data().get() + first.offset(), first.data().get() + first.offset() + first.size(),
                   second.data().get() + second.offset(),
                   result.data().get(), [&](const T1 f, const T2 s) {
          return result_t(f) - result_t(s);
        });
    return result;
  };


  // Binary operations with scalars
  template<typename T1, typename T2>
  typename std::enable_if<is_scalar<T2>::value, ndarray < decltype(T1{} + T2{})> >::type
  operator+(const ndarray <T1> &first, T2 second) {
    using result_t = decltype(T1{} + T2{});
    ndarray<result_t> result(first.shape());
    std::transform(first.data().get() + first.offset(), first.data().get() + first.offset() + first.size(),
                   result.data().get(), [&](const T1 f) {
          return result_t(f) + result_t(second);
        });
    return result;
  };

  template<typename T1, typename T2>
  typename std::enable_if<is_scalar<T1>::value, ndarray < decltype(T1{} + T2{})> >::type
  operator+(T1 first, const ndarray <T2> &second) {
    return second + first;
  }

  template<typename T1, typename T2>
  typename std::enable_if<is_scalar<T2>::value, ndarray < decltype(T1{} - T2{})> >::type
  operator-(const ndarray <T1> &first, T2 second) {
    using result_t = decltype(T1{} - T2{});
    ndarray<result_t> result(first.shape());
    std::transform(first.data().get() + first.offset(), first.data().get() + first.offset() + first.size(),
                   result.data().get(), [&](const T1 f) {
          return result_t(f) - result_t(second);
        });
    return result;
  };

  template<typename T1, typename T2>
  typename std::enable_if<is_scalar<T1>::value, ndarray < decltype(T1{} - T2{})> >::type
  operator-(T1 first, const ndarray <T2> &second) {
    return second - first;
  }

}


#endif // ALPS_NDARRAY_MATH_H
