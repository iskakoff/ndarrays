/*
 * Copyright (c) 2021-2022 Sergei Iskakov
 *
 */

#ifndef ALPS_NDARRAY_MATH_H
#define ALPS_NDARRAY_MATH_H

#include <ndarray/ndarray.h>

namespace ndarray {

  namespace detail {
    template<typename T>
    ndarray<T> transpose_impl(const ndarray<T>& array, const std::vector<size_t> &pattern) {
      std::vector<size_t> shape(array.shape().size());
      for (size_t i(0); i < array.shape().size(); ++i) {
        shape[pattern[i]] = array.shape()[i];
      }
      ndarray<T> result(shape);
      std::vector<size_t> indices(array.dim(), 0);
      for (size_t i(0); i < array.size(); ++i) {
        size_t res = i;
        for (size_t ind(0); ind < array.dim(); ++ind) {
          indices[pattern[array.dim() - ind - 1]] = res % array.shape()[array.dim() - ind - 1];
          res /= array.shape()[array.dim() - ind - 1];
        }
        std::transform(indices.begin(), indices.end(), result.strides().begin(), indices.begin(),
                       std::multiplies<size_t>());
        size_t ind = std::accumulate(indices.begin(), indices.end(), size_t(0), std::plus<size_t>());
        result.data().get()[ind] = array.data().get()[array.offset() + i];
      }
      return result;
    }
  }

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
    std::transform(first.begin(), first.end(),
                   second.begin(),
                   first.begin(), [&](const T1 f, const T2 s) {
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
    std::transform(first.begin(), first.end(),
                   second.begin(),
                   first.begin(), [&](const T1 f, const T2 s) {
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
    std::transform(first.begin(), first.end(),
                   second.begin(),
                   result.begin(), [&](const T1 f, const T2 s) {
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
    std::transform(first.begin(), first.end(),
                   second.begin(),
                   result.begin(), [&](const T1 f, const T2 s) {
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
    std::transform(first.begin(), first.end(), result.begin(), [&](const T1 f) {
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
    std::transform(first.begin(), first.end(),
                   result.begin(), [&](const T1 f) {
          return result_t(f) - result_t(second);
        });
    return result;
  };

  template<typename T1, typename T2>
  typename std::enable_if<is_scalar<T1>::value, ndarray < decltype(T1{} - T2{})> >::type
  operator-(T1 first, const ndarray <T2> &second) {
    return second - first;
  }

  // Unary operation

  template<typename T1>
  ndarray<T1> operator-(const ndarray <T1> &first) {
    ndarray<T1> result(first.shape());
    std::transform(first.begin(), first.end(),
                   result.begin(), [&](const T1 f) {return -f;});
    return result;
  };

  // Comparisons

  template<typename T1, typename T2>
  bool operator==(const ndarray <T1> &lhs, const ndarray <T2> &rhs) {
    using result_t = decltype(T1{} + T2{});
#ifndef NDEBUG
    if (!std::equal(lhs.shape().begin(), lhs.shape().end(), rhs.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](T1 l, T2 r) {
      return std::abs(result_t(l) - result_t(r))< 1e-12;
    });
  };


  template<typename T>
  ndarray<T> transpose(const ndarray<T>& array, const std::string &string_pattern) {
    size_t find = string_pattern.find("->");
    if (find == std::string::npos) {
      throw std::runtime_error("Incorrect transpose_impl pattern.");
    }
    std::string from = trim(string_pattern.substr(0, find));
    std::string to = trim(string_pattern.substr(find + 2, string_pattern.size() - 1));

    if (from.length() != to.length()) {
      throw std::runtime_error("Transpose source and target indices have different size.");
    }
    if (from.length() != array.dim()) {
      throw std::runtime_error("Number of transpose_impl indices and array dimension are different size.");
    }
    if((!all_latin(from)) || (!all_latin(to))) {
      throw std::runtime_error("Transpose indices should be latin letters.");
    }

#ifndef NDEBUG
    for(const auto & s1 : from) {
      bool in = false;
      for(const auto & s2 : to) {
        if(s1 == s2) {
          in = true;
          break;
        }
      }
      if(!in) {
        throw std::runtime_error("Some LHS transpose indices are not found in RHS transpose_impl indices.");
      }
    }
#endif

    std::map<char, size_t> index_map;
    for (size_t i = 0; i < to.length(); ++i) {
      index_map[to[i]] = i;
    }
    std::vector<size_t> pattern(to.length());
    for (size_t j = 0; j < from.length(); ++j) {
      pattern[j] = index_map[from[j]];
    }
    return detail::transpose_impl(array, pattern);
  }

}


#endif // ALPS_NDARRAY_MATH_H
