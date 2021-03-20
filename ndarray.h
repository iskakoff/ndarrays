#pragma once

#include <array>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm>


template<typename T>
struct ndarray {

  template<typename...Indices>
  ndarray(size_t d1, Indices...inds) : ndarray(std::array<size_t, sizeof...(inds) + 1>{{d1, size_t(inds)...}}) {}
  template<size_t D>
  ndarray(const std::array<size_t, D> &dim) : size(get_size(dim)),
                                              shape(dim.begin(), dim.end()),
                                              strides(get_strides(dim)),
                                              data(new T[size], std::default_delete<T[]>()) {}

  template<typename...Indices>
  ndarray(const ndarray<T> & ref, size_t d1, Indices...inds) /* TODO: implement me! */{}

  template<typename Scalar>
  ndarray(Scalar)


  virtual ~ndarray() {
  }

  size_t size;
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  std::shared_ptr<T> data;

  template<typename...Indices>
  const T* ref(Indices...inds) const {
    return &data.get()[get_index(inds...)];
  }

  template<typename...Indices>
  T* ref(Indices...inds) {
    return &data.get()[get_index(inds...)];
  }

  void set_zero() {
    std::fill(data, data.get()+size, 0);
  }


  /**
   * TODO:
   *
   * operator() (indices...inds)
   *    - return ndarray with new shape to be shape[sizeof...(indices):] if sizeof...(indices) < shape or 0
   *    - implicit conversion between scalar types and ndarray
   *
   */

  template<typename...Indices>
  ndarray<T> operator() (Indices...inds) {
    ndarray<T> res(*this, inds...);
  };

  template<typename...Indices>
  ndarray<const typename std::remove_const<T>::type> operator() (Indices...inds) const {

  };



private:
  template<typename ...Indices>
  size_t get_index(Indices...inds) const {
#ifndef NDEBUG
    if (sizeof...(Indices) > shape.size())
      throw std::logic_error("wrong dimensions");
#endif

    std::array<size_t, sizeof...(Indices)> ind_arr{{size_t(inds)...}};
#ifndef NDEBUG
    for(size_t i = 0; i < ind_arr.size(); ++i ) {
      if( ind_arr[i] >= shape[i] )
        throw std::logic_error(std::to_string(i) + "-th index is larger than its dimension.");
    }
#endif
    std::transform(ind_arr.begin(), ind_arr.end(), strides.begin(), ind_arr.begin(), std::multiplies<size_t>());
    size_t ind = std::accumulate(ind_arr.begin(), ind_arr.end(), size_t(0), std::plus<size_t>());
    return ind;
  }

  template<size_t D>
  size_t get_size(const std::array<size_t, D> & shape) const {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  }

  template<size_t D>
  std::vector<size_t> get_strides(const std::array<size_t, D> & shape) const {
    std::vector<size_t> str(shape.size());
    str[shape.size() - 1] = 1;
    for (int k = int(shape.size()) - 2; k >= 0; --k)
      str[k] = str[k + 1] * shape[k + 1];
    return str;
  }
};


