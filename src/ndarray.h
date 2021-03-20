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
  ndarray(const std::array<size_t, D> &dim) : size_(get_size(dim)), offset_(0),
                                              shape_(dim.begin(), dim.end()),
                                              strides_(get_strides(dim)),
                                              data_(new T[size_], std::default_delete<T[]>()) {}


  template<typename...Indices>
  ndarray(ndarray<T> &ref, size_t d1, Indices...inds)
  /* TODO: implement me!
   * 1. assign ref.data to data
   * 2. compute new offset
   * 3. compute new shape
   * 4. compute new strides
   * */{}

//  template<typename Scalar>
//  ndarray(Scalar scalar) :  {
//
//  }


  virtual ~ndarray() {
  }

  template<typename...Indices>
  const T* ref(Indices...inds) const {
    return &data_.get()[get_index(inds...)];
  }

  template<typename...Indices>
  T* ref(Indices...inds) {
    return &data_.get()[get_index(inds...)];
  }

  void set_zero() {
    std::fill(data_, data_.get() + size_, 0);
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

  // Data accessors

  const std::shared_ptr<T> &data() const {
    return data_;
  }

  std::shared_ptr<T> &data() {
    return data_;
  }

  size_t size() const {
    return size_;
  }

  size_t offset() const {
    return offset_;
  }

  const std::vector<size_t> &shape() const {
    return shape_;
  }

  const std::vector<size_t> &strides() const {
    return strides_;
  }

private:
  size_t size_;
  size_t offset_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  std::shared_ptr<T> data_;

  template<typename ...Indices>
  size_t get_index(Indices...inds) const {
#ifndef NDEBUG
    if (sizeof...(Indices) > shape_.size())
      throw std::logic_error("wrong dimensions");
#endif

    std::array<size_t, sizeof...(Indices)> ind_arr{{size_t(inds)...}};
#ifndef NDEBUG
    for(size_t i = 0; i < ind_arr.size(); ++i ) {
      if (ind_arr[i] >= shape_[i])
        throw std::logic_error(std::to_string(i) + "-th index is larger than its dimension.");
    }
#endif
    std::transform(ind_arr.begin(), ind_arr.end(), strides_.begin(), ind_arr.begin(), std::multiplies<size_t>());
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


