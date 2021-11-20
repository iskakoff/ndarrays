#pragma once

#include <algorithm>
#include <memory>
#include <array>
#include <complex>
#include <numeric>
#include <string>
#include <vector>
#include <map>

namespace ndarray {

  template<typename T>
  struct is_complex : std::false_type {
  };
  template<typename T>
  struct is_complex<std::complex<T>> : std::true_type {
  };
  template<typename T>
  using is_scalar = std::integral_constant<bool, std::is_arithmetic<T>::value || is_complex<T>::value>;

  template<typename T>
  struct ndarray {
    static_assert(is_scalar<T>::value, "");

    /**
     * Constructor for initialization from dimensions (allocates memory for attribute data_).
     *
     * @tparam Indices type for indices.
     * @param[in] d1 is first dimension.
     * @param[in] inds are after first dimensions.
     */
    template<typename...Indices>
    ndarray(size_t d1, Indices...inds) : ndarray(
        std::array<size_t, sizeof...(inds) + 1>{{d1, size_t(inds)...}}) {}

    /**
     * Constructor for initialization from array of dimensions (allocates memory for attribute data_).
     *
     * @param[in] shape is array while D is its dimension.
     */
    template<size_t D>
    ndarray(const std::array<size_t, D> &shape) : shape_(shape.begin(), shape.end()),
                                                  strides_(get_strides(shape)),
                                                  size_(get_size(shape)), offset_(0),
                                                  data_(new T[size_], std::default_delete<T[]>()) {
      set_value(0.0);
    }

    ndarray(const std::vector<size_t> &shape) : shape_(shape.begin(), shape.end()),
                                                strides_(get_strides(shape)),
                                                size_(get_size(shape)), offset_(0),
                                                data_(new T[size_], std::default_delete<T[]>()) {
      set_value(0.0);
    }

    /**
     * Constructor for slicing of existing instance.
     *
     * @tparam Indices type for indices.
     * @param[in] ref is existing instance for slicing.
     * @param[in] inds are indices for slicing.
     */
    template<typename T2=typename std::remove_const<T>::type, typename Indtype, typename...Indices>
    ndarray(const ndarray<T2> &ref, Indtype d1, Indices...inds) :
        ndarray(ref, std::array<size_t, sizeof...(inds) + 1ul>{{size_t(d1), size_t(inds)...}}) {}

    /**
     * Constructor for slicing of existing instance.
     *
     * @param[in] ref is existing instance for slicing.
     * @param[in] inds is array contains indices for slicing.
     */
    template<typename T2=typename std::remove_const<T>::type, size_t D>
    ndarray(const ndarray<T2> &ref, const std::array<size_t, D> &inds) :
        shape_(get_shape(ref.shape(), inds)),
        strides_(get_strides(shape_)),
        size_(get_size(shape_)),
        offset_(ref.offset() + get_offset(ref.strides(), inds)),
        data_(ref.data()) {}

    template<typename T2=typename std::remove_const<T>::type>
    ndarray(const ndarray<T2> &rhs) : shape_(rhs.shape()),
                                      strides_(rhs.strides()),
                                      size_(rhs.size()),
                                      offset_(rhs.offset()),
                                      data_(rhs.data()) {}

    template<typename T2=typename std::remove_const<T>::type>
    ndarray(const ndarray<const T2> &rhs) : shape_(rhs.shape()),
                                            strides_(rhs.strides()),
                                            size_(rhs.size()),
                                            offset_(rhs.offset()),
                                            data_(rhs.data()) {}

    /**
     * Conversion into scalar type
     *
     * @tparam Scalar type of LHS argument
     * @return value of zero-dimension tensor
     */
    template<typename Scalar, typename = typename std::enable_if<
        is_scalar<Scalar>::value && std::is_convertible<T, Scalar>::value>::type>
    operator Scalar() const {
#ifndef NDEBUG
      check_zero_dimension();
#endif
      return Scalar(data_.get()[offset_]);
    }

    /**
     * Obtain reference to the scalar value of the zero-dimension array
     *
     * @return reference to the scalar
     */
    operator T &() {
#ifndef NDEBUG
      check_zero_dimension();
#endif
      return data_.get()[offset_];
    }

    operator const T &() const {
#ifndef NDEBUG
      check_zero_dimension();
#endif
      return data_.get()[offset_];
    }

    /**
     * Assign scalar value to zero-dimension tensor
     *
     * @tparam Scalar - scalar type
     * @param rhs     - value of a scalar
     * @return current tensor with updated value
     */
    template<typename Scalar, typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
    ndarray<T> &operator=(const Scalar rhs) {
#ifndef NDEBUG
      check_zero_dimension();
#endif
      data_.get()[offset_] = T(rhs);
      return *this;
    };


    /**
     * Deep copy of array
     *
     * @return new array that is a full copy of current array
     */
    ndarray<typename std::remove_const<T>::type> copy() const {
      ndarray<typename std::remove_const<T>::type> ret(shape_);
      std::copy(data_.get(), data_.get() + size_, ret.data().get());
      return ret;
    }

    virtual ~ndarray() {
    }

    template<typename...Indices>
    const T *ref(Indices...inds) const {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      check_dimensions(num_of_inds);
#endif
      return &data_.get()[offset_ + get_index(inds...)];
    }

    template<typename...Indices>
    T *ref(Indices...inds) {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      check_dimensions(num_of_inds);
#endif
      return &data_.get()[offset_ + get_index(inds...)];
    }

    // TODO (Aleks): 1. write comments
    // TODO (Aleks): 2. return ndarray with new shape to be shape[sizeof...(indices):] if sizeof...(indices) < shape or 0
    // TODO (Aleks): 3. implicit conversion between scalar types and ndarray
    template<typename...Indices>
    ndarray<T> operator()(Indices...inds) {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      check_dimensions(shape_, num_of_inds);
#endif
      ndarray<T> res(*this, inds...);
      return res;
    };

    template<typename...Indices>
    ndarray<const typename std::remove_const<T>::type> operator()(Indices...inds) const {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      check_dimensions(shape_, num_of_inds);
#endif
      ndarray<const typename std::remove_const<T>::type> res(*this, inds...);
      return res;
    };

    // ToDo make test
    template<typename T2>
    typename std::enable_if<is_scalar<T2>::value && std::is_convertible<T2, T>::value>::type
    set_value(T2 value) {
      std::fill(data_.get() + offset_, data_.get() + offset_ + size_, T(value));
    }

    void set_zero() {
      set_value(0);
    }

    ndarray<T> reshape(const std::vector<size_t> &shape) const {
      ndarray<T> result(*this);
      return result.inplace_reshape(shape);
    }

    ndarray<T> inplace_reshape(const std::vector<size_t> &shape) {
#ifndef NDEBUG
      if (get_size(shape) != size_)
        throw std::logic_error("new shape is not consistent with old one");
#endif
      shape_ = shape;
      strides_ = get_strides(shape);
      return *this;
    }

    ndarray<T> transpose(const std::string &string_pattern) const {
      size_t find = string_pattern.find("->");
      if (find == std::string::npos) {
        throw std::runtime_error("Incorrect transpose pattern.");
      }
      std::string from = string_pattern.substr(0, find);
      std::string to = string_pattern.substr(find + 2, string_pattern.size() - 1);
      if (from.length() != to.length()) {
        throw std::runtime_error("Transpose source and target indices have different size.");
      }
      if (from.length() != dim()) {
        throw std::runtime_error("Number of transpose indices and array dimension are different size.");
      }


      // TODO (Sergei): 1. check that all indices of input are in the output indices - throw exception if not
      // TODO (Sergei): 2. remove leading and trailing spaces from indices - possible option do regular expression
      // TODO (Sergei): 3. check that all indices are Latin letters - possible option do regular expression
      // TODO (Sergei): 4. Add tests for 1-3.


      std::map<char, size_t> index_map;
      for (size_t i = 0; i < to.length(); ++i) {
        index_map[to[i]] = i;
      }
      std::vector<size_t> pattern(to.length());
      for (size_t j = 0; j < from.length(); ++j) {
        pattern[j] = index_map[from[j]];
      }
      return transpose_inner(pattern);
    }

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

    size_t dim() const {
      return shape_.size();
    }

  private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    size_t offset_;
    std::shared_ptr<T> data_;

    template<typename ...Indices>
    size_t get_index(Indices...inds) const {
#ifndef NDEBUG
      if (sizeof...(Indices) > shape_.size())
        throw std::logic_error("wrong dimensions");
#endif

      std::array<size_t, sizeof...(Indices)> ind_arr{{size_t(inds)...}};
#ifndef NDEBUG
      for (size_t i = 0; i < ind_arr.size(); ++i) {
        if (ind_arr[i] >= shape_[i])
          throw std::logic_error(std::to_string(i) + "-th index is larger than its dimension.");
      }
#endif
      std::transform(ind_arr.begin(), ind_arr.end(), strides_.begin(), ind_arr.begin(),
                     std::multiplies<size_t>());
      size_t ind = std::accumulate(ind_arr.begin(), ind_arr.end(), size_t(0), std::plus<size_t>());
      return ind;
    }

    template<typename Container, typename Container2>
    size_t get_offset(const Container &strides, const Container2 &inds) const {
      return std::inner_product(inds.begin(), inds.end(), strides.begin(), 0ul);
    }

    template<typename Container>
    size_t get_size(const Container &shape) const {
      return std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>());
    }

    template<size_t D>
    std::vector<size_t>
    get_shape(const std::vector<size_t> &old_shape, const std::array<size_t, D> &inds) const {
#ifndef NDEBUG
      size_t num_of_inds = D;
      check_dimensions(old_shape, num_of_inds);
      for (size_t i = 0; i < inds.size(); ++i) {
        if (inds[i] >= old_shape[i])
          throw std::logic_error(std::to_string(i) + "-th index is larger than its dimension.");
      }
#endif
      std::vector<size_t> shape(old_shape.size() - D, 0);
      std::copy(old_shape.data() + D, old_shape.data() + old_shape.size(), shape.data());
      return shape;
    }

    template<typename Container>
    std::vector<size_t> get_strides(const Container &shape) const {
      std::vector<size_t> str(shape.size());
      if (shape.size() == 0)
        return str;
      str[shape.size() - 1] = 1;
      for (int k = int(shape.size()) - 2; k >= 0; --k)
        str[k] = str[k + 1] * shape.data()[k + 1];
      return str;
    }

    /**
     * Check that array is zero-dimension. Throw an exception if it's not.
     */
    void check_zero_dimension() const {
      if (shape_.size() != 0) {
        throw std::runtime_error("Array is not directly castable to a scalar. Array's dimension is " +
                                 std::to_string(shape_.size()));
      }
    }

    void check_dimensions(const std::vector<size_t> &shape, size_t num_of_inds) const {
      if (num_of_inds > shape.size()) {
        throw std::runtime_error("Number of indices (" +
                                 std::to_string(num_of_inds) + ") is larger than array's dimension (" +
                                 std::to_string(shape.size()) + ")");
      }
    }

    ndarray<T> transpose_inner(const std::vector<size_t> &pattern) const {
      std::vector<size_t> shape(shape_.size());
      for (size_t i(0); i < shape_.size(); ++i) {
        shape[pattern[i]] = shape_[i];
      }
      ndarray<T> result(shape);
      std::vector<size_t> indices(dim(), 0);
      for (size_t i(0); i < size_; ++i) {
        size_t res = i;
        for (size_t ind(0); ind < dim(); ++ind) {
          indices[pattern[dim() - ind - 1]] = res % shape_[dim() - ind - 1];
          res /= shape_[dim() - ind - 1];
        }
        std::transform(indices.begin(), indices.end(), result.strides_.begin(), indices.begin(),
                       std::multiplies<size_t>());
        size_t ind = std::accumulate(indices.begin(), indices.end(), size_t(0), std::plus<size_t>());
        result.data_.get()[ind] = data_.get()[offset_ + i];
      }
      return result;
    }
  };
}
