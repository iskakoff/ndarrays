/*
 * Copyright (c) 2020 University of Michigan.
 *
 */

#ifndef ALPS_TENSOR_H
#define ALPS_TENSOR_H

#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>
#include <numeric>

#include <mapbox/variant.hpp>

#include <index_sequence.h>

namespace alps {

  /**
   * Check that all values in pack are true
   *
   * @tparam IndicesTypes - types of indices template parameter pack
   */
  template<bool...>
  struct bool_pack;
  template<bool... v>
  using all_true = std::is_same<bool_pack<true, v...>, bool_pack<v..., true>>;

  /**
   * Raw data storage class
   * In order to be able to cast data from one type to another we store raw data as unsigned char
   */
  class storage {

  public:

//    storage() : _size(0) {}

    /**
     * Create storage of size `sizeof(T)*size` bytes
     *
     * @param size - size of the storage in bytes
     */
    explicit storage(int64_t size) : _size(size), _total_size(size), _offset(0),
                                     _data(std::shared_ptr<unsigned char>(new unsigned char[size],
                                                                          std::default_delete<unsigned char[]>())) {}

    /**
     * Copy constructor
     */
    storage(const storage &rhs) : _size(rhs._size), _total_size(rhs._total_size), _offset(rhs._offset),
                                  _data(rhs._data) {
      std::cout << "cpy storage" << std::endl;
    }

    /**
     * Move constructor
     *
     * we need to allocate new memory and perform deep copy of data.
     */
    storage(storage &&rhs) : _size(rhs._size), _total_size(rhs._size), _offset(0),
                             _data(std::shared_ptr<unsigned char>(new unsigned char[rhs._size],
                                                                  std::default_delete<unsigned char[]>())) {
      std::copy_n(rhs._data.get() + rhs._offset, _size, _data.get());
      std::cout << "mv storage" << std::endl;
    }

    storage &operator=(storage &&rhs) = delete;

    storage &operator=(const storage &rhs) = delete;

    /**
     * Create view/copy storage of the `rhs`
     *
     * @param rhs    - storage to be viewed or copied
     * @param offset - offset from the beggining of the rhs
     * @param copy   - create deep copy or view
     */
    storage(const storage &rhs, int64_t size, int64_t offset, bool copy = false) : _size(size),
                                                                                   _total_size(0),
                                                                                   _offset(
                                                                                       rhs._offset + offset) {
      if (!copy) {
        _data = rhs._data;
        _total_size = rhs._total_size;
      } else {
        _data = std::shared_ptr<unsigned char>(new unsigned char[_size],
                                               std::default_delete<unsigned char[]>());
        std::copy_n(rhs._data.get() + rhs._offset, _size, _data.get());
        _total_size = _size;
      }
    }

    /**
     * Get pointer to the data casted to the specific type
     *
     * @tparam T - type of the target array
     * @return pointer to the first element of the storage
     */
    template<typename T>
    const T *data() const { return reinterpret_cast<const T *>(_data.get() + _offset); }

    template<typename T>
    T *data() { return reinterpret_cast<T *>(_data.get() + _offset); }

    int64_t total_size() const {
      return _total_size;
    }

    int64_t size() const {
      return _size;
    }

    int64_t offset() const {
      return _offset;
    }

  private:
    // size of the current storage
    int64_t _size;
    // size of the total allocation
    int64_t _total_size;
    // offset from the origin
    int64_t _offset;
    // reference to the original data
    std::shared_ptr<unsigned char> _data;
  };

  template<typename T>
  class tensor;

  struct tensor_index {

    template<typename...Indices>
    tensor_index(int64_t x, Indices...xxx) : tensor_index(
        std::array<int64_t, sizeof...(xxx) + 1>{x, int64_t(xxx)...}) {}

    template<size_t D>
    tensor_index(const std::array<int64_t, D> &xxx) : indices(xxx.begin(), xxx.end()) {}

    std::vector<int64_t> indices;
  };

  template<typename T>
  using value_ref = mapbox::util::variant<std::reference_wrapper<tensor<T>>, std::reference_wrapper<T> >;

  template<typename T>
  struct value_type {

    value_type(value_ref<T> &ref) : ref(ref) {}

    operator T() const {
      return ref.template get<T>();
    }

    template<typename T2>
    value_type &operator=(const T2 &t) {
      static_assert(std::is_convertible<T2, T>::value, "Wrong types");
      ref.template get<T>() = T(t);
      return *this;
    }

    value_ref<T> &ref;
  };

  template<typename T>
  class tensor {
  public:

    explicit tensor(int64_t size1) : tensor<T>(std::array<int64_t, 1>{{size1}}) {
      std::cout << "const 1" << std::endl;
    }

    tensor(const tensor<T> &rhs) : _dim(rhs._dim), _shape(rhs._strides), _strides(rhs._strides),
                                   _storage(rhs._storage) {
      std::cout << "copy\n";
    }

    tensor(tensor<T> &&rhs) = default;
    //: _dim(rhs._dim), _shape(rhs._strides), _strides(rhs._strides), _storage(rhs._storage) {
    // std::cout<<"move\n";
    // }

    tensor<T> &operator=(const tensor<T> &rhs) {
      std::cout << "copy\n";
      return *this;
    }

    tensor<T> &operator=(tensor<T> &&rhs) {
      std::cout << "move\n";
      return *this;
    }


    template<typename...Indices>
    explicit tensor(
        typename std::enable_if<all_true<std::is_convertible<Indices, std::int64_t>::value...>::value, int64_t>::type size1,
        Indices...sizes) : tensor<T>(
        std::array<int64_t, sizeof...(Indices) + 1>{{size1, int64_t(sizes)...}}) {
      std::cout << "const 2" << std::endl;
    }

    template<size_t D>
    tensor(const std::array<int64_t, D> &shape) : _dim(D), _shape(shape.begin(), shape.end()),
                                                  _strides(get_strides(shape)),
                                                  _storage(sizeof(T) * size(shape)) {
      std::cout << "const 3" << std::endl;
    }


    template<typename...Indices>
    tensor(const storage &rhs_storage, int64_t offset,
           Indices ...dims) : tensor(rhs_storage, offset,
                                     std::array<int64_t, sizeof...(Indices)>{{int64_t(dims)...}}) {
      std::cout << "const 4" << std::endl;
    }

    template<size_t D>
    tensor(const storage &rhs_storage, int64_t offset,
           const std::array<int64_t, D> &new_shape) : _dim(D), _shape(new_shape.begin(), new_shape.end()),
                                                      _strides(get_strides(new_shape)),
                                                      _storage(rhs_storage, size(new_shape) * sizeof(T),
                                                               offset * sizeof(T)) {
      std::cout << "const 5" << std::endl;
    }

    tensor(const storage &rhs_storage, int64_t offset,
           const std::vector<int64_t> &new_shape) : _dim(new_shape.size()), _shape(new_shape),
                                                    _strides(get_strides(new_shape)),
                                                    _storage(rhs_storage, size(new_shape) * sizeof(T),
                                                             offset * sizeof(T)) {
      std::cout << "const 6" << std::endl;
    }


    /**
     * @return current dimension of the tensor
     */
    int64_t dim() const {
      return _dim;
    }

    /**
     * @return number of elements in the tesnor
     */
    size_t size() const {
      return _storage.size() / sizeof(T);
    }

    /**
     * @return shape of the tensors
     */
    const std::vector<int64_t> &shape() const {
      return _shape;
    }

    /**
     * @return tensor's strides
     */
    const std::vector<int64_t> &strides() const {
      return _strides;
    }

    /**
      * Get reference to the data point at the (t1, indices...) point
      */
    template<typename ...IndexTypes>
    T &operator()(size_t t1, IndexTypes ... indices) {
      if ((sizeof...(IndexTypes)) != _dim - 1) {
        throw std::logic_error("");
      }

      size_t idx = index(t1, size_t(indices)...);
      assert(idx * sizeof(T) < _storage.size());
      return _storage.data<T>()[idx];
    }

    tensor<T> operator[](const tensor_index &index) {
      size_t new_dim = index.indices.size();
      if (new_dim == _dim) {
        throw std::logic_error("");
      }
      size_t ind = std::inner_product(index.indices.begin(), index.indices.end(), _strides.begin(), 0);
      std::vector<int64_t> new_shape;
      new_shape.insert(new_shape.end(), _shape.begin() + new_dim, _shape.end());
      return tensor<T>(_storage, ind, new_shape);
    }

    /// return index in the raw buffer for specified indices
    template<typename ...Indices>
    inline size_t index(const Indices &...indices) const {
      return size_t(index_impl(make_index_sequence<sizeof...(Indices)>(), size_t(indices)...));
    }

  private:

    int64_t _dim;
    std::vector<int64_t> _shape;
    std::vector<int64_t> _strides;

    storage _storage;


    /**
     * compute offset multiplier for each dimension
     */
    template<typename C>
    std::vector<int64_t> get_strides(const C &shape) {
      std::vector<int64_t> strides(shape.size(), 0);
      strides[shape.size() - 1] = 1;
      for (int k = int(shape.size()) - 2; k >= 0; --k)
        strides[k] = strides[k + 1] * shape[k + 1];
      return strides;
    }

    /// compte size for the specific dimensions
    template<typename C>
    size_t size(const C &shape) {
      return std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
    }

    /**
      * Internal implementation of indexing
      */
    template<size_t... I, typename ...Indices>
    inline size_t index_impl(index_sequence<I...>, const Indices &... indices) const {
      std::array<size_t, sizeof...(Indices)> a{{indices * _strides[I] ...}};
      return std::accumulate(a.begin(), a.end(), size_t(0), std::plus<size_t>());
    }
  };
}


#endif //ALPS_TENSOR_H
