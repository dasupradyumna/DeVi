// DeVi: C++17 library for Computer Vision and Deep Learning
// Copyright (C) 2023 Dasu Pradyumna dasupradyumna@gmail.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef _HEADER_GUARD__DEVI_SRC_CORE_ARRAY_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_ARRAY_HH_

#include "__header_check__"
#include "view.hh"

namespace devi::core::internal
{
  // Data-owning multi-dimensional array class
  template<type _DType>
  class array {
    using native_type = typename native_type<_DType>::type;

  public:
    //////////////////////////// CONSTRUCTORS ////////////////////////////

    /* Constructs a zero-initialized `array` with its shape specified by the argument `s`
     *
     * Errors:
     * `new` can throw an `std::bad_alloc` exception
     */
    array(const class shape &s);
    array(class shape &&s);

    /* Constructs an `array` with every element equal to `fill` and its shape specified by
     * the argument `s`
     *
     * Errors:
     * `new` can throw an `std::bad_alloc` exception
     */
    array(const class shape &s, const native_type fill);
    array(class shape &&s, const native_type fill);

    // Default destructor
    ~array() noexcept = default;

    //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

    // Copy constructor
    array(const array &copy);

    // Move constructor
    array(array &&move) noexcept;

    // Copy-and-Swap idiom for assignment operator
    array &operator=(array rhs) noexcept;

    ///////////////////////// OPERATOR OVERLOADS /////////////////////////

    // Equality operator overload
    [[nodiscard]] bool operator==(const array &other) const noexcept;
    template<enum type _Other>
    [[nodiscard]] bool operator==(const array<_Other> &other) const noexcept;

    // Inequality operator overload
    [[nodiscard]] bool operator!=(const array &other) const noexcept;
    template<enum type _Other>
    [[nodiscard]] bool operator!=(const array<_Other> &other) const noexcept;

    // XXX: remove this operator; currently only used for testing
    // Flat indexing into owned data
    [[nodiscard]] native_type &operator[](const std::size_t i) noexcept;
    [[nodiscard]] native_type operator[](const std::size_t i) const noexcept;

    /* Multi-dimensional full indexing using integers
     *
     * Errors:
     * 1) `std::invalid_argument` if the number of `indices` arguments is not equal to
     *    array's dimensionality
     * 2) `std::out_of_range` if the argument index is valid but out of bounds for atleast
     *    one dimension in array's shape
     */
    template<typename... _Indices,
      typename = std::enable_if_t<(std::is_integral_v<_Indices> && ...)>>
    [[nodiscard]] native_type &operator()(const _Indices... indices);
    template<typename... _Indices,
      typename = std::enable_if_t<(std::is_integral_v<_Indices> && ...)>>
    [[nodiscard]] native_type operator()(const _Indices... indices) const;

    // Multi-dimensional partial indexing using integers and slices
    template<typename... _Slices, typename = std::enable_if_t<is_valid_slice<_Slices...>>>
    [[nodiscard]] view<_DType> operator()(const _Slices &...slices);
    // TODO: implement `const_view` class for a non-mutable window into memory

  public:
    ////////////////////////////// GETTERS ///////////////////////////////

    // Returns the dimensionality of the array
    [[nodiscard]] unsigned ndims() const noexcept;

    // Returns the shape of the array
    [[nodiscard]] const class shape &shape() const noexcept;

    // Returns the total size of the array
    [[nodiscard]] std::size_t size() const noexcept;

    // Returns the `devi::core::type` of the array
    [[nodiscard]] enum type type() const noexcept;

    ////////////////////////////// CREATION //////////////////////////////

    // Returns a element-wise type-casted copy of the current array
    template<enum type _AsType>
    [[nodiscard]] array<_AsType> astype() const;

    // Returns a copy of the current array
    [[nodiscard]] array copy() const;

  public:
    ////////////////////////////// MUTATION //////////////////////////////

    // Sets every element in the array to `val`
    void fill(const native_type val) noexcept;

    // Flattens the current array to a single dimension
    void flatten();

    // Reshapes the current array
    template<typename... _Args>
    void reshape(const _Args... args);
    void reshape(const class shape &s);
    void reshape(class shape &&s) noexcept;

    // Squeeze the array shape to remove all unit dimensions
    void squeeze() noexcept;

    // Swap state with existing array
    void swap(array &b) noexcept;
    // Swap state with temporary array
    void swap(array &&b) noexcept;

  private:
    ///////////////////////////// ATTRIBUTES /////////////////////////////

    std::unique_ptr<native_type[]> p_data;
    class shape m_shape;

    ////////////////////////////// FRIENDS ///////////////////////////////

    template<enum type _Other>
    friend class array;  // for accessing private members of arrays of all other types

    friend class view<_DType>;

  };  // class array

  // type aliases for `array`
  // TODO: move this into `devi/core` and make it a union of array and view
  using bool8   = array<type::bool8>;
  using int8    = array<type::int8>;
  using int16   = array<type::int16>;
  using int32   = array<type::int32>;
  using int64   = array<type::int64>;
  using uint8   = array<type::uint8>;
  using uint16  = array<type::uint16>;
  using uint32  = array<type::uint32>;
  using uint64  = array<type::uint64>;
  using float32 = array<type::float32>;
  using float64 = array<type::float64>;

}  // namespace devi::core::internal

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

#include "dimension/index.hh"

namespace devi::core::internal
{
  //////////////////////////// CONSTRUCTORS ////////////////////////////

  template<type _DType>
  array<_DType>::array(const class shape &s)
    : p_data { new native_type[s.size()] {} }, m_shape { s }
  { }

  template<type _DType>
  array<_DType>::array(class shape &&s)
    : p_data { new native_type[s.size()] {} }, m_shape { std::move(s) }
  { }

  template<type _DType>
  array<_DType>::array(const class shape &s, const native_type fill) : array { s }
  {
    this->fill(fill);
  }

  template<type _DType>
  array<_DType>::array(class shape &&s, const native_type fill) : array { std::move(s) }
  {
    this->fill(fill);
  }

  //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

  template<type _DType>
  array<_DType>::array(const array &copy) : array { copy.m_shape }
  {
    std::copy_n(copy.p_data.get(), m_shape.size(), p_data.get());
  }

  template<type _DType>
  array<_DType>::array(array &&move) noexcept
    : p_data { std::move(move.p_data) }, m_shape { std::move(move.m_shape) }
  { }

  template<type _DType>
  array<_DType> &array<_DType>::operator=(array rhs) noexcept
  {
    this->swap(rhs);
    return *this;
  }

  ///////////////////////// OPERATOR OVERLOADS /////////////////////////

  template<type _DType>
  bool array<_DType>::operator==(const array &other) const noexcept
  {
    return m_shape == other.m_shape
        && std::equal(p_data.get(), p_data.get() + m_shape.size(), other.p_data.get());
  }

  template< type _DType>
  template<type _Other>
  bool array<_DType>::operator==(const array<_Other> &other) const noexcept
  {
    return false;
  }

  template<type _DType>
  bool array<_DType>::operator!=(const array &other) const noexcept
  {
    return !(*this == other);
  }

  template< type _DType>
  template<type _Other>
  bool array<_DType>::operator!=(const array<_Other> &other) const noexcept
  {
    return true;
  }

  template<type _DType>
  typename array<_DType>::native_type &array<_DType>::operator[](
    const std::size_t i) noexcept
  {
    return p_data[i];
  }

  template<type _DType>
  typename array<_DType>::native_type array<_DType>::operator[](
    const std::size_t i) const noexcept
  {
    return const_cast<array &>(*this)[i];
  }

  template<type _DType>
  template<typename... _Indices, typename>
  typename array<_DType>::native_type &array<_DType>::operator()(
    const _Indices... indices)
  {
    index index { indices... };
    index.throw_if_dimensionality_not_equal_to(m_shape);
    index.throw_if_out_of_bounds_of(m_shape);

    return p_data[index.flat(m_shape)];
  }

  template<type _DType>
  template<typename... _Indices, typename>
  typename array<_DType>::native_type array<_DType>::operator()(
    const _Indices... indices) const
  {
    return const_cast<array &>(*this)(indices...);
  }

  namespace  // for internal linkage
  {
    // Calculate view specifications from slice and array shape
    inline void slice_to_view(const slice &slice, const std::size_t dim,
      std::size_t &v_begin, std::size_t &v_dim, std::size_t &v_stride)
    {
      if (slice.m_end > dim || slice.m_begin >= dim)
        throw std::out_of_range { "Slicing: given slice out of bounds" };

      v_begin += v_stride * slice.m_begin;
      // if `slice.m_end` is zero, replace with current `dim`
      v_dim = (slice.m_end + (slice.m_end == 0) * dim) - slice.m_begin;
      // v_dim = ceil(v_dim / slice.m_stride)
      v_dim = v_dim / slice.m_stride + (v_dim % slice.m_stride > 0);
      v_stride *= slice.m_stride;
    }

    // Calculate view specifications from index and array shape
    inline void slice_to_view(const std::size_t idx, const std::size_t dim,
      std::size_t &v_begin, std::size_t &v_dim, std::size_t &v_stride)
    {
      if (idx >= dim) throw std::out_of_range { "Slicing: given index out of bounds" };

      v_begin += v_stride * idx;
      v_dim = v_stride = 0;
    }
  }

  template<type _DType>
  template<typename... _Slices, typename>
  view<_DType> array<_DType>::operator()(const _Slices &...slices)
  {
    if (sizeof...(slices) > m_shape.ndims())
      throw std::invalid_argument {
        "Slice must have atmost the same dimensionality as array shape"
      };

    unsigned i { 0 };
    std::size_t v_begin { 0 };
    // Both shape and stride are pre-filled with default values
    auto v_shape { m_shape };                           // Shape of the resulting view
    auto v_stride { slice_data::get_stride(m_shape) };  // Stride of the resulting view

    // Handle all the slices/indices specified by the user
    ((slice_to_view(slices, m_shape[i], v_begin, v_shape[i], v_stride[i]), ++i), ...);

    // Remove all the zero dimensions (the dimensions specified by an index)
    v_shape.remove_zeros();
    v_stride.remove_zeros();

    return { p_data.get(), std::move(v_shape), v_begin, std::move(v_stride) };
  }

  ////////////////////////////// GETTERS ///////////////////////////////

  template<type _DType>
  unsigned array<_DType>::ndims() const noexcept
  {
    return m_shape.ndims();
  }

  template<type _DType>
  const shape &array<_DType>::shape() const noexcept
  {
    return m_shape;
  }

  template<type _DType>
  std::size_t array<_DType>::size() const noexcept
  {
    return m_shape.size();
  }

  template<type _DType>
  type array<_DType>::type() const noexcept
  {
    return _DType;
  }

  ////////////////////////////// CREATION //////////////////////////////

  template<type _DType>
  template<type _AsType>
  array<_AsType> array<_DType>::astype() const
  {
    array<_AsType> ret { m_shape };
    std::transform(p_data.get(), p_data.get() + m_shape.size(), ret.p_data.get(),
      [](const native_type value) {
        return static_cast<typename core::internal::native_type<_AsType>::type>(value);
      });

    return ret;
  }

  template<type _DType>
  array<_DType> array<_DType>::copy() const
  {
    return array { *this };
  }

  ////////////////////////////// MUTATION //////////////////////////////

  template<type _DType>
  void array<_DType>::fill(const native_type val) noexcept
  {
    std::fill_n(p_data.get(), m_shape.size(), val);
  }

  template<type _DType>
  void array<_DType>::flatten()
  {
    this->reshape(m_shape.size());
  }

  template<type _DType>
  template<typename... _Args>
  void array<_DType>::reshape(const _Args... args)
  {
    this->reshape(core::internal::shape { args... });
  }

  template<type _DType>
  void array<_DType>::reshape(const class shape &s)
  {
    m_shape = s;
  }

  template<type _DType>
  void array<_DType>::reshape(class shape &&s) noexcept
  {
    m_shape = std::move(s);
  }

  template<type _DType>
  void array<_DType>::squeeze() noexcept
  {
    m_shape.squeeze();
  }

  template<type _DType>
  void array<_DType>::swap(array &b) noexcept
  {
    std::swap(p_data, b.p_data);
    std::swap(m_shape, b.m_shape);
  }

  template<type _DType>
  void array<_DType>::swap(array &&b) noexcept
  {
    this->swap(b);
  }

}  // namespace devi::core::internal

#endif
