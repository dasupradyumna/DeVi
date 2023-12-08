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

#ifndef _HEADER_GUARD__DEVI_SRC_CORE_VIEW_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_VIEW_HH_

#include "__header_check__"
#include "dimension/index.hh"
#include "types.hh"

namespace devi::core::internal
{
  template<type _DType>
  class array;  // to avoid recursive include error

  template<type _DType>
  class view {
    using native_type = typename native_type<_DType>::type;
    class iterator;

  public:
    // Default destructor
    ~view() noexcept = default;

    ///////////////////////// OPERATOR OVERLOADS /////////////////////////

    // Flat indexing into data window
    [[nodiscard]] native_type &operator[](const std::size_t i) noexcept;
    [[nodiscard]] native_type operator[](const std::size_t i) const noexcept;

    /* Multi-dimensional full indexing using integers
     *
     * Errors:
     * 1) `std::invalid_argument` if the number of `indices` arguments is not equal to
     *    view's dimensionality
     * 2) `std::out_of_range` if the argument index is valid but out of bounds for atleast
     *    one dimension in view's shape
     */
    template<typename... _Indices,
      typename = std::enable_if_t<(std::is_integral_v<_Indices> && ...)>>
    [[nodiscard]] native_type &operator()(const _Indices... indices);
    template<typename... _Indices,
      typename = std::enable_if_t<(std::is_integral_v<_Indices> && ...)>>
    [[nodiscard]] native_type operator()(const _Indices... indices) const;

    ////////////////////////////// GETTERS ///////////////////////////////

    // Returns the dimensionality of the view
    [[nodiscard]] unsigned ndims() const noexcept;

    // Returns the shape of the view
    [[nodiscard]] const class shape &shape() const noexcept;

    // Returns the total size of the view
    [[nodiscard]] std::size_t size() const noexcept;

    // Returns the `devi::core::type` of the view
    [[nodiscard]] enum type type() const noexcept;

  private:
    //////////////////////////// CONSTRUCTORS ////////////////////////////

    // Direct value constructor
    view(native_type *const source, const class shape &shape, const std::size_t start,
      const slice_data &stride);

    ///////////////////////////// ATTRIBUTES /////////////////////////////

    const iterator p_iter;
    class shape m_shape;

    friend class array<_DType>;  // for access to constructor

    ////////////////////////////// ITERATOR //////////////////////////////

    // Internal iterator which stores the memory layout of the view
    class iterator {
    public:
      // Direct value initialization constructor
      iterator(native_type *const source, const class shape &shape,
        const std::size_t start, const slice_data &stride);

      // Returns a flat offset denoted by `index` into the memory managed by the iterator
      native_type &flat(const index &index) const;
      // Returns a flat offset denoted by `i` into the memory managed by the iterator
      native_type &flat(const std::size_t i) const;

    public:
      native_type *p_source;
      class shape m_shape;
      std::size_t m_start;
      slice_data m_stride;

    };  // class iterator

  };  // class view

}  // namespace devi::core::internal

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

namespace devi::core::internal
{
  //////////////////////////// CONSTRUCTORS ////////////////////////////

  template<type _DType>
  view<_DType>::view(native_type *const source, const class shape &shape,
    const std::size_t start, const slice_data &stride)
    : p_iter { source, shape, start, stride }, m_shape { shape }
  { }

  ///////////////////////// OPERATOR OVERLOADS /////////////////////////

  template<type _DType>
  typename view<_DType>::native_type &view<_DType>::operator[](
    const std::size_t i) noexcept
  {
    return p_iter.flat(i);
  }

  template<type _DType>
  typename view<_DType>::native_type view<_DType>::operator[](
    const std::size_t i) const noexcept
  {
    return const_cast<view &>(*this)[i];
  }

  template<type _DType>
  template<typename... _Indices, typename>
  typename view<_DType>::native_type &view<_DType>::operator()(const _Indices... indices)
  {
    index index { indices... };
    index.throw_if_dimensionality_not_equal_to(m_shape);
    index.throw_if_out_of_bounds_of(m_shape);

    return p_iter.flat(index.transform(m_shape, p_iter.m_shape));
  }

  template<type _DType>
  template<typename... _Indices, typename>
  typename view<_DType>::native_type view<_DType>::operator()(
    const _Indices... indices) const
  {
    return const_cast<view &>(*this)(indices...);
  }

  ////////////////////////////// GETTERS ///////////////////////////////
  // XXX: REFACTOR: implementation exactly same as array

  template<type _DType>
  unsigned view<_DType>::ndims() const noexcept
  {
    return m_shape.ndims();
  }

  template<type _DType>
  const shape &view<_DType>::shape() const noexcept
  {
    return m_shape;
  }

  template<type _DType>
  std::size_t view<_DType>::size() const noexcept
  {
    return m_shape.size();
  }

  template<type _DType>
  type view<_DType>::type() const noexcept
  {
    return _DType;
  }

  ////////////////////////////// ITERATOR //////////////////////////////

  template<type _DType>
  view<_DType>::iterator::iterator(native_type *const source, const class shape &shape,
    const std::size_t start, const slice_data &stride)
    : p_source { source }, m_shape { shape }, m_start { start }, m_stride { stride }
  { }

  template<type _DType>
  typename view<_DType>::native_type &view<_DType>::iterator::flat(
    const index &index) const
  {
    index.throw_if_dimensionality_not_equal_to(m_stride);

    return p_source[m_start + index.dot(m_stride)];
  }

  template<type _DType>
  typename view<_DType>::native_type &view<_DType>::iterator::flat(
    const std::size_t i) const
  {
    return this->flat(index::from_flat(m_shape, i));
  }

}  // namespace devi::core::internal

#endif
