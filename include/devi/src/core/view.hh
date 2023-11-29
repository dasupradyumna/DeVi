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

    // Multi-dimensional full indexing using integers
    template<typename... _Indices>
    [[nodiscard]] native_type &operator()(const _Indices... indices);
    template<typename... _Indices>
    [[nodiscard]] native_type operator()(const _Indices... indices) const;

    ////////////////////////////// GETTERS ///////////////////////////////

    /* Return the element value at given flat index
     *
     * Errors:
     * `std::out_of_range` is thrown if argument `i` fails the bounds check
     * */
    [[nodiscard]] native_type &at(const std::size_t i);
    [[nodiscard]] native_type at(const std::size_t i) const;

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
  template<typename... _Indices>
  typename view<_DType>::native_type &view<_DType>::operator()(const _Indices... indices)
  {
    if (sizeof...(indices) != m_shape.ndims())
      throw std::invalid_argument { "Index must have same dimensionality as view shape" };

    return p_iter.flat(index(indices...).transform(m_shape, p_iter.m_shape));
  }

  template<type _DType>
  template<typename... _Indices>
  typename view<_DType>::native_type view<_DType>::operator()(
    const _Indices... indices) const
  {
    return const_cast<view &>(*this)(indices...);
  }

  ////////////////////////////// GETTERS ///////////////////////////////
  // XXX: REFACTOR: implementation exactly same as array

  template<type _DType>
  typename view<_DType>::native_type &view<_DType>::at(const std::size_t i)
  {
    if (i >= m_shape.size()) throw std::out_of_range { "Flat index out of bounds" };

    return (*this)[i];
  }

  template<type _DType>
  typename view<_DType>::native_type view<_DType>::at(const std::size_t i) const
  {
    return const_cast<view &>(*this).at(i);
  }

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
