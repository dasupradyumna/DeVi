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

#ifndef _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_INDEX_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_INDEX_HH_

#include "../__header_check__"
#include "slice.hh"

namespace devi::core::internal
{
  // Represents a multi-dimensional index into an array or a view
  class index : public base_dimension {
  public:
    //////////////////////////// CONSTRUCTORS ////////////////////////////

    /* Constructs and returns an `index` from a variadic list of integer arguments
     *
     * Errors:
     * 1) `new` can throw an `std::bad_alloc` exception
     * 2) compile-time error if the argument list has more than 10 integers
     */
    template<typename... _Args>
    index(const _Args... args);

    // Static constructor for an `index` from argument `shape` and flat index `i`
    [[nodiscard]] static index from_flat(const shape &shape, const std::size_t i);

    ////////////////////////////// GENERAL ///////////////////////////////

    /* Returns the dot product between current index and argument `data`
     *
     * Precondition: `data` must have same dimensionality as the index
     */
    [[nodiscard]] std::size_t dot(const slice_data &data) const noexcept;

    /* Returns a flat offset obtained by converting current index using argument shape
     *
     * Precondition:
     * index must have same dimensionality as `shape`
     * index must be within bounds for `shape`
     */
    [[nodiscard]] std::size_t flat(const shape &shape) const noexcept;

    /* Transforms current index from shape `src` to shape `dst`
     *
     * Precondition:
     * index must have same dimensionality as `src`
     * index must be within bounds for `src`
     */
    [[nodiscard]] const index &transform(const shape &src, const shape &dst) noexcept;

    // Replace current index with the index calculated from `shape` and flat index `i`
    [[nodiscard]] const index &unflat(const shape &shape, const std::size_t i) noexcept;

    //////////////////////// PRECONDITION CHECKS /////////////////////////

    // Checks if current index is invalid for argument shape
    void throw_if_out_of_bounds_of(const shape &shape) const;

    // Checks if current index has same dimensionality as the argument
    void throw_if_dimensionality_not_equal_to(const shape &other) const;
    void throw_if_dimensionality_not_equal_to(const slice_data &other) const;

  };  // class index

}  // namespace devi::core::internal

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

#include <cassert>

namespace devi::core::internal
{
  //////////////////////////// CONSTRUCTORS ////////////////////////////

  template<typename... _Args>
  index::index(const _Args... args) : base_dimension { args... }
  { }

  inline index index::from_flat(const shape &shape, const std::size_t i)
  {
    index index {};
    return index.unflat(shape, i);
  }

  ////////////////////////////// GENERAL ///////////////////////////////

  inline std::size_t index::dot(const slice_data &data) const noexcept
  {
    std::size_t dot { 0 };
    for (unsigned i { -1U }; ++i < m_size;) dot += p_data[i] * data[i];

    return dot;
  }

  inline std::size_t index::flat(const shape &shape) const noexcept
  {
    std::size_t stride { 1 }, flat { 0 };
    for (auto i { m_size }; i--; stride *= shape[i]) flat += stride * p_data[i];

    return flat;
  }

  inline const index &index::transform(const shape &src, const shape &dst) noexcept
  {
    // Check for identity and skip if satisfied
    if (src == dst) return *this;

    assert(
      (src.size() != dst.size())
      && "Argument shapes `src` and `dst` are not compatible for index transformation");

    return this->unflat(dst, this->flat(src));
  }

  inline const index &index::unflat(const shape &shape, const std::size_t i) noexcept
  {
    m_size = shape.ndims();
    std::size_t flat { i }, stride { 1 };
    for (auto i { m_size }; i--; stride *= shape[i])
      p_data[i] = (flat % (stride * shape[i])) / stride;

    return *this;
  }

  //////////////////////// PRECONDITION CHECKS /////////////////////////

  inline void index::throw_if_out_of_bounds_of(const shape &shape) const
  {
    for (unsigned i { -1U }; ++i < m_size;)
      if (p_data[i] >= shape[i])
        throw std::out_of_range { "Index is out of bounds for the argument `shape`" };
  }

  inline void index::throw_if_dimensionality_not_equal_to(const shape &other) const
  {
    if (m_size != other.ndims())
      throw std::invalid_argument {
        "Index does not have same dimensionality as argument `shape`"
      };
  }

  inline void index::throw_if_dimensionality_not_equal_to(const slice_data &other) const
  {
    if (m_size != other.ndims())
      throw std::invalid_argument {
        "Index does not have same dimensionality as argument `slice_data`"
      };
  }

}  // namespace devi::core::internal

#endif
