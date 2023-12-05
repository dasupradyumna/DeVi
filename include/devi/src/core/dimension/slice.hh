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

#ifndef _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_SLICE_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_SLICE_HH_

#include "shape.hh"

namespace devi::core::internal
{
  // Represents a one-dimensional slice object
  struct slice {
    std::size_t m_begin, m_end, m_stride;

    // Direct value initialization constructor
    slice(const std::size_t begin = 0, const std::size_t end = 0,
      const std::size_t stride = 1);
  };

  // Data structure for storing multi-dimensional slice information
  // CHECK: can this be removed? is it same as `base_dimension`?
  class slice_data : public base_dimension {
  public:
    //////////////////////////// CONSTRUCTORS ////////////////////////////

    /* Constructs and returns a `slice_data` from a variadic list of integer arguments
     *
     * Errors:
     * 1) `new` can throw an `std::bad_alloc` exception
     * 2) compile-time error if the argument list has more than 10 integers
     */
    template<typename... _Args>
    slice_data(const _Args... args);

    // Static constructor for a `slice_data` stride from argument `shape`
    static slice_data get_stride(const shape &shape);

    ///////////////////////// OPERATOR OVERLOADS /////////////////////////

    // Inherited equality and inequality operators
    using base_dimension::operator==;
    using base_dimension::operator!=;

    // Returns a non-const reference to value at specified index
    [[nodiscard]] std::size_t &operator[](const unsigned index) noexcept;
    [[nodiscard]] std::size_t operator[](const unsigned index) const noexcept;

    ////////////////////////////// GENERAL ///////////////////////////////

    // Getter method for `m_size` attribute
    using base_dimension::ndims;

    // Remove all zero values from current data
    using base_dimension::remove_zeros;

  };  // class slice_data

  namespace  // for internal linkage
  {
    template<typename...>
    struct slice_counter;

    template<>
    struct slice_counter<> {
      static constexpr unsigned value { 0 };
    };

    template<typename _Type>
    struct slice_counter<_Type> {
      static constexpr unsigned value { std::is_same_v<_Type, slice> };
    };

    template<typename _Type1, typename... _TypesN>
    struct slice_counter<_Type1, _TypesN...> {
      static constexpr unsigned value { slice_counter<_Type1>::value
                                        + slice_counter<_TypesN...>::value };
    };
  }

  // Compile-time type checker for a parameter pack of slices
  template<typename... _Slices>
  static constexpr bool is_valid_slice {
    (... && (std::is_integral_v<_Slices> || std::is_same_v<_Slices, slice>))
    && slice_counter<_Slices...>::value > 0
  };

}  // namespace devi::core::internal

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

////////////////////////////////// SLICE_DATA //////////////////////////////////

namespace devi::core::internal
{
  inline slice::slice(
    const std::size_t begin, const std::size_t end, const std::size_t stride)
    : m_begin { begin }, m_end { end }, m_stride { stride }
  {
    if (end && begin >= end)
      // `begin` must be less than `end` when `end` is non-zero
      throw std::invalid_argument {
        "`end` must be atleast one more than `begin` in a slice"
      };
    else if (stride == 0)
      throw std::invalid_argument { "`stride` must be non-zero in a slice" };
  }

}  // namespace devi::core::internal

////////////////////////////////// SLICE_DATA //////////////////////////////////
namespace devi::core::internal
{
  //////////////////////////// CONSTRUCTORS ////////////////////////////

  template<typename... _Args>
  slice_data::slice_data(const _Args... args) : base_dimension { args... }
  { }

  inline slice_data slice_data::get_stride(const shape &shape)
  {
    slice_data strides {};
    strides.m_size = shape.ndims();
    std::size_t stride { 1 };
    for (unsigned i { strides.m_size }; i--; stride *= shape[i]) strides[i] = stride;
    return strides;
  }

  ///////////////////////// OPERATOR OVERLOADS /////////////////////////

  inline std::size_t &slice_data::operator[](const unsigned index) noexcept
  {
    return p_data[index];
  }

  inline std::size_t slice_data::operator[](const unsigned index) const noexcept
  {
    return p_data[index];
  }

}  // namespace devi::core::internal

#endif
