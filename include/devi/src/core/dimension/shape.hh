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

#ifndef _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_SHAPE_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_SHAPE_HH_

#include "../__header_check__"
#include "base.hh"

namespace devi::core::internal
{
  // Represents the shape of an array and its dimensionality
  class shape : public base_dimension {
  public:
    //////////////////////////// CONSTRUCTORS ////////////////////////////

    /* Constructs and returns a `shape` from a variadic list of integer arguments
     *
     * Errors:
     * 1) `new` can throw an `std::bad_alloc` exception
     * 2) compile-time error if the argument list has more than 10 integers
     */
    template<typename... _Args>
    shape(const _Args... args);

    //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

    // Copy constructor
    shape(const shape &copy);

    // Move constructor
    shape(shape &&move) noexcept;

    // Copy-and-Swap idiom for assignment operator
    shape &operator=(shape rhs) noexcept;

    ///////////////////////// OPERATOR OVERLOADS /////////////////////////

    // Inherited equality and inequality operators
    using base_dimension::operator==;
    using base_dimension::operator!=;

    // Returns value at specified index
    [[nodiscard]] std::size_t &operator[](const unsigned idx) noexcept;
    [[nodiscard]] std::size_t operator[](const unsigned idx) const noexcept;

    // Overloading `std::ostream` for easy printing via `std::cout`
    friend std::ostream &operator<<(std::ostream &out, const shape &s);

    ////////////////////////////// GENERAL ///////////////////////////////

    // Returns the dimensionality of current shape
    using base_dimension::ndims;

    // Remove all zero values from current data
    using base_dimension::remove_zeros;

    // Returns the total size held by current shape
    [[nodiscard]] std::size_t size() const noexcept;

    // Squeeze current shape to remove all unit dimensions
    void squeeze() noexcept;

    // Returns the string representation of current shape
    [[nodiscard]] std::string str() const;

  };  // class shape

}  // namespace devi::core::internal

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

#include <numeric>

namespace devi::core::internal
{
  //////////////////////////// CONSTRUCTORS ////////////////////////////

  template<typename... _Args>
  shape::shape(const _Args... args) : base_dimension { args... }
  {
    static_assert(sizeof...(args) > 0, "`devi::core::shape` cannot be empty");
  }

  //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

  inline shape::shape(const shape &copy) : base_dimension { copy } { }

  inline shape::shape(shape &&move) noexcept : base_dimension { std::move(move) } { }

  inline shape &shape::operator=(shape rhs) noexcept
  {
    this->swap(rhs);
    return *this;
  }

  ///////////////////////// OPERATOR OVERLOADS /////////////////////////

  inline std::size_t &shape::operator[](const unsigned idx) noexcept
  {
    return p_data[idx];
  }

  inline std::size_t shape::operator[](const unsigned idx) const noexcept
  {
    return p_data[idx];
  }

  inline std::ostream &operator<<(std::ostream &out, const shape &s)
  {
    return out << s.str();
  }

  ////////////////////////////// GENERAL ///////////////////////////////

  inline std::size_t shape::size() const noexcept
  {
    return std::accumulate(
      p_data.get(), p_data.get() + m_size, 1UL, std::multiplies<std::size_t> {});
  }

  inline void shape::squeeze() noexcept
  {
    unsigned i { -1U }, j { 0 };
    while (++i < m_size)
      if (p_data[i] != 1) p_data[j++] = p_data[i];
    m_size = j;
  }

  inline std::string shape::str() const
  {
    using namespace std;
    static constexpr auto join = [](auto &a, auto b) { return a + ' ' + to_string(b); };
    return accumulate(p_data.get(), p_data.get() + m_size, string { '(' }, join) + " )";
  }

}  // namespace devi::core::internal

#endif
