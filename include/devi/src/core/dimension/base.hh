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

#ifndef _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_BASE_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_BASE_HH_

#include "../__header_check__"

#include <memory>

namespace devi::core::internal
{
  // Base class for all dimensionality-related classes
  class base_dimension {
  public:
    // Default destructor
    ~base_dimension() noexcept = default;

  protected:
    //////////////////////////// CONSTRUCTORS ////////////////////////////

    // Variadic integer list constructor
    template<typename... _Args,
      typename = std::enable_if_t<(std::is_integral_v<_Args> && ...)>>
    base_dimension(const _Args... args);

    //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

    // Copy constructor
    base_dimension(const base_dimension &copy);

    // Move constructor
    base_dimension(base_dimension &&move) noexcept;

    // Copy-and-Swap idiom for assignment operator
    base_dimension &operator=(base_dimension rhs) noexcept;

    ////////////////////////////// GENERAL ///////////////////////////////

    // Equality operator overload
    [[nodiscard]] bool operator==(const base_dimension &other) const noexcept;

    // Inequality operator overload
    [[nodiscard]] bool operator!=(const base_dimension &other) const noexcept;

    // Swap state with existing object
    void swap(base_dimension &b) noexcept;
    // Swap state with temporary object
    void swap(base_dimension &&b) noexcept;

    ///////////////////////////// ATTRIBUTES /////////////////////////////

    std::unique_ptr<std::size_t[]> p_data;
    unsigned m_size;
    static constexpr unsigned MAX_SIZE { 10 };

  };  // class base_dimension

}  // namespace devi::core::internal

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

#include <algorithm>

namespace devi::core::internal
{
  //////////////////////////// CONSTRUCTORS ////////////////////////////

  template<typename... _Args, typename>
  base_dimension::base_dimension(const _Args... args)
    : p_data { new std::size_t[MAX_SIZE] {} }, m_size { 0 }
  {
    static_assert(sizeof...(args) <= MAX_SIZE,
      "No. of arguments to `devi::core::base_dimension` must be atmost 10");

    ((p_data[m_size++] = args), ...);
  }

  //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

  inline base_dimension::base_dimension(const base_dimension &copy)
    : p_data { new std::size_t[MAX_SIZE] }, m_size { copy.m_size }
  {
    std::copy_n(copy.p_data.get(), m_size, p_data.get());
  }

  inline base_dimension::base_dimension(base_dimension &&move) noexcept
    : p_data { std::move(move.p_data) }, m_size { move.m_size }
  { }

  inline base_dimension &base_dimension::operator=(base_dimension rhs) noexcept
  {
    this->swap(rhs);
    return *this;
  }

  ////////////////////////////// GENERAL ///////////////////////////////

  inline bool base_dimension::operator==(const base_dimension &other) const noexcept
  {
    return m_size == other.m_size
        && std::equal(p_data.get(), p_data.get() + m_size, other.p_data.get());
  }

  inline bool base_dimension::operator!=(const base_dimension &other) const noexcept
  {
    return !(*this == other);
  }

  inline void base_dimension::swap(base_dimension &b) noexcept
  {
    std::swap(p_data, b.p_data);
    std::swap(m_size, b.m_size);
  }

  inline void base_dimension::swap(base_dimension &&b) noexcept { this->swap(b); }

}  // namespace devi::core::internal

#endif
