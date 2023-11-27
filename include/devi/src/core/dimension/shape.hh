#ifndef _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_SHAPE_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_SHAPE_HH_

#include "../__header_check__"
#include "base.hh"

namespace devi::core::internal
{
  // Represents the shape of an array and its dimensionality
  class shape : public _base_ {
  public:

    /* Constructs and returns a `shape` from a variadic list of integer arguments
     *
     * Errors:
     * 1) `new` can throw an `std::bad_alloc` exception
     * 2) compile-time error if the argument list has more than 10 integers
     */
    template<typename... _Args>
    shape(const _Args... args);

    ~shape() noexcept = default;

    //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

    shape(const shape &copy);
    shape(shape &&move) noexcept;
    shape &operator=(shape rhs) noexcept;

    ///////////////////////// OPERATOR OVERLOADS /////////////////////////

    [[nodiscard]] bool operator==(const shape &other) const noexcept;
    [[nodiscard]] bool operator!=(const shape &other) const noexcept;
    [[nodiscard]] std::size_t operator[](const unsigned idx) const noexcept;
    friend std::ostream &operator<<(std::ostream &out, const shape &s);

    ////////////////////////////// GENERAL ///////////////////////////////

    /* Return the dimension value at given index
     *
     * Errors:
     * `std::out_of_range` is thrown if argument `idx` fails the bounds check
     * */
    [[nodiscard]] std::size_t at(const unsigned idx) const;

    // Returns the dimensionality of the current shape
    [[nodiscard]] unsigned ndims() const noexcept;

    // Returns the total size held by the current shape
    [[nodiscard]] std::size_t size() const noexcept;

    // Squeeze the current shape to remove all unit dimensions
    void squeeze() noexcept;

    // Returns the string representation
    [[nodiscard]] std::string str() const;

  };  // class shape

}  // namespace devi::core

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

#include <numeric>

namespace devi::core::internal
{
  template<typename... _Args>
  shape::shape(const _Args... args) : _base_ { args... }
  { }

  //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

  shape::shape(const shape &copy) : _base_ { copy } { }

  shape::shape(shape &&move) noexcept : _base_ { std::move(move) } { }

  shape &shape::operator=(shape rhs) noexcept
  {
    this->swap(rhs);
    return *this;
  }

  ///////////////////////// OPERATOR OVERLOADS /////////////////////////

  bool shape::operator==(const shape &other) const noexcept
  {
    return m_size == other.m_size
        && std::equal(p_data.get(), p_data.get() + m_size, other.p_data.get());
  }

  bool shape::operator!=(const shape &other) const noexcept { return !(*this == other); }

  std::size_t shape::operator[](const unsigned idx) const noexcept { return p_data[idx]; }

  std::ostream &operator<<(std::ostream &out, const shape &s) { return out << s.str(); }

  ////////////////////////////// GENERAL ///////////////////////////////

  std::size_t shape::at(const unsigned idx) const
  {
    if (idx >= m_size) throw std::out_of_range { "index out of bounds" };

    return (*this)[idx];
  }

  unsigned shape::ndims() const noexcept { return m_size; }

  std::size_t shape::size() const noexcept
  {
    return std::accumulate(
      p_data.get(), p_data.get() + m_size, 1UL, std::multiplies<std::size_t> {});
  }

  void shape::squeeze() noexcept
  {
    unsigned i { -1U }, j { 0 };
    while (++i < m_size)
      if (p_data[i] != 1) p_data[j++] = p_data[i];
    m_size = j;
  }

  std::string shape::str() const
  {
    using namespace std;
    static constexpr auto join = [](auto &a, auto b) { return a + ' ' + to_string(b); };
    return accumulate(p_data.get(), p_data.get() + m_size, string { '(' }, join) + " )";
  }

}  // namespace devi::core

#endif
// vim: ft=cpp
