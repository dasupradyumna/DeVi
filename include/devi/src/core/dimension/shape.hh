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
    [[nodiscard]] std::size_t operator[](const unsigned idx) const noexcept;

    // Overloading `std::ostream` for easy printing via `std::cout`
    friend std::ostream &operator<<(std::ostream &out, const shape &s);

    ////////////////////////////// GENERAL ///////////////////////////////

    /* Return the dimension value at given flat index
     *
     * Errors:
     * `std::out_of_range` is thrown if argument `i` fails the bounds check
     * */
    [[nodiscard]] std::size_t at(const unsigned i) const;

    // Returns the dimensionality of current shape
    [[nodiscard]] unsigned ndims() const noexcept;

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
  { }

  //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

  inline shape::shape(const shape &copy) : base_dimension { copy } { }

  inline shape::shape(shape &&move) noexcept : base_dimension { std::move(move) } { }

  inline shape &shape::operator=(shape rhs) noexcept
  {
    this->swap(rhs);
    return *this;
  }

  ///////////////////////// OPERATOR OVERLOADS /////////////////////////

  inline std::size_t shape::operator[](const unsigned idx) const noexcept
  {
    return p_data[idx];
  }

  inline std::ostream &operator<<(std::ostream &out, const shape &s)
  {
    return out << s.str();
  }

  ////////////////////////////// GENERAL ///////////////////////////////

  inline std::size_t shape::at(const unsigned i) const
  {
    if (i >= m_size) throw std::out_of_range { "Flat index out of bounds" };

    return (*this)[i];
  }

  inline unsigned shape::ndims() const noexcept { return m_size; }

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
