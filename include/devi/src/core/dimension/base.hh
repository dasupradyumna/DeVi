#ifndef _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_BASE_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_BASE_HH_

#include "../__header_check__"

#include <memory>

namespace devi::core::internal
{
  class _base_ {
  public:

    ~_base_() noexcept = default;

  protected:

    // Variadic integer list constructor
    template<typename... _Args,
      typename = std::enable_if_t<(std::is_integral_v<_Args> && ...)>>
    _base_(const _Args... args);

    //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

    _base_(const _base_ &copy);
    _base_(_base_ &&move) noexcept;
    _base_ &operator=(_base_ rhs) noexcept;

    // Swap state with existing shape
    void swap(_base_ &b) noexcept;
    // Swap state with temporary _base_
    void swap(_base_ &&b) noexcept;

    ///////////////////////////// ATTRIBUTES /////////////////////////////

    std::unique_ptr<std::size_t[]> p_data;
    unsigned m_size;
    static constexpr unsigned MAX_SIZE { 10 };

  };  // class _base_

}  // namespace devi::core::internal

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

#include <algorithm>

namespace devi::core::internal
{

  template<typename... _Args, typename>
  _base_::_base_(const _Args... args)
    : p_data { new std::size_t[MAX_SIZE] {} }, m_size { 0 }
  {
    static_assert(sizeof...(args) > 0 && sizeof...(args) <= MAX_SIZE,
      "No. of arguments to devi::core::shape() must belong to [1, 10]");

    ((p_data[m_size++] = args), ...);
  }

  //////////////////////// COPY-MOVE SEMANTICS /////////////////////////

  _base_::_base_(const _base_ &copy)
    : p_data { new std::size_t[MAX_SIZE] }, m_size { copy.m_size }
  {
    std::copy_n(copy.p_data.get(), m_size, p_data.get());
  }

  _base_::_base_(_base_ &&move) noexcept
    : p_data { std::move(move.p_data) }, m_size { move.m_size }
  { }

  _base_ &_base_::operator=(_base_ rhs) noexcept
  {
    this->swap(rhs);
    return *this;
  }

  void _base_::swap(_base_ &b) noexcept
  {
    std::swap(p_data, b.p_data);
    std::swap(m_size, b.m_size);
  }

  void _base_::swap(_base_ &&b) noexcept { this->swap(b); }

}  // namespace devi::core::internal

#endif
// vim: ft=cpp
