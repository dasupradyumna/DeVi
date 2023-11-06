#pragma once

#include "devi/core/types.hh"

namespace devi::core
{
  template<type _DType>
  class array;

  // aliases
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

  template<type _DType>
  class array {
  public:
    using native_type = typename native_type<_DType>::type;

    array() noexcept;
    ~array() noexcept;

  private:
    native_type *p_data;
  };

}  // namespace devi::core
