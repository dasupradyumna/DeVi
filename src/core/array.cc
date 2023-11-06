#include "devi/core/array.hh"

#include <iostream>

namespace devi::core
{
  template<type _DType>
  array<_DType>::array() noexcept : p_data { new native_type[10] }
  {
    std::cout << "constructing array[10]: " << (_DType == type::bool8) << std::endl;
  }

  template<type _DType>
  array<_DType>::~array() noexcept
  {
    delete[] p_data;
  }

  // template instantiations
  template class array<type::bool8>;
  template class array<type::int8>;
  template class array<type::int16>;
  template class array<type::int32>;
  template class array<type::int64>;
  template class array<type::uint8>;
  template class array<type::uint16>;
  template class array<type::uint32>;
  template class array<type::uint64>;
  template class array<type::float32>;
  template class array<type::float64>;

}  // namespace devi::core
