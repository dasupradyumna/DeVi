#include "devi/array.hh"

#include <iostream>

namespace devi
{
  template<typename _DType>
  array<_DType>::array() noexcept
  {
    std::cout << "constructing array" << std::endl;
  }

  template<typename _DType>
  void array<_DType>::test() const noexcept
  {
    std::cout << "testing method" << std::endl;
  }

  // template instantiations
  template class array<int>;
}
