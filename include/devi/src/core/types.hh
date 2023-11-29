#ifndef _HEADER_GUARD__DEVI_SRC_CORE_TYPES_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_TYPES_HH_

#include "__header_check__"

#include <climits>
#include <cstdint>

// Checking C++ floating point width
// TODO: add automatic datatype adjustment in case of failure
static_assert(CHAR_BIT * sizeof(float) == 32, "'float' type should be 32 bits\n");
static_assert(CHAR_BIT * sizeof(double) == 64, "'double' type should be 64 bits\n");

namespace devi::core::internal
{
  // Library supported datatypes for array and view objects
  enum class type {
    // boolean
    bool8,
    // signed integers
    int8,
    int16,
    int32,
    int64,
    // unsigned integers
    uint8,
    uint16,
    uint32,
    uint64,
    // floating point numbers
    float32,
    float64
  };

  // Compile-time type mapper from supported datatypes to native C++ datatypes
  template<type _Type>
  struct native_type;

#define CORE2NATIVE(core_t, native_t) \
  template<>                          \
  struct native_type<type::core_t> {  \
    using type = native_t;            \
  };

  CORE2NATIVE(bool8, bool);
  CORE2NATIVE(int8, std::int8_t);
  CORE2NATIVE(int16, std::int16_t);
  CORE2NATIVE(int32, std::int32_t);
  CORE2NATIVE(int64, std::int64_t);
  CORE2NATIVE(uint8, std::uint8_t);
  CORE2NATIVE(uint16, std::uint16_t);
  CORE2NATIVE(uint32, std::uint32_t);
  CORE2NATIVE(uint64, std::uint64_t);
  CORE2NATIVE(float32, float);
  CORE2NATIVE(float64, double);

}  // namespace devi::core::internal

#endif
