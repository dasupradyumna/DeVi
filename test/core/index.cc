#include "../utils.hh"

#define _HEADER_GUARD__DEVI_CORE_MODULE_  // to bypass the internal header check
#include <devi/src/core/dimension/index.hh>

using namespace devi::core::internal;

unsigned test_index()
{
  const shape s1 { 10, 8 }, s2 { 5, 16 };
  const index i1 { 6, 5 };

  // dot()
  ASSERT(1, CODE(i1.dot(slice_data { 2, 3 }) == 27));

  // flat()
  ASSERT(2, i1.flat(s1) == 53);
  EXPECT_THROW(3, std::out_of_range, (void)i1.flat(s2));

  // transform()
  index i2 { 0, 0 }, i3 { 9, 7 };
  ASSERT(4, i2.flat(s1) == 0 && i3.flat(s1) == 79);
  ASSERT(5, i2.transform(s1, s2).flat(s2) == 0 && i3.transform(s1, s2).flat(s2) == 79);

  TEST_SUCCESS;
}

unsigned test_slice_data()
{
  // get_stride()
  const shape s { 7, 6, 5, 4, 3, 2, 1 };
  auto stride { slice_data::get_stride(s) };
  ASSERT(1, CODE(stride == slice_data { 720, 120, 24, 6, 2, 1, 1 }));

  // append()
  stride[1] = 0;
  stride.append(0);
  ASSERT(2, CODE(stride == slice_data { 720, 0, 24, 6, 2, 1, 1, 0 }));

  // ndims()
  stride.append(5);
  ASSERT(3, stride.ndims() == 9);

  // append() exception
  stride.append(0);
  EXPECT_THROW(4, std::length_error, stride.append(100));

  // remove_zeros() and make_shape()
  stride.remove_zeros();
  ASSERT(5, CODE(stride == slice_data { 720, 24, 6, 2, 1, 1, 5 }));
  ASSERT(6, CODE(stride.make_shape() == shape { 720, 24, 6, 2, 1, 1, 5 }));

  TEST_SUCCESS;
}

int main()
{
  UnitTestRunner tester { "src/core/dimension/(index|slice).hh",
    "devi::core::internal::(index|slice_data)" };

  tester.run("Index", test_index);
  tester.run("Slice Data", test_slice_data);

  return tester.passed() == tester.total() ? EXIT_SUCCESS : EXIT_FAILURE;
}
