#include "../utils.hh"

#define _HEADER_GUARD__DEVI_CORE_MODULE_  // to bypass the internal header check
#include <devi/src/core/dimension/index.hh>

using namespace devi::core::internal;

static const index i { 3, 5 };
static const shape s1 { 10, 8 }, s2 { 2, 6 };

unsigned general()
{
  ASSERT(1, i.flat(s1) == 29);
  EXPECT_THROW(2, std::out_of_range, (void)i.flat(s2));

  TEST_SUCCESS;
}

int main()
{
  UnitTestRunner tester { "core/dimension/index", "devi::core::index" };

  tester.run("General", general);

  return tester.passed() == tester.total() ? EXIT_SUCCESS : EXIT_FAILURE;
}
