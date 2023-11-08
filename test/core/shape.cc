#include "../utils.hh"

#include <devi/core/dimension/shape>

using devi::core::shape;

static const shape s1 { 3, 10, 1 }, s2 { 5, 0 };

unsigned operators()
{
  // indexing
  ASSERT(1, s1[0] == 3 && s1[1] == 10 && s1[2] == 1);
  ASSERT(2, s2[0] == 5 && s2[1] == 0);
  EXPECT_THROW(std::out_of_range, s1[10]);
  EXPECT_THROW(std::out_of_range, s2[5]);

  // equality
  ASSERT(3, s1 == s1 && s1 != s2 && s2 == s2);
  ASSERT(4, s1 == shape(3, 10, 1) && s2 == shape(5, 0));

  TEST_SUCCESS;
}

unsigned construction()
{
  // NOTE: following code block will throw a compile-time error, when uncommented
  // {
  //   shape t1 {};
  //   shape t2 { -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 };
  //   TEST_FAILURE(1);
  // }

  TEST_SUCCESS;
}

unsigned copy_move()
{
  // construction
  auto s3 { s1 }, _s2 { s1 };
  auto s4 { shape(5, 0) };
  ASSERT(1, s3 == s1 && s4 == s2);

  // assignment
  s3 = s4;
  s4 = std::move(_s2);
  ASSERT(2, s3 != s4 && s3 == s2 && s4 == s1);

  TEST_SUCCESS;
}

unsigned general()
{
  ASSERT(1, s1.ndims() == 3 && s1.size() == 30);
  ASSERT(2, s2.ndims() == 2 && s2.size() == 0);
  ASSERT(3, s1.str() == "( 3 10 1 )" && s2.str() == "( 5 0 )");

  TEST_SUCCESS;
}

int main()
{
  UnitTestRunner tester { "core/dimension/shape", "devi::core::shape" };

  tester.run("Operators", operators);
  tester.run("Construction", construction);
  tester.run("Copy-Move", copy_move);
  tester.run("General", general);

  return tester.passed() == tester.total() ? EXIT_SUCCESS : EXIT_FAILURE;
}