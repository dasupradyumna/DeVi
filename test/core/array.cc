#include "../utils.hh"

#include <devi/core/array>

using namespace devi::core;

static const int32 a { shape(2, 2) };

unsigned operators()
{
  // indexing
  ASSERT(1, a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0);
  int32 a_ { shape(2, 2) };
  a_[1] = a_[3] = 1;

  // equality
  ASSERT(2, CODE(a != int32 { shape(4, 1) }));
  ASSERT(3, a == a && a != a_ && a_ == a_);

  TEST_SUCCESS;
}

unsigned construction()
{
  // fill
  int32 a_ { shape(2, 2), 10 };
  ASSERT(1, a_[0] == 10 && a_[1] == 10 && a_[2] == 10 && a_[3] == 10);

  TEST_SUCCESS;
}

unsigned copy_move()
{
  const auto t { int32 { shape(4, 1) } };

  // construction
  auto a1 { a };
  auto a2 { int32 { shape(4, 1) } };
  ASSERT(1, CODE(a1 == a && a2 == t));

  // assignment
  a1 = std::move(a2);
  a2 = a;
  ASSERT(2, CODE(a2 == a && a1 == t));

  TEST_SUCCESS;
}

unsigned getters()
{
  EXPECT_THROW(1, std::out_of_range, a.at(5));
  ASSERT(2, a.ndims() == 2);
  ASSERT(3, a.shape() == shape(2, 2));
  ASSERT(4, a.size() == 4);
  ASSERT(5, a.type() == type::int32);

  TEST_SUCCESS;
}

int main()
{
  UnitTestRunner tester { "core/array", "devi::core::array" };

  tester.run("Operators", operators);
  tester.run("Construction", construction);
  tester.run("Copy-Move", copy_move);
  tester.run("Getters", getters);

  return tester.passed() == tester.total() ? EXIT_SUCCESS : EXIT_FAILURE;
}
