#include "../utils.hh"

#include <devi/core>

using namespace devi::core;

static const int32 a { shape(2, 2) };

unsigned operators()
{
  // flat indexing
  ASSERT(1, a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0);

  // multi-dimensional indexing
  int32 a1 { shape(2, 2) };
  a1(0, 1) = 1;
  a1[3]    = 3;
  ASSERT(2, a1[1] == 1 && a1(1, 1) == 3);
  EXPECT_THROW(3, std::out_of_range, (void)a1(2, 0));
  EXPECT_THROW(4, std::invalid_argument, (void)a1(1, 1, 0));

  // multi-dimensional slicing
  using s = slice;
  int64 a2 { shape(5, 8, 6), 1 };
  // too many slices
  EXPECT_THROW(5, std::invalid_argument, (void)a2(s(0, 5, 2), s(3, 6), 3, 4));
  // zero stride is invalid
  EXPECT_THROW(6, std::invalid_argument, (void)a2(s(0, 5, 0), s(3, 6), 3));
  // slice; begin > end
  EXPECT_THROW(7, std::invalid_argument, (void)a2(s(0, 5, 2), s(8, 6), 3));
  // slice out of bounds
  EXPECT_THROW(8, std::out_of_range, (void)a2(s(0, 5, 2), s(3, 10), 3));
  view<type::int64> v1 { a2(s(0, 5, 2), s(3, 6), 3) };
  ASSERT(9, v1.ndims() == 2 && v1.shape() == shape(3, 3) && v1.size() == 9
              && v1.type() == type::int64);
  v1(0, 0) = 2;
  v1(2, 1) = 3;
  ASSERT(10, a2(0, 3, 3) == 2 && a2(4, 4, 3) == 3);
  auto v2 { a2(s(), s(4)) };
  ASSERT(11, v2.shape() == shape(5, 4, 6) && v2.type() == type::int64);

  // equality
  ASSERT(12, a != int32(shape(4, 1)));
  ASSERT(13, a == a && a != a1 && a1 == a1);
  ASSERT(14, a != int64(shape(2, 2)) && !(a == float32(shape(2, 2))));

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
  const int32 t { shape(4, 1) };

  // construction
  auto a1 { a };
  auto a2 { int32(shape(4, 1)) };
  ASSERT(1, CODE(a1 == a && a2 == t));

  // assignment
  a1 = std::move(a2);
  a2 = a;
  ASSERT(2, CODE(a2 == a && a1 == t));

  TEST_SUCCESS;
}

unsigned getters()
{
  ASSERT(1, a.ndims() == 2);
  ASSERT(2, a.shape() == shape(2, 2));
  ASSERT(3, a.size() == 4);
  ASSERT(4, a.type() == type::int32);

  TEST_SUCCESS;
}

unsigned creation()
{
  auto a_f32t { a.astype<type::float32>() };
  ASSERT(1, a_f32t.shape() == a.shape() && a_f32t.type() == type::float32
              && a_f32t[0] == 0 && a_f32t[1] == 0 && a_f32t[2] == 0 && a_f32t[3] == 0);

  auto a_ = a.copy();
  ASSERT(2, a_ == a);

  TEST_SUCCESS;
}

unsigned mutation()
{
  auto a_ { a };
  shape s { 2, 1, 2 };

  a_.flatten();
  ASSERT(1, a_.shape() == shape(4) && a_.size() == 4);
  a_.reshape(2, 1, 2);
  ASSERT(2, a_.shape() == s && a_.size() == 4);
  a_.reshape(s);
  ASSERT(3, a_.shape() == s && a_.size() == 4);
  a_.reshape(shape(2, 1, 2));
  ASSERT(4, a_.shape() == s && a_.size() == 4);
  a_.squeeze();
  ASSERT(5, a_ == a);

  TEST_SUCCESS;
}

int main()
{
  UnitTestRunner tester { "src/core/array.hh", "devi::core::array" };

  tester.run("Operators", operators);
  tester.run("Construction", construction);
  tester.run("Copy-Move", copy_move);
  tester.run("Getters", getters);
  tester.run("Creation", creation);
  tester.run("Mutation", mutation);

  return tester.passed() == tester.total() ? EXIT_SUCCESS : EXIT_FAILURE;
}
