#ifndef _HEADER_GUARD__DEVI_TEST_UTILS_HH_
#define _HEADER_GUARD__DEVI_TEST_UTILS_HH_

// Wraps arguments (which contain commas) passed to other macros
#define CODE(...) __VA_ARGS__

/* Function-friendly assertion
 * Returns false if condition fails, continues execution otherwise
 */
#define ASSERT(id, condition) \
  if (!(condition)) return id;

/* Expect-exception block
 * Returns false if the input code snippet does not throw the specified exception
 */
#define EXPECT_THROW(id, exception, snippet) \
  while (true) {                             \
    try {                                    \
      snippet;                               \
    }                                        \
    catch (const exception &) {              \
      break;                                 \
    }                                        \
    catch (...) {                            \
      return id;                             \
    }                                        \
    return id;                               \
  }

// Success statement
#define TEST_SUCCESS return 0;

// Failure statement with a failure code
#define TEST_FAILURE(id) return id;

#include <iostream>

class UnitTestRunner {

  unsigned m_passed;
  unsigned m_total;

public:

  UnitTestRunner(const std::string &class_file, const std::string &class_name)
    : m_passed { 0 }, m_total { 0 }
  {
    std::cout << "Testing: `" << class_name << "` from <" << class_file << ">\n";
  }

  ~UnitTestRunner()
  {
    std::cout << "Done.\nPassed : " << m_passed << '/' << m_total << '\n';
  }

  unsigned passed() const noexcept { return m_passed; }

  unsigned total() const noexcept { return m_total; }

  template<typename... _TestFuncArgs>
  void run(const std::string &test_name, unsigned test_func(_TestFuncArgs...),
    _TestFuncArgs... args)
  {
    ++m_total;
    std::cout << "  " << test_name << " - ";
    auto ret { test_func(args...) };
    switch (ret) {
      case 0:
        std::cout << "passed\n";
        ++m_passed;
        break;
      default: std::cout << "FAILED (" << ret << ")\n";
    }
  }
};

#endif
