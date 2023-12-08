# DeVi : Deep Vision library

**DeVi** is a C++ library with the goal of implementing computer vision and deep learning techniques
without the use of external libraries ***except** for the C libraries handling image and video
codecs*. My vision for this library is to design an API that provides a user experience as similar
as possible to its *Python* contemporaries in a C++ development environment.
This project conforms to the **ISO C++17** standard.

---

***NOTE:** This project is both an ambitious long-term experiment to see if my vision is feasible
while being performant as well as a learning opportunity to tackle all the challenges that come with
building a compute-heavy multithreaded C++ library. I will have to study software design, system
architecture, multithreading, CUDA GPU programming and ofcourse, the various techniques themselves
to be able to successfully implement the library. During development, I will actively avoid
referencing contemporary C++ implementations from libraries like **Eigen**, **OpenCV**, **PyTorch**,
and **TensorFlow** as much as possible.*

---

### Modules

The library is currently planned to have 3 modules: **core**, **vis** and **net**.  
The **core** module will hold basic classes and common utilities that will be needed by the other 2
modules, the **vis** module will hold the computer vision techniques and image utilities, whereas
the **net** module will hold all the deep learning techniques and neural network related classes.

#### Current Status

*(Under Active Development)*  
**core** module is being developed at the moment, which houses a multi-dimensional `array` template
class heavily inspired by the **numpy.ndarray** API. This array class is necessary for implementing
`image` and `tensor` classes in the upcoming **vis** and **net** modules.

## API

The top-level namespace that encloses all of **DeVi**'s functionality is `devi`, and the **core**
module is exported as the nested `devi::core` namespace. The upcoming **vis** and **net** modules
will be exported as `devi::vis` and `devi::net` namespaces respectively.

### `core` module

To use the functionality enclosed in the **core** module, add `#include <devi/core>` in the files
that require them.

#### 1. `devi::core::shape`

This class represents the shape and dimensionality of an array object. It is copyable, movable and
assignable. Current implementation supports only upto **10 dimensions**; any code that attempts to
create a `shape` of higher dimensionality will throw a *compile-time* error.

```cpp
#include <devi/core>

using namespace devi::core;

// All following examples will assume the above include statement and using-directive

#include <cassert>
#include <iostream>

int main()
{
    const shape s1 { 1, 800, 3, 600 };          // Creates a 4D shape object

    std::cout << s1 << std::endl;               // Prints "( 1 800 3 600 )"
    std::cout << s1.ndims() << std::endl;       // Prints "4"
    std::cout << s1.size() << std::endl;        // Prints "1440000"

    auto s2 { s1 };             // Creates a copy of `s1`
    s2[2] = 1;                  // `s2` is now `shape(1, 800, 1, 600)`
    // s2[5];                   // Out of bounds access: undefined behavior
    s2.squeeze();               // Removes all unit dimensions
    assert(s2 == shape(800, 600) && s2 != s1);
}
```

#### 2. `devi::core::type`

This class represents the (library-supported) datatype of an array object. It is an enumeration type
and currently supports **11** datatypes.

```cpp
// Below are the following supported datatypes
devi::core::type::bool8;    // boolean
devi::core::type::int8;     // 8-bit signed integer
devi::core::type::int16;    // 16-bit signed integer
devi::core::type::int32;    // 32-bit signed integer
devi::core::type::int64;    // 64-bit signed integer
devi::core::type::uint8;    // 8-bit unsigned integer
devi::core::type::uint16;   // 16-bit unsigned integer
devi::core::type::uint32;   // 32-bit unsigned integer
devi::core::type::uint64;   // 64-bit unsigned integer
devi::core::type::float32;  // 32-bit floating point
devi::core::type::float64;  // 64-bit floating point
```

#### 3. `devi::core::array`

This class represents the actual **data-owning** array object. It is copyable, movable and
assignable. The type of data stored by the array is specified by its template argument of type
`devi::core::type`. The memory owned by an array is *guaranteed* to be **contiguous**.

- **Convenience Aliases**  
    Since repeatedly invoking the template syntax and dealing with the datatype enumeration directly
    is cumbersome, the library provides type alises for all supported datatypes.

    ```cpp
    using devi::core::bool8 = devi::core::array<devi::core::type::bool8>;
    using devi::core::int8 = devi::core::array<devi::core::type::int8>;
    using devi::core::int16 = devi::core::array<devi::core::type::int16>;
    // ... so on, for every value defined by `devi::core::type`
    ```

- **Constructors**  
  - `array<type _DType>(const shape &s)`: Default Constructor  
    `array<type _DType>(shape &&s)`: R-value Reference Overload  
    Creates an array having specified shape with all elements initialized to **zero**
  - `array<type _DType>(const shape &s, const native_type fill)`: Fill Constructor  
    `array<type _DType>(shape &&s, const native_type fill)`: R-value Reference Overload  
    Creates an array having specified shape with all elements initialized to the value **fill**  
    *(`native_type` is the C++ builtin type that corresponds to the respective `devi::core::type`)*

    ```cpp
    array<type::int32> I { shape(1920, 1080) };        // 1920x1080 array filled with 0s
    int32 i1 { shape(1920, 1080) };                    // Identical to above declaration
    array<type::float32> F { shape(1920, 1080), 10 };  // 1920x1080 array filled with 10s
    float32 f1 { shape(1920, 1080), 10 };              // Identical to above declaration
    ```

- **Equality and Inequality**  
  - `bool array::operator==(const array &other) const noexcept`: Equality  
    `template<enum type _Other>`  
    `bool operator==(const array<_Other> &other) const noexcept`: Different Datatype Overload  
    2 arrays are said to be equal if their datatypes, shapes and contained elements are all equal
  - `bool array::operator!=(const array &other) const noexcept`: Inequality  
    `template<enum type _Other>`  
    `bool operator!=(const array<_Other> &other) const noexcept`: Different Datatype Overload  
    Opposite of equality conditions

    ```cpp
    int32 i2 { i1 }, i3 { shape(1920, 1080), 10 }, i4 { shape(1080, 1920), 10 };
    assert(i2 == i1);          // Same dataype, and contained elements are equal
    assert(i2 != f1);          // Contained elements are unequal, and different datatype
    assert(i3 != i1);          // Same dataype, but contained elements are unequal
    assert(i3 != f1);          // Contained elements are equal, but different datatype
    assert(i3 != i4);          // Same datatype and contained elements, but different shape
    ```

- **Getters**  
  - `unsigned array::ndims() const noexcept`  
    Returns the dimensionality of the array
  - `const class shape &array::shape() const noexcept`  
    Returns the shape of the array
  - `std::size_t array::size() const noexcept`  
    Returns the total size of the array
  - `enum type array::type() const noexcept`  
    Returns the `devi::core::type` of the array

    ```cpp
    assert(i1.ndims() == 2);
    assert(i1.shape() == shape(1920, 1080));
    assert(i1.size() == 2073600);
    assert(i1.type() == type::int32);
    ```

- **Array Creation**  
  - `template<enum type _AsType>`  
    `array<_AsType> array::astype() const`  
    Returns a element-wise type-casted copy of the current array
  - `array array::copy() const`  
    Returns a copy of the current array

    ```cpp
    assert(f1.astype<type::int32>() == i3);
    assert(f1 == f1.copy());
    ```

- **In-place Mutation**  
  - `void array::fill(const native_type val) noexcept`  
    Sets all elements of the array to the value `val`
  - `void array::flatten()`  
    Flattens the multi-dimensional array into a 1D array while preserving the total owned size
  - `template<typename... _Args>`  
    `void array::reshape(const _Args... args)`: Parameter Pack Version  
    `void array::reshape(const shape &s)`: L-value Reference `shape` Version  
    `void array::reshape(shape &&s) noexcept`: R-value Reference `shape` Version  
    Changes the shape of the array while preserving the total owned size
  - `void array::squeeze() noexcept`  
    Removes all unit dimensions in the shape of the array
  - `void array::swap(array &b) noexcept`: L-value Reference Version  
    `void array::swap(array &&b) noexcept`: R-value Reference Version  
    Swaps the owned data and the shapes of the arrays

    ```cpp
    i2.fill(10);                    // All elements are now equal to 10
    assert(i2 == i3);
    i2.flatten();                   // `i2` now has shape `( 2073600 )`
    assert(i2.shape() == shape(2073600));
    i2.reshape(1080, 1920);         // `i2` now has shape `( 1080 1920 )`
    const shape s { 1080, 1920 };
    i2.reshape(s);                  // Identical to above method call
    assert(i2 == i4);
    i2.reshape(shape(1, 120, 9, 1, 1, 1920, 1));
    i2.squeeze();                   // All unit dimensions are removed
    assert(i2.shape() == shape(120, 9, 1920));
    ```

- **Indexing**  
    `template<typename... _Indices, typename>`  
    `native_type &array::operator()(const _Indices... indices)`  
    Returns a non-const reference to the value specified by the index `indices`  
    `template<typename... _Indices, typename>`  
    `native_type array::operator()(const _Indices... indices) const`  
    Returns a copy of the value specified by the index `indices`

    *The second template parameter is used for compile-time type checking using **SFINAE***

    Exceptions:  
    1) `std::invalid_argument` if the number of `indices` arguments is not equal to array's
       dimensionality
    2) `std::out_of_range` if the number of argument `indices` is valid, but out of bounds for
       atleast one dimension in array's shape

    ```cpp
    uint16 u1 { shape(20, 100, 80) };
    u1(5, 67, 83) = 42;
    u1(3, 12, 21, 1);    // Throws `std::invalid_argument`; too many indices
    u1(10, 8);           // Throws `std::invalid_argument`; not enough indices
    u1(16, 79, 91);      // Throws `std::out_of_range`; 3rd index is out of bounds
    ```

- **Slicing**  

  - `devi::core::slice`  
    The library provides this struct for users to specify how to slice an array dimension. A `slice`
    constitutes of a **begin** index, an **end** index and a **stride** value; this slice denotes
    all values beginning from (and including) **begin** until (and excluding) **end**, picking
    elements **stride** indices apart.  
    All three arguments are optional. **begin** index defaults to 0 (start of a dimension), **end**
    index defaults to 0 (will be converted to end of a dimension), **stride** value defaults to 1
    (no element is skipped).  
    The **end** index (when specified) *MUST* be atleast one more than the **begin** index, whereas
    **stride** *MUST* be a non-zero value.

    Exceptions (by constructor):  
    `std::invalid_argument` if **begin** index is greater than or equal to **end** index, or if
    **stride** value is **zero**

  - `template<typename... _Slices, typename>`  
    `view<_DType> array::operator()(const _Slices &...slices)`  
    Returns a `view` object that is the result of the specified slicing.  
    Each `slices` argument can either be a `slice` object or an integral index.  
    If the number of `slices` arguments is less than the array's dimensionality, the missing slices
    are automatically assumed to span the entire corresponding dimension.

    *The second template parameter is used for compile-time type checking using **SFINAE***

    Exceptions:  
    1) `std::invalid_argument` if the number of `slices` arguments is greater than array's
       dimensionality  
    2) `std::out_of_range` if the number of argument `slices` is valid, but out of bounds for
       atleast one dimension in array's shape, i.e. if `slices` is an integer, it is out of bounds
       of array shape and if `slices` is a `devi::core::slice` object, the begin or end or both are
       out of bounds of array shape

    ```cpp
    using s_ = slice;
    view<type::uint16> v1 { u1(s_(7, 15), 56, s_(10, 69, 3)) };
    // The view `v1` has shape `( 8 20 )`
    // 1st dimension: from 7 to 14, (implicit) every element = (15 - 7) / 1 = 8
    // 2nd dimension: picked 56 = no dimension
    // 3rd dimension: from 10 to 67, every 3rd element = (69 - 10) / 3 + 1 = 20
    view<type::uint16> v2 { u1(6, s_(20)) };
    // The view `v2` has shape `( 80 80 )`
    // 1st dimension: picked 6 = no dimension
    // 2nd dimension: from 20 to (implicit) 100, every element = (100 - 20) / 1 = 80
    // 3rd dimension: unspecified = entire dimension = 80
    ```

#### 4. `devi::core::view`

This class represents a **data-viewing** and **non-owning** window into an array or another view
object. It is copyable, movable and assignable. There is *no guarantee* for the memory that is
handled by the view to be contiguous. A **view** can never be constructed by the user; they can
**only** be created by a slicing operation on arrays or other views.

*It is intended for `view` and `array` to have the same API for a consistent experience*

- **Getters**  
  - `unsigned view::ndims() const noexcept`  
    Returns the dimensionality of the view
  - `const class shape &view::shape() const noexcept`  
    Returns the shape of the view
  - `std::size_t view::size() const noexcept`  
    Returns the total size of the view
  - `enum type view::type() const noexcept`  
    Returns the `devi::core::type` of the view

    ```cpp
    using s_ = slice;
    uint64 u2 { shape(4, 10, 150, 90), 1 };
    auto v2 { u2(s_(0, 4), 3, s_(40, 120, 8), s_(30, 80)) };
    assert(v2.ndims() == 3);
    assert(v2.shape() == shape(4, 10, 50));
    assert(v2.size() == 2000);
    assert(v2.type() == type::uint64);
    ```

- **Indexing**  
    `template<typename... _Indices, typename>`  
    `native_type &view::operator()(const _Indices... indices)`  
    Returns a non-const reference to the value specified by the index `indices`  
    `template<typename... _Indices, typename>`  
    `native_type view::operator()(const _Indices... indices) const`  
    Returns a copy of the value specified by the index `indices`

    *The second template parameter is used for compile-time type checking using **SFINAE***

    Exceptions:  
    1) `std::invalid_argument` if the number of `indices` arguments is not equal to view's
       dimensionality
    2) `std::out_of_range` if the number of argument `indices` is valid, but out of bounds for
       atleast one dimension in view's shape

    ```cpp
    v2(2, 7, 33) = 42;      // This will modify the value in the appropriate `u2` memory
    assert(u2(2, 3, 96, 63) == 42);
    v2(3, 7, 21, 1);        // Throws `std::invalid_argument`; too many indices
    v2(1, 8);               // Throws `std::invalid_argument`; not enough indices
    v2(16, 9, 41);          // Throws `std::out_of_range`; 3rd index is out of bounds
    ```

***TODO:** implement a `const_view` object which is a non-mutable window into the memory it slices*
