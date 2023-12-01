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

#### `devi::core::shape`

This class represents the shape and dimensionality of an array object. It is copyable, movable and
assignable. Current implementation supports only upto **10 dimensions**; any attempt at creating a
`shape` of higher dimensionality will throw a *compile-time* error.

```cpp
#include <cassert>
#include <iostream>

#include <devi/core>

using namespace devi;

int main()
{
    const core::shape s1 { 1, 800, 3, 600 };    // Creates a 4D shape object

    std::cout << s1 << std::endl;               // Prints "( 1 800 3 600 )"
    std::cout << s1.ndims() << std::endl;       // Prints "4"
    std::cout << s1.size() << std::endl;        // Prints "1440000"

    auto s2 { s1 };             // Creates a copy of `s1`
    s2[2] = 1;                  // `s2` is now `core::shape(1, 800, 1, 600)`
    // s2[5];                   // Out of bounds access: undefined behavior
    s2.squeeze();               // Removes all unit dimensions
    assert(s2 == core::shape(800, 600) && s2 != s1);
}
```

#### `devi::core::type`

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

#### `devi::core::array`

**#TODO#**
