#ifndef _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_INDEX_HH_
#define _HEADER_GUARD__DEVI_SRC_CORE_DIMENSION_INDEX_HH_

#include "../__header_check__"
#include "slice.hh"

namespace devi::core::internal
{
  // Represents a multi-dimensional index into an array or a view
  class index : public base_dimension {
  public:
    //////////////////////////// CONSTRUCTORS ////////////////////////////

    /* Constructs and returns an `index` from a variadic list of integer arguments
     *
     * Errors:
     * 1) `new` can throw an `std::bad_alloc` exception
     * 2) compile-time error if the argument list has more than 10 integers
     */
    template<typename... _Args>
    index(const _Args... args);

    // Static constructor for an `index` from argument `shape` and flat index `i`
    static index from_flat(const shape &shape, const std::size_t i);

    ////////////////////////////// GENERAL ///////////////////////////////

    // Returns the dot product between current index and argument `data`
    [[nodiscard]] std::size_t dot(const slice_data &data) const;

    // Returns a flat offset obtained by converting current index using argument shape
    [[nodiscard]] std::size_t flat(const shape &shape) const;

    // Transforms current index from shape `src` to shape `dst`
    [[nodiscard]] const index &transform(const shape &src, const shape &dst);

  private:
    // Checks if current index is invalid for argument shape
    [[nodiscard]] bool is_invalid(const shape &shape) const;

    // Replace current index with the index calculated from `shape` and flat index `i`
    [[nodiscard]] const index &unflat(const shape &shape, const std::size_t i) noexcept;

  };  // class index

}  // namespace devi::core::internal

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// IMPLEMENTATION /////////////////////////////////////

namespace devi::core::internal
{
  //////////////////////////// CONSTRUCTORS ////////////////////////////

  template<typename... _Args>
  index::index(const _Args... args) : base_dimension { args... }
  { }

  inline index index::from_flat(const shape &shape, const std::size_t i)
  {
    index index {};
    return index.unflat(shape, i);
  }

  ////////////////////////////// GENERAL ///////////////////////////////

  inline std::size_t index::dot(const slice_data &data) const
  {
    if (m_size != data.ndims())
      throw std::invalid_argument { "Index is incompatible with given slice data" };

    std::size_t dot { 0 };
    for (unsigned i { -1U }; ++i < m_size;) dot += p_data[i] * data[i];

    return dot;
  }

  inline std::size_t index::flat(const shape &shape) const
  {
    if (this->is_invalid(shape))
      throw std::out_of_range { "Index out of bounds for the given shape" };

    std::size_t stride { 1 }, flat { 0 };
    for (auto i { m_size }; i--; stride *= shape[i]) flat += stride * p_data[i];

    return flat;
  }

  inline const index &index::transform(const shape &src, const shape &dst)
  {
    if (src == dst) return *this;

    if (src.size() != dst.size())
      throw std::invalid_argument {
        "Argument shapes are not compatible for transformation"
      };

    return this->unflat(dst, this->flat(src));
  }

  inline bool index::is_invalid(const shape &shape) const
  {
    if (m_size != shape.ndims())
      throw std::invalid_argument { "Index is incompatible with given shape" };

    for (unsigned i { -1U }; ++i < m_size;)
      if (p_data[i] >= shape[i]) return true;
    return false;
  }

  inline const index &index::unflat(const shape &shape, const std::size_t i) noexcept
  {
    m_size = shape.ndims();
    std::size_t flat { i }, stride { 1 };
    for (auto i { m_size }; i--; stride *= shape[i])
      p_data[i] = (flat % (stride * shape[i])) / stride;

    return *this;
  }

}  // namespace devi::core::internal

#endif
