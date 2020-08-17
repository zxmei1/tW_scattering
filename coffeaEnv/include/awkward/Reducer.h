// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REDUCER_H_
#define AWKWARD_REDUCER_H_

#include <memory>

#include "awkward/Index.h"

namespace awkward {
  /// @class Reducer
  ///
  /// @brief Abstract class for all reducer algorithms.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL Reducer {
  public:
    /// @brief Name of the reducer algorithm.
    virtual const std::string
      name() const = 0;

    /// @brief Data type to prefer, as a NumPy dtype, if the array has
    /// UnknownType.
    virtual util::dtype
      preferred_dtype() const = 0;

    /// @brief Return type for a `given_dtype` as a NumPy dtype.
    virtual util::dtype
      return_dtype(util::dtype given_dtype) const;

    /// @brief Apply the reducer algorithm to an array of boolean values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of signed 8-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of unsigned 8-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of signed 16-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of unsigned 16-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of signed 32-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of unsigned 32-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of signed 64-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of unsigned 64-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of 32-bit
    /// floating-point values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of 64-bit
    /// floating-point values.
    ///
    /// @param data The array to reduce.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const = 0;
  };

  /// @class ReducerCount
  ///
  /// @brief Reducer algorithm that simply counts items. The identity is `0`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerCount: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"count"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerCount is `double`.
    util::dtype
      preferred_dtype() const override;

    /// @copydoc Reducer::return_dtype()
    ///
    /// The return type for ReducerCount is `int64`.
    util::dtype
      return_dtype(util::dtype given_dtype) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerCountNonzero
  ///
  /// @brief Reducer algorithm that counts non-zero items. The identity is `0`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerCountNonzero: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"count_nonzero"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_nptype()
    ///
    /// The preferred type for ReducerCountNonzero is `double`.
    util::dtype
      preferred_dtype() const override;

    /// @copydoc Reducer::return_dtype()
    ///
    /// The return type for ReducerCountNonzero is `int64`.
    util::dtype
      return_dtype(util::dtype given_dtype) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerSum
  ///
  /// @brief Reducer algorithm that adds up items. The identity is `0`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerSum: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"sum"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerSum is `double`.
    util::dtype
      preferred_dtype() const override;

    /// @copydoc Reducer::return_dtype()
    ///
    /// The return type for ReducerSum promotes integers and booleans to
    /// 64-bit but leaves floating-point number types as they are.
    util::dtype
      return_dtype(util::dtype given_dtype) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerProd
  ///
  /// @brief Reducer algorithm that multiplies items. The identity is `1`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerProd: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"prod"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerProd is `int64`.
    util::dtype
      preferred_dtype() const override;

    /// @copydoc Reducer::return_dtype()
    ///
    /// The return type for ReducerProd promotes integers and booleans to
    /// 64-bit but leaves floating-point number types as they are.
    util::dtype
      return_dtype(util::dtype given_dtype) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerAny
  ///
  /// @brief Reducer algorithm that returns `true` if any values are `true`,
  /// `false` otherwise. The identity is `false`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerAny: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"any"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerAny is `boolean`.
    util::dtype
      preferred_dtype() const override;

    /// @copydoc Reducer::return_dtype()
    ///
    /// The return type for ReducerAny is `boolean`.
    util::dtype
      return_dtype(util::dtype given_dtype) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerAll
  ///
  /// @brief Reducer algorithm that returns `true` if all values are `true`,
  /// `false` otherwise. The identity is `true`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerAll: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"all"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerAll is `boolean`.
    util::dtype
      preferred_dtype() const override;

    /// @copydoc Reducer::return_dtype()
    ///
    /// The return type for ReducerAll is `boolean`.
    util::dtype
      return_dtype(util::dtype given_dtype) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerMin
  ///
  /// @brief Reducer algorithm that returns the minimumm value. The identity
  /// is infinity or the largest possible value.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerMin: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"min"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerMin is `double`.
    util::dtype
      preferred_dtype() const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerMax
  ///
  /// @brief Reducer algorithm that returns the maximum value. The identity
  /// is minus infinity or the smallest possible value.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerMax: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"max"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerMax is `double`.
    util::dtype
      preferred_dtype() const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerArgmin
  ///
  /// @brief Reducer algorithm that returns the position of the minimum value.
  /// The identity is meaningless and should be covered using `mask = true`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerArgmin: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"argmin"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerArgmin is `int64`.
    util::dtype
      preferred_dtype() const override;

    /// @copydoc Reducer::return_dtype()
    ///
    /// The return type for ReducerArgmin is `int64`.
    util::dtype
      return_dtype(util::dtype given_dtype) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerArgmax
  ///
  /// @brief Reducer algorithm that returns the position of the maximum value.
  /// The identity is meaningless and should be covered using `mask = true`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL ReducerArgmax: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"argmax"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_dtype()
    ///
    /// The preferred type for ReducerArgmax is `int64`.
    util::dtype
      preferred_dtype() const override;

    /// @copydoc Reducer::return_dtype()
    ///
    /// The return type for ReducerArgmax is `int64`.
    util::dtype
      return_dtype(util::dtype given_dtype) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

}

#endif // AWKWARD_REDUCER_H_
