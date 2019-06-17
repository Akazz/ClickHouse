#pragma once

#include <DataTypes/DataTypesNumber.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnNullable.h>
#include <DataTypes/DataTypeNullable.h>
#include <Common/typeid_cast.h>
#include <IO/WriteHelpers.h>
#include <Functions/IFunction.h>
#include <Functions/FunctionHelpers.h>
#include <Common/FieldVisitors.h>
#include <type_traits>


#if USE_EMBEDDED_COMPILER
#include <DataTypes/Native.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <llvm/IR/IRBuilder.h>
#pragma GCC diagnostic pop
#endif


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int TOO_FEW_ARGUMENTS_FOR_FUNCTION;
    extern const int ILLEGAL_COLUMN;
}

/** Behaviour in the presence of NULLs:
  *
  * Functions AND, XOR, NOT use default implementation for NULLs:
  * - if one of arguments is Nullable, they return Nullable result where NULLs are returned when at least one argument was NULL.
  *
  * But function OR is different.
  * It always return non-Nullable result and NULL are equivalent to 0 (false).
  * For example, 1 OR NULL returns 1, not NULL.
  */
namespace
{

enum class TriValent: UInt8 { False = 0, True = -1, Null = 1 };

inline TriValent makeTriValentValue(bool value, bool is_null = false)
{
    return value * TriValent::True | is_null * TriValent::Null;
}

using UInt8Container = ColumnUInt8::Container;
using UInt8ColumnPtrs = std::vector<const ColumnUInt8 *>;

}

struct AndImpl
{
    static bool supports_ternary_logic = true;
    static bool has_shortcut = true;
    static TriValent shortcut_value = TriValent::False;
    static inline constexpr TriValent apply(TriValent a, TriValent b) { return a & b; }
};

struct OrImpl
{
    static bool supports_ternary_logic = true;
    static bool has_shortcut = true;
    static TriValent shortcut_value = TriValent::True;
    static inline constexpr TriValent apply(TriValent a, TriValent b) { return a | b; }
};

struct XorImpl
{
    static bool supports_ternary_logic = false;
    static bool has_shortcut = false;
    static TriValent shortcut_value = TriValent::Null;
    static inline constexpr TriValent apply(TriValent a, TriValent b) { return a ^ b; }

#if USE_EMBEDDED_COMPILER
    static inline llvm::Value * apply(llvm::IRBuilder<> & builder, llvm::Value * a, llvm::Value * b)
    {
        return builder.CreateXor(a, b);
    }
#endif
};

template <typename A>
struct NotImpl
{
    using ResultType = UInt8;

    static inline UInt8 apply(A a)
    {
        return !a;
    }

#if USE_EMBEDDED_COMPILER
    static inline llvm::Value * apply(llvm::IRBuilder<> & builder, llvm::Value * a)
    {
        return builder.CreateNot(a);
    }
#endif
};


std::pair<ColumnRawPtrs, ColumnRawPtrs> separateConstAndVarColumns(const ColumnRawPtrs & columns)
{
    ColumnRawPtrs const_columns;
    ColumnRawPtrs var_columns;

    const_columns.reserve(columns.size());
    var_columns.reserve(columns.size());

    for (const auto col : columns)
    {
        if (col->isColumnConst())
        {
            const_columns.push_back(col);
        }
        else
        {
            var_columns.push_back(col);
        }
    }

    return {const_columns, var_columns};
}

Columns buildConvertedUInt8Columns(ColumnRawPtrs & columns)
{
    Columns result;
    for (auto column : columns)
    {
        const size_t rows_count = column->size();

        if (column->isColumnConst())
        {
            TriValent converted_value;
            if (column->isColumnNullable())
                converted_value = makeTriValentValue(column->getUInt(0), column->isNullAt(0));
            else
                converted_value = makeTriValentValue(column->getUInt(0));

            result.push_back(
                ColumnConst::create(ColumnUInt8::create(1, converted_value), rows_count)
            );
        }
        else
        {
            const auto new_column = ColumnUInt8::create(rows_count);
            auto & new_data = new_column->getData();

            if (const auto nullable_column = checkAndGetColumn<ColumnNullable>(column))
            {
                const auto & null_map = nullable_column->getNullMapData();
                const auto nested_column = nullable_column->getNestedColumn();

                for (size_t i = 0; i < rows_count; ++i)
                    new_data[i] = makeTriValentValue(nested_column.getUInt(i), null_map[i]);
            }
            else
            {
                for (size_t i = 0; i < rows_count; ++i)
                    new_data[i] = makeTriValentValue(column->getUInt(i));
            }

            result.push_back(new_column);
        }
    }

    return result;
}


template <typename Op, typename VectorType, size_t N>
struct AssociativeOperationImpl
{
    /// Erases the N last columns from `in` (if there are less, then all) and puts into `result` their combination.
//    static void NO_INLINE execute(std::vector<const IColumn *> & in, UInt8Container & result)
    static void NO_INLINE execute(Columns & in, UInt8Container & result)
    {
        if (N > in.size())
        {
            AssociativeOperationImpl<Op, N - 1>::execute(in, result);
            return;
        }

        AssociativeOperationImpl<Op, N> operation(in);
        in.erase(in.end() - N, in.end());

        const size_t n = result.size();
        for (size_t i = 0; i < n; ++i)
            result[i] = operation.apply(i);
    }

    const VectorType & vec;
    AssociativeOperationImpl<Op, N - 1> continuation;

    /// Remembers the last N columns from `in`.
    AssociativeOperationImpl(UInt8ColumnPtrs & in)
        : vec(in[in.size() - N]->getData()), continuation(in) {}

    /// Returns a combination of values in the i-th row of all columns stored in the constructor.
    inline UInt8 apply(size_t i) const
    {
        UInt8 a = vec[i];
        return Op::is_saturable && a == Op::shortcut_value
            ? Op::shortcut_value
            : Op::apply(a, continuation.apply(i));
    }
};

template <typename Op>
struct AssociativeOperationImpl<Op, typename VectorType, 1>
{
    static void execute(UInt8ColumnPtrs &, UInt8Container &)
    {
        throw Exception("Logical error: AssociativeOperationImpl<Op, 1>::execute called", ErrorCodes::LOGICAL_ERROR);
    }

    const UInt8Container & vec;

    AssociativeOperationImpl(UInt8ColumnPtrs & in)
        : vec(in[in.size() - 1]->getData()) {}

    inline UInt8 apply(size_t i) const
    {
        return vec[i];
    }
};


template <typename Op>
ColumnUInt8::Ptr executeAssociativeOperation(const ColumnPtr const_column, const Columns & input_columns, size_t rows_count)
{
    ColumnUInt8::MutablePtr result_column;

    if (const_column != nullptr)
    {
        result_column = const_column->convertToFullColumnIfConst();
    }
    else
    {
        result_column = ColumnUInt8::create(rows_count);
    }

    std::vector<const IColumn *> input_column_ptrs;
    for (const auto & column : input_columns) {
        input_column_ptrs.push_back(column.get());
    }

    if (const_column != nullptr)
        input_column_ptrs.push_back(result_column.get());

    while (input_column_ptrs.size() > 1)
    {
        /// TODO: Rewrite the comment below
        /// Effeciently combine all the columns of the correct type.
        /// With a large block size, combining 10 columns per pass is the fastest.
        /// When small - more, is faster.
        AssociativeOperationImpl<Op, 10>::execute(input_column_ptrs, result_column->getData());
        input_column_ptrs.push_back(result_column.get());
    }

    return result_column;
}


template <typename Impl, typename Name>
class FunctionAnyArityLogical : public IFunction
{
public:
    static constexpr auto name = Name::name;
    static FunctionPtr create(const Context &) { return std::make_shared<FunctionAnyArityLogical>(); }

private:
    template <typename T>
    static bool convertTypeToUInt8(const IColumn * column, UInt8Container & res) const
    {
        auto col = checkAndGetColumn<ColumnVector<T>>(column);
        if (!col)
            return false;
        const auto & vec = col->getData();
        size_t n = res.size();
        for (size_t i = 0; i < n; ++i)
            res[i] = !!vec[i];

        return true;
    }

    template <typename T>
    static bool convertNullableTypeToUInt8(const IColumn * column, UInt8Container & res) const
    {
        auto col_nullable = checkAndGetColumn<ColumnNullable>(column);

        auto col = checkAndGetColumn<ColumnVector<T>>(col_nullable->getNestedColumnPtr());
        if (!col)
            return false;

        const auto & null_map = col_nullable->getNullMapData();
        const auto & vec = col->getData();

        const size_t n = res.size();
        for (size_t i = 0; i < n; ++i)
            res[i] = !null_map[i] && !!vec[i];

        return true;
    }

    static void convertToUInt8(const IColumn * column, UInt8Container & res) const
    {
        convertTypeToUInt8<Int8>(column, res);
        if (!convertTypeToUInt8<Int8>(column, res) &&
            !convertTypeToUInt8<Int16>(column, res) &&
            !convertTypeToUInt8<Int32>(column, res) &&
            !convertTypeToUInt8<Int64>(column, res) &&
            !convertTypeToUInt8<UInt16>(column, res) &&
            !convertTypeToUInt8<UInt32>(column, res) &&
            !convertTypeToUInt8<UInt64>(column, res) &&
            !convertTypeToUInt8<Float32>(column, res) &&
            !convertTypeToUInt8<Float64>(column, res) &&
            !convertNullableTypeToUInt8<Int8>(column, res) &&
            !convertNullableTypeToUInt8<Int16>(column, res) &&
            !convertNullableTypeToUInt8<Int32>(column, res) &&
            !convertNullableTypeToUInt8<Int64>(column, res) &&
            !convertNullableTypeToUInt8<UInt8>(column, res) &&
            !convertNullableTypeToUInt8<UInt16>(column, res) &&
            !convertNullableTypeToUInt8<UInt32>(column, res) &&
            !convertNullableTypeToUInt8<UInt64>(column, res) &&
            !convertNullableTypeToUInt8<Float32>(column, res) &&
            !convertNullableTypeToUInt8<Float64>(column, res))
            throw Exception("Unexpected type of column: " + column->getName(), ErrorCodes::ILLEGAL_COLUMN);
    }

public:
    String getName() const override
    {
        return name;
    }

    bool isVariadic() const override { return true; }
    size_t getNumberOfArguments() const override { return 0; }

    bool useDefaultImplementationForNulls() const override { return Impl::supports_ternary_logic; }

    /// Get result types by argument types. If the function does not apply to these arguments, throw an exception.
    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (arguments.size() < 2)
            throw Exception("Number of arguments for function " + getName() + " doesn't match: passed "
                + toString(arguments.size()) + ", should be at least 2.",
                ErrorCodes::TOO_FEW_ARGUMENTS_FOR_FUNCTION);

        bool has_nullable_args = false;
        for (size_t i = 0; i < arguments.size(); ++i)
        {
            has_nullable_args |= arguments[i]->isNullable();
            if (!(isNativeNumber(arguments[i])
                || (Impl::supports_ternary_logic && (arguments[i]->onlyNull() || isNativeNumber(removeNullable(arguments[i]))))))
                throw Exception("Illegal type ("
                    + arguments[i]->getName()
                    + ") of " + toString(i + 1) + " argument of function " + getName(),
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        }

        DataTypePtr resultType = std::make_shared<DataTypeUInt8>();
        return (has_nullable_args && Impl::supports_ternary_logic)
            ? makeNullable(resultType)
            : resultType;
    }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result, size_t /*input_rows_count*/) override
    {
        ColumnRawPtrs arg_columns(arguments.size());
        for (size_t i = 0; i < arguments.size(); ++i)
            arg_columns[i] = block.getByPosition(arguments[i]).column.get();

        auto separated_columns = separateConstAndVarColumns(arg_columns);

        auto converted_const_columns = buildConvertedUInt8Columns(separated_columns.first);
        bool has_consts = converted_const_columns.size() > 0;

        ColumnUInt8::Ptr const_column_result = executeAssociativeOperation<Impl>(nullptr, converted_const_columns, 1);
        const auto const_value = const_column_result->getElement(0);

        // If const_value uniquely determines the result, return it.
        if (has_consts && Impl::has_shortcut && const_value == Impl::shortcut_value)
        {
            block.getByPosition(result).column = const_column_result;
            return;
        }

        /// If const_value equals neutral element, let's forget about it.
        if (has_consts &&
            Impl::apply(const_value, TriValent::False) == TriValent::False &&
            Impl::apply(const_value, TriValent::True) == TriValent::True)
            has_consts = false;

        /// Here we have to convert the remaining columns to UInt8 and apply
        auto col_res = ColumnUInt8::create();
        UInt8Container & vec_res = col_res->getData();

        size_t rows = col_res->size();
        if (has_consts)
        {
            vec_res.assign(rows, const_value);
            arg_columns.push_back(col_res.get());
        }
        else
        {
            vec_res.resize(rows);
        }

        /// Convert all columns to UInt8
        UInt8ColumnPtrs uint8_in;
        Columns converted_var_columns;

        for (const auto column : arg_columns)
        {
            if (auto uint8_column = checkAndGetColumn<ColumnUInt8>(column))
                uint8_in.push_back(uint8_column);
            else
            {
                auto converted_column = ColumnUInt8::create(rows);
                convertToUInt8(column, converted_column->getData());
                uint8_in.push_back(converted_column.get());
                converted_var_columns.emplace_back(std::move(converted_column));
            }
        }

        /// This is possible if there is exactly one non-constant among the arguments, and it is of type UInt8.
        if (uint8_in[0] != col_res.get())
            vec_res.assign(uint8_in[0]->getData());

        block.getByPosition(result).column = executeAssociativeOperation<Impl>(converted_var_columns, rows);
    }

#if USE_EMBEDDED_COMPILER
    bool isCompilableImpl(const DataTypes &) const override { return true; }

    llvm::Value * compileImpl(llvm::IRBuilderBase & builder, const DataTypes & types, ValuePlaceholders values) const override
    {
        auto & b = static_cast<llvm::IRBuilder<> &>(builder);
        if constexpr (!Impl::isSaturable())
        {
            auto * result = nativeBoolCast(b, types[0], values[0]());
            for (size_t i = 1; i < types.size(); i++)
                result = Impl::apply(b, result, nativeBoolCast(b, types[i], values[i]()));
            return b.CreateSelect(result, b.getInt8(1), b.getInt8(0));
        }
        constexpr bool breakOnTrue = Impl::isShortcutValue(true);
        auto * next = b.GetInsertBlock();
        auto * stop = llvm::BasicBlock::Create(next->getContext(), "", next->getParent());
        b.SetInsertPoint(stop);
        auto * phi = b.CreatePHI(b.getInt8Ty(), values.size());
        for (size_t i = 0; i < types.size(); i++)
        {
            b.SetInsertPoint(next);
            auto * value = values[i]();
            auto * truth = nativeBoolCast(b, types[i], value);
            if (!types[i]->equals(DataTypeUInt8{}))
                value = b.CreateSelect(truth, b.getInt8(1), b.getInt8(0));
            phi->addIncoming(value, b.GetInsertBlock());
            if (i + 1 < types.size())
            {
                next = llvm::BasicBlock::Create(next->getContext(), "", next->getParent());
                b.CreateCondBr(truth, breakOnTrue ? stop : next, breakOnTrue ? next : stop);
            }
        }
        b.CreateBr(stop);
        b.SetInsertPoint(stop);
        return phi;
    }
#endif
};


template <typename A, typename Op>
struct UnaryOperationImpl
{
    using ResultType = typename Op::ResultType;
    using ArrayA = typename ColumnVector<A>::Container;
    using ArrayC = typename ColumnVector<ResultType>::Container;

    static void NO_INLINE vector(const ArrayA & a, ArrayC & c)
    {
        const size_t size = a.size();
        for (size_t i = 0; i < size; ++i)
            c[i] = Op::apply(a[i]);
    }

    static void constant(A a, ResultType & c)
    {
        c = Op::apply(a);
    }
};


template <template <typename> class Impl, typename Name>
class FunctionUnaryLogical : public IFunction
{
public:
    static constexpr auto name = Name::name;
    static FunctionPtr create(const Context &) { return std::make_shared<FunctionUnaryLogical>(); }

private:
    template <typename T>
    bool executeType(Block & block, const ColumnNumbers & arguments, size_t result)
    {
        if (auto col = checkAndGetColumn<ColumnVector<T>>(block.getByPosition(arguments[0]).column.get()))
        {
            auto col_res = ColumnUInt8::create();

            typename ColumnUInt8::Container & vec_res = col_res->getData();
            vec_res.resize(col->getData().size());
            UnaryOperationImpl<T, Impl<T>>::vector(col->getData(), vec_res);

            block.getByPosition(result).column = std::move(col_res);
            return true;
        }

        return false;
    }

public:
    String getName() const override
    {
        return name;
    }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (!isNativeNumber(arguments[0]))
            throw Exception("Illegal type ("
                + arguments[0]->getName()
                + ") of argument of function " + getName(),
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

        return std::make_shared<DataTypeUInt8>();
    }

    bool useDefaultImplementationForConstants() const override { return true; }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result, size_t /*input_rows_count*/) override
    {
        if (!(executeType<UInt8>(block, arguments, result)
            || executeType<UInt16>(block, arguments, result)
            || executeType<UInt32>(block, arguments, result)
            || executeType<UInt64>(block, arguments, result)
            || executeType<Int8>(block, arguments, result)
            || executeType<Int16>(block, arguments, result)
            || executeType<Int32>(block, arguments, result)
            || executeType<Int64>(block, arguments, result)
            || executeType<Float32>(block, arguments, result)
            || executeType<Float64>(block, arguments, result)))
           throw Exception("Illegal column " + block.getByPosition(arguments[0]).column->getName()
                    + " of argument of function " + getName(),
                ErrorCodes::ILLEGAL_COLUMN);
    }

#if USE_EMBEDDED_COMPILER
    bool isCompilableImpl(const DataTypes &) const override { return true; }

    llvm::Value * compileImpl(llvm::IRBuilderBase & builder, const DataTypes & types, ValuePlaceholders values) const override
    {
        auto & b = static_cast<llvm::IRBuilder<> &>(builder);
        return b.CreateSelect(Impl<UInt8>::apply(b, nativeBoolCast(b, types[0], values[0]())), b.getInt8(1), b.getInt8(0));
    }
#endif
};


struct NameAnd { static constexpr auto name = "and"; };
struct NameOr { static constexpr auto name = "or"; };
struct NameXor { static constexpr auto name = "xor"; };
struct NameNot { static constexpr auto name = "not"; };

using FunctionAnd = FunctionAnyArityLogical<AndImpl, NameAnd>;
using FunctionOr = FunctionAnyArityLogical<OrImpl, NameOr>;
using FunctionXor = FunctionAnyArityLogical<XorImpl, NameXor>;
using FunctionNot = FunctionUnaryLogical<NotImpl, NameNot>;

}
