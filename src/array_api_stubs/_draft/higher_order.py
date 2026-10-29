__all__ = ["reduce", "elementwise"]

from ._types import Callable, Optional, Tuple, Union, array, dtype


binary_ops = {
    "+": "add",
    "add": "add",
    "-": "subtract",
    "sub": "subtract",
    "subtract": "subtract",
    "*": "multiply",
    "mul": "multiply",
    "multiply": "multiply",
    "/": "divide",
    "div": "divide",
    "divide": "divide",
    "//": "floor_divide",
    "fld": "floor_divide",
    "floor_divide": "floor_divide",
    "%": "remainder",
    "mod": "remainder",
    "remainder": "remainder",
    "**": "pow",
    "pow": "pow",
    "power": "pow",
    "==": "equal",
    "eq": "equal",
    "equal": "equal",
    "!=": "not_equal",
    "ne": "not_equal",
    "not_equal": "not_equal",
    "<": "less",
    "lt": "less",
    "less": "less",
    "<=": "less_equal",
    "le": "less_equal",
    "less_equal": "less_equal",
    ">": "greater",
    "gt": "greater",
    "greater": "greater",
    ">=": "greater_equal",
    "ge": "greater_equal",
    "greater_equal": "greater_equal",
    "&": "bitwise_and",
    "bitwise_and": "bitwise_and",
    "|": "bitwise_or",
    "bitwise_or": "bitwise_or",
    "^": "bitwise_xor",
    "bitwise_xor": "bitwise_xor",
    "<<": "bitwise_left_shift",
    "lshift": "bitwise_left_shift",
    "bitwise_left_shift": "bitwise_left_shift",
    ">>": "bitwise_right_shift",
    "rshift": "bitwise_right_shift",
    "bitwise_right_shift": "bitwise_right_shift",
    "and": "logical_and",
    "or": "logical_or",
    "min": "minimum",
    "max": "maximum",
    "logaddexp": "logaddexp",
}


unary_ops = {
    "+": "positive",
    "pos": "positive",
    "positive": "positive",
    "-": "negative",
    "neg": "negative",
    "negative": "negative",
    "~": "bitwise_invert",
    "invert": "bitwise_invert",
    "bitwise_invert": "bitwise_invert",
    "not": "logical_not",
    "logical_not": "logical_not",
    "abs": "abs",
    "absolute": "abs",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "log1p": "log1p",
    "log10": "log10",
    "log2": "log2",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
}


reduction_ops = {
    "+": "sum",
    "add": "sum",
    "sum": "sum",
    "*": "prod",
    "mul": "prod",
    "prod": "prod",
    "and": "all",
    "or": "any",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "mean": "mean",
    "std": "std",
    "var": "var",
    "count_nonzero": "count_nonzero",
    # "&": "bitwise_and",
    # "|": "bitwise_or",
    # "^": "bitwise_xor",
}


def reduce(
    op: Callable[[array, array], array],
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[dtype] = None,
    keepdims: bool = False,
    init: Optional[Union[bool, int, float, complex]] = None,
) -> array:
    """
    Reduces the input array ``x`` with the binary operator ``op``. Reduces along
    the specified ``axis``, with an initial value ``init``.

    Parameters
    ----------
    op: Callable[[array, array], array]
        binary operator with which to perform the reduction (e.g., :func:`~array_api.add`).
    x: array
        input array. Should have a numeric data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which reduction must be computed. By default, the reduction
        must be computed over the entire array. If a tuple of integers, reductions must
        be computed over multiple axes. Default: ``None``.

    dtype: Optional[dtype]
        data type of the returned array. If ``None``, a suitable data type will be
        calculated.

    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.

    init: Optional[Union[bool, int, float, complex]]
        Initial value for the reduction. If ``None``, a suitable initial value will be
        calculated. The initial value must be compatible with the operation defined by
        ``op``. For example, if ``op`` is addition, the initial value should be zero; if
        ``op`` is multiplication, the initial value should be one.

    Returns
    -------
    out: array
        If the reduction was computed over the entire array, a zero-dimensional array
        containing the reduction; otherwise, a non-zero-dimensional array containing the
        reduction. The returned array must have a data type as described by the
        ``dtype`` parameter above.
    """


def elementwise(f: Callable[..., array], /, *args: array) -> array:
    """
    Applies the function ``f`` elementwise to the given arguments, following
    broadcasting rules (see :ref:`broadcasting`).
    The function ``f`` should be a callable that takes the same number of arguments
    as the number of arrays passed to ``elementwise``.

    The function will automatically handle broadcasting of the input arrays to
    ensure they have compatible shapes.  For example, ``elementwise(add, x, y)``
    is equivalent to ``x + y``.

    Parameters
    ----------
    f: Callable[..., array]
        the function to apply elementwise.
    args: array
        the arrays to apply the function to. These arrays should be
        compatible for broadcasting.

    Returns
    -------
    out: array
        the array, ``out``, of results from applying ``f`` elementwise to
        the input arrays.  After broadcasting the arguments to the same shape, for
        each index ``i``, ``out[*i] = f(args[0][*i], args[1][*i], ...)``.
    """
