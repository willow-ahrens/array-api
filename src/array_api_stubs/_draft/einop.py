__all__ = ["einop"]

from ._types import array

nary_ops = {
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
    "**": "power",
    "pow": "power",
    "power": "power",
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
    "not": "logical_not",
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
    "abs": "absolute",
    "absolute": "absolute",
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
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "asinh": "arcsinh",
    "acosh": "arccosh",
    "atanh": "arctanh",
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

def einop(prgm: str, /, **kwargs: array) -> array:
    """
    Executes an einsum-like expression.

    This function parses and executes einsum-like expressions with extended syntax
    that supports various operations beyond traditional Einstein summation notation.

    Parameters
    ----------
    prgm: str
        einsum program string specifying the computation. The syntax **must** support:

        -   Assignment: ``"C[i,j] = A[i,j] + B[j,i]"``
        -   Increment: ``"C[i,j] += A[i,k] * B[k,j]"``
        -   Reductions: ``"C[i] += A[i,j]"``, ``"C[i] max= A[i,j]"``, ``"C[i] &= A[i,j]"``
        -   Arithmetic operations: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
        -   Comparison operations: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
        -   Logical operations: ``and``, ``or``, ``not``
        -   Bitwise operations: ``&``, ``|``, ``^``, ``<<``, ``>>``
        -   Function calls and complex expressions with parentheses
        -   Mathematical functions: ``abs``, ``sqrt``, ``exp``, ``log``, ``sin``, ``cos``, ``tan``, etc.
        -   Literal values: integers, floats, booleans, and complex numbers
        -   Python operator precedence and parentheses for grouping
    kwargs: array
        named arrays referenced in the einsum expression. The keys **should** match the
        array names used in ``prgm``.

    Returns
    -------
    out: array
        the result array from executing the einsum expression.

    Notes
    -----

    .. code-block::

      # Matrix addition with transpose
      C = einop("C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
      # Matrix multiplication
      D = einop("D[i,j] += A[i,k] * B[k,j]", A=A, B=B)
      # Min-Plus multiplication with shift
      E = einop("E[i] min= A[i,k] + D[k,j] << 1", A=A, D=D)
    """
