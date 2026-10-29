__all__ = ["einop"]

from ._types import array


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

    -   Operations **must** be evaluated as with :func:`~array_api.elementwise`, and reductions as with :func:`~array_api.reduce`.

    .. code-block::

      # Matrix addition with transpose
      C = einop("C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
      # Matrix multiplication
      D = einop("D[i,j] += A[i,k] * B[k,j]", A=A, B=B)
      # Min-Plus multiplication with shift
      E = einop("E[i] min= A[i,k] + D[k,j] << 1", A=A, D=D)
    """
