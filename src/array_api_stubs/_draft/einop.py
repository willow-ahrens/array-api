__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]

from ._types import Tuple, Union, Sequence, array, Optional, Literal, dtype, device

def einop(xp, prgm, **kwargs):
    """Execute an einsum-like expression using the specified array framework.

    This function parses and executes einsum-like expressions with extended syntax
    that supports various operations beyond traditional Einstein summation notation.

    Args:
        xp: Array framework module (e.g., numpy, cupy, or other array library)
            that provides the underlying array operations.
        prgm (str): Einsum program string specifying the computation. The syntax
            supports:
            - Assignment: "C[i,j] = A[i,j] + B[j,i]"
            - Increment: "C[i,j] += A[i,k] * B[k,j]"
            - Reductions: "C[i] += A[i,j]", "C[i] max= A[i,j]", "C[i] &= A[i,j]"
            - Arithmetic operations: +, -, *, /, //, %, **
            - Comparison operations: ==, !=, <, <=, >, >=
            - Logical operations: and, or, not
            - Bitwise operations: &, |, ^, <<, >>
            - Function calls and complex expressions with parentheses
            - Mathematical functions: abs, sqrt, exp, log, sin, cos, tan, etc.
            - Literal values: integers, floats, booleans, and complex numbers
            - Python operator precedence and parentheses for grouping
        **kwargs: Named arrays referenced in the einsum expression. The keys
            should match the tensor names used in the program string.

    Returns:
        The result array from executing the einsum expression.

    Examples:
        >>> import numpy as np
        >>> A = np.random.rand(3, 4)
        >>> B = np.random.rand(4, 3)
        >>> # Matrix addition with transpose
        >>> C = einsum(np, "C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
        >>> # Matrix multiplication
        >>> D = einsum(np, "D[i,j] += A[i,k] * B[k,j]", A=A, B=B)
        >>> # Min-Plus multiplication with shift
        >>> E = einsum(np, "E[i] min= A[i,k] + D[k,j] << 1", A=A, D=D)
    """
    prgm = parse_einop(prgm)
    return prgm.run(xp, kwargs)