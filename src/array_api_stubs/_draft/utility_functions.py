__all__ = ["all", "any", "diff", "replace"]


from ._types import Optional, Tuple, Union, array


def all(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    """
    Tests whether all input array elements evaluate to ``True`` along a specified axis.

    Parameters
    ----------
    x: array
        input array.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which to perform a logical AND reduction. By default, a logical AND reduction **must** be performed over the entire array. If a tuple of integers, logical AND reductions **must** be performed over multiple axes. A valid ``axis`` **must** be an integer on the interval ``[-N, N)``, where ``N`` is the number of axes in ``x``. If an ``axis`` is specified as a negative integer, the function **must** determine the axis along which to perform a reduction by counting backward from the last axis (where ``-1`` refers to the last axis). If provided an invalid ``axis``, the function **must** raise an exception. Default: ``None``.
    keepdims: bool
        If ``True``, the reduced axes **must** be included in the result as singleton dimensions, and, accordingly, the result **must** be broadcast-compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes **must not** be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if a logical AND reduction was performed over the entire array, the returned array **must** be a zero-dimensional array containing the test result. Otherwise, the returned array **must** be a non-zero-dimensional array containing the test results. The returned array **must** have a data type of ``bool``.

    Notes
    -----

    -   Positive infinity, negative infinity, and NaN **must** evaluate to ``True``.
    -   If ``x`` has a complex floating-point data type, elements having a non-zero component (real or imaginary) **must** evaluate to ``True``.
    -   If ``x`` is an empty array or the size of the axis along which to evaluate elements is zero, the test result **must** be ``True``.

    .. versionchanged:: 2022.12
       Added complex data type support.
    """


def any(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    """
    Tests whether any input array element evaluates to ``True`` along a specified axis.

    Parameters
    ----------
    x: array
        input array.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which to perform a logical OR reduction. By default, a logical OR reduction **must** be performed over the entire array. If a tuple of integers, logical OR reductions **must** be performed over multiple axes. A valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N`` is the number of axes in ``x``. If an ``axis`` is specified as a negative integer, the function **must** determine the axis along which to perform a reduction by counting backward from the last dimension (where ``-1`` refers to the last axis). If provided an invalid ``axis``, the function **must** raise an exception. Default: ``None``.
    keepdims: bool
        If ``True``, the reduced axes **must** be included in the result as singleton dimensions, and, accordingly, the result **must** be broadcast-compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes **must not** be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if a logical OR reduction was performed over the entire array, the returned array **must** be a zero-dimensional array containing the test result. Otherwise, the returned array **must** be a non-zero-dimensional array containing the test results. The returned array **must** have a data type of ``bool``.

    Notes
    -----

    -   Positive infinity, negative infinity, and NaN **must** evaluate to ``True``.
    -   If ``x`` has a complex floating-point data type, elements having a non-zero component (real or imaginary) **must** evaluate to ``True``.
    -   If ``x`` is an empty array or the size of the axis along which to evaluate elements is zero, the test result **must** be ``False``.

    .. versionchanged:: 2022.12
       Added complex data type support.
    """


def diff(
    x: array,
    /,
    *,
    axis: int = -1,
    n: int = 1,
    prepend: Optional[array] = None,
    append: Optional[array] = None,
) -> array:
    """
    Calculates the n-th discrete forward difference along a specified axis.

    Parameters
    ----------
    x: array
        input array. **Should** have a numeric data type.
    axis: int
        axis along which to compute differences. A valid ``axis`` **must** be an integer on the interval ``[-N, N)``, where ``N`` is the number of axes in ``x``. If an ``axis`` is specified as a negative integer, the function **must** determine the axis along which to compute differences by counting backward from the last axis (where ``-1`` refers to the last axis). If provided an invalid ``axis``, the function **must** raise an exception. Default: ``-1``.
    n: int
        number of times to recursively compute differences. Default: ``1``.
    prepend: Optional[array]
        values to prepend to a specified axis prior to computing differences. **Must** have the same shape as ``x``, except for the axis specified by ``axis`` which can have any size. **Should** have the same data type as ``x``. Default: ``None``.
    append: Optional[array]
        values to append to a specified axis prior to computing differences. **Must** have the same shape as ``x``, except for the axis specified by ``axis`` which can have any size. **Should** have the same data type as ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array containing the n-th differences. **Should** have the same data type as ``x``. **Must** have the same shape as ``x``, except for the axis specified by ``axis`` which **must** have a size determined as follows:

        -   Let ``M`` be the number of elements along an axis specified by ``axis``.
        -   Let ``N1`` be the number of prepended values along an axis specified by ``axis``.
        -   Let ``N2`` be the number of appended values along an axis specified by ``axis``.
        -   The final size of the axis specified by ``axis`` **must** be ``M + N1 + N2 - n``.

    Notes
    -----

    -   The first-order differences are given by ``out[i] = x[i+1] - x[i]`` along a specified axis. Higher-order differences **must** be calculated recursively (e.g., by calling ``diff(out, axis=axis, n=n-1)``).
    -   If a conforming implementation chooses to support ``prepend`` and ``append`` arrays which have a different data type than ``x``, behavior is unspecified and thus implementation-defined. Implementations **may** choose to type promote (:ref:`type-promotion`), cast ``prepend`` and/or ``append`` to the same data type as ``x``, or raise an exception.

    .. versionadded:: 2024.12
    """


def replace(
    x: array,
    old: Union[bool, int, float, complex],
    new: Union[bool, int, float, complex],
    /,
) -> array:
    """
    Replaces each element of an input array ``x`` which is equal to a scalar value ``old`` with a scalar value ``new``.

    Parameters
    ----------
    x: array
        input array.
    old: Union[bool, int, float, complex]
        scalar value to replace. **Should** have a type and value compatible with the data type of ``x`` (see :ref:`mixing-scalars-and-arrays`).
    new: Union[bool, int, float, complex]
        replacement scalar value. **Should** have a type and value compatible with the data type of ``x`` (see :ref:`mixing-scalars-and-arrays`).

    Returns
    -------
    out: array
        an array in which each element ``x_i`` equal to ``old`` is replaced by ``new`` and all other elements are equal to the corresponding elements in ``x``. The returned array **must** have the same shape and data type as ``x``.

    Notes
    -----

    -   An element ``x_i`` **must** be replaced if ``x_i`` is equal to ``old`` (see :func:`~array_api.equal`) or if both ``x_i`` and ``old`` are ``NaN`` (see :func:`~array_api.isnan`).
    -   The function **must** consider every element of ``x``, including elements which are not explicitly stored (e.g., implicit fill values in a sparse array representation).
    -   The function **must not** modify ``x``.
    -   If ``old`` or ``new`` does not have a type and value compatible with the data type of ``x``, behavior is unspecified and thus implementation-defined.
    """
