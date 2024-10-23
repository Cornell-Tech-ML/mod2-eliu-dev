"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Return the product of x and y.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        The product of x and y as a float.

    """
    return x * y


def id(x: float) -> float:
    """Return the identity of x.

    Args:
    ----
        x: A float.

    Returns:
    -------
        The input value (identity) as a float.

    """
    return x


def add(x: float, y: float) -> float:
    """Return the sum of x and y.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        The sum of x and y as a float.

    """
    return x + y


def neg(x: float) -> float:
    """Return the negative version of x.

    Args:
    ----
        x: A float.

    Returns:
    -------
        The negative version of x as a float.

    """
    return float(-x)


def lt(x: float, y: float) -> float:
    """Check if x is less than y.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A boolean indicating if x is less than y.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A boolean indicating if x is exactly equal to y.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Compare x and y and return the larger value.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        The maximum of x and y as a float.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close to y (within 1e-2).

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A boolean indicating if x is close to y.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Return the sigmoid of x. The calculation is split into two cases for numerical stability.

    Args:
    ----
        x: A float.

    Returns:
    -------
        The sigmoid of x as a float.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the ReLU of x. The Recitified Linear Unit (ReLU) function is used as an activation function in neural networks.

    Args:
    ----
        x: A float.

    Returns:
    -------
        The ReLU of x as a float.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Return the natural logarithm of x.

    Args:
    ----
        x: A float.

    Returns:
    -------
        The natural logarithm of x as a float.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Return the exponential of x.

    Args:
    ----
        x: A float.

    Returns:
    -------
        The exponential of x, exp(x), as a float.

    """
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Return the derivative log(x) times a second argument y.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        The derivative log(x) times a second argument y as a float.

    """
    return y / (x + EPS)


def inv(x: float) -> float:
    """Return the reciprocal of x.

    Args:
    ----
        x: A float.

    Returns:
    -------
        The reciprocal of x as a float.

    """
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Return the derivative of the reciprocal of x times a second argument y.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        The derivative 1/x times a second argument y as a float.

    """
    return (-1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Return the derivative of the ReLU function times a second argument y.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        The derivative of the ReLU function times a second argument y as a float.

    """
    if x > 0:
        return y
    else:
        return 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher Order Function that returns a map function using the Callable function fn.

    Args:
    ----
        fn: A Callable function that takes a float and returns a float. Used to map over the list of elements.
        ls: A list of floats to apply the Callable function to.

    Returns:
    -------
        A list of floats.

    """

    def apply(ls: Iterable[float]) -> Iterable[float]:
        result: list[float] = []
        for element in ls:
            result.append(fn(element))
        return result

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher Order Function that combines two iterables using a given function.

    Args:
    ----
        fn: A Callable function that combines two iterables of floats into an iterable of floats.
        a: An iterable of floats.
        b: An iterable of floats.

    Returns:
    -------
        A Callable function that takes two iterables and combines them into a single iterable of floats.

    """

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        result = []
        for x, y in zip(ls1, ls2):
            result.append(fn(x, y))
        return result

    return apply


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce a list to a single value by applying a function to the elements of the list.

    Args:
    ----
        fn: A Callable function that takes an iterable of floats and returns a float. Used to reduce the list to a single value.
        start: The starting value for the reduction.

    Returns:
    -------
        A Callable function that takes an iterable of floats and returns a float.

    """

    def apply(ls: Iterable[float]) -> float:
        # Ensure the list is not empty. We use a bool check to account for the fact that Iterables may not support len().
        val = start
        for element in ls:
            val = fn(val, element)
        return val

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Return the negative version of all elements in a list using map.

    Args:
    ----
        ls: A list of floats to return the negative version of.

    Returns:
    -------
        A list of the negative versions of the input floats.

    """
    return map(neg)(ls)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add two lists using zipWith.

    Args:
    ----
        a: A list of floats.
        b: A list of floats.

    Returns:
    -------
        A list of floats that are the element-by-element sum of the input lists.

    """
    return zipWith(add)(a, b)


def sum(ls: Iterable[float]) -> float:
    """Sum a list using reduce.

    Args:
    ----
        ls: A list of floats to sum.

    Returns:
    -------
        The sum of the input list as a float.

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Take the product of a list using reduce.

    Args:
    ----
        ls: A list of floats.

    Returns:
    -------
        The product of the input list as a float.

    """
    return reduce(mul, 1.0)(ls)
