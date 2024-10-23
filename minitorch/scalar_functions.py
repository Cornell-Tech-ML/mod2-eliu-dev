from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple

    Args:
    ----
        x: Value to wrap

    Returns:
    -------
        Tuple

    """
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        """Apply the scalar function to the given values.

        Args:
        ----
            *vals: ScalarLike

        Returns:
        -------
            Scalar

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two scalars

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: float
            b: float

        Returns:
        -------
            Sum of a and b

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Add backward

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: Derivative of the output

        Returns:
        -------
            Derivative of the input

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Log the scalar

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: Scalar to log

        Returns:
        -------
            Logged scalar

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Log backward

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: Derivative of the output

        Returns:
        -------
            Derivative of the input

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return the product of a and b.

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: float
            b: float

        Returns:
        -------
            The product of a and b as a float.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Return the derivatives of a and b with respect to the output.

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: float

        Returns:
        -------
            The derivatives of a and b with respect to the output as a tuple.

        """
        (a, b) = ctx.saved_values
        dx_a = d_output * b
        dx_b = d_output * a
        return (dx_a, dx_b)


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the inverse of a.

        Args:
        ----
            ctx: Context to store information during the forward pass
        a: float

        Returns:
        -------
            The inverse of a as a float.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of a with respect to the output.

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: float

        Returns:
        -------
            The derivative of a with respect to the output as a float.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the negation of a.

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: float

        Returns:
        -------
            The negation of a as a float.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of a with respect to the output.

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: float

        Returns:
        -------
            The derivative of a with respect to the output as a float.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the sigmoid of a.

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: float

        Returns:
        -------
            The sigmoid of a as a float.

        """
        sig = operators.sigmoid(a)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of a with respect to the output.

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: float

        Returns:
        -------
            The derivative of a with respect to the output as a float.

        """
        (sig,) = ctx.saved_values
        d_sig = sig * (1.0 - sig)
        return float(d_output * d_sig)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the ReLU of a.

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: float

        Returns:
        -------
            The ReLU of a as a float.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of a with respect to the output.

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: float

        Returns:
        -------
            The derivative of a with respect to the output as a float.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the exponential of a.

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: float

        Returns:
        -------
            The exponential of a as a float.

        """
        exp_a = operators.exp(a)
        ctx.save_for_backward(exp_a)
        return exp_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of a with respect to the output.

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: float

        Returns:
        -------
            The derivative of a with respect to the output as a float.

        """
        (exp_a,) = ctx.saved_values
        return float(d_output * exp_a)


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1.0 if a is less than b, else return 0.0.

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: float
            b: float

        Returns:
        -------
            The result of the less-than comparison as a float.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Return the derivative of a with respect to the output.

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: float

        Returns:
        -------
            0 as a float because the derivative of a less-than comparison is the derivative of a constant.

        """
        return (0.0, 0.0)


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1.0 if a is equal to b, else return 0.0.

        Args:
        ----
            ctx: Context to store information during the forward pass
            a: float
            b: float

        Returns:
        -------
            The result of the equality comparison as a float.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Return the derivative of a with respect to the output.

        Args:
        ----
            ctx: Context to store information during the forward pass
            d_output: float

        Returns:
        -------
            0 as a float because the derivative of an equality comparison is the derivative of a constant.

        """
        return (0.0, 0.0)
