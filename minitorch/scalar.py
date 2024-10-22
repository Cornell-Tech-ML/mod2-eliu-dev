from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(init=False, default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        """Representation function

        Args:
        ----
            None

        Returns:
        -------
            String representation of the Scalar

        """
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiplication function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Multiplication of self and b as a Scalar

        """
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Division function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Division of self and b as a Scalar

        """
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Right division function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Division of b and self as a Scalar

        """
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        """Addition function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Addition of self and b as a Scalar

        """
        return Add.apply(self, b)

    def __bool__(self) -> bool:
        """Boolean function

        Args:
        ----
            None

        Returns:
        -------
            Boolean of self as a Scalar

        """
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Less than function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Scalar: 1.0 if self is less than b, else 0.0

        """
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Greater than function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Scalar: 1.0 if self is greater than b, else 0.0

        """
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        """Equal function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Scalar: 1.0 if self is equal to b, else 0.0

        """
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtraction function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Subtraction of self and b as a Scalar

        """
        return Add.apply(self, -b)

    def __neg__(self) -> Scalar:
        """Negation function

        Args:
        ----
            None

        Returns:
        -------
            Negation of self as a Scalar

        """
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Right addition function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Addition of self and b as a Scalar

        """
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Right multiplication function

        Args:
        ----
            b: ScalarLike

        Returns:
        -------
            Multiplication of self and b as a Scalar

        """
        return self * b

    def log(self) -> Scalar:
        """Logarithm function

        Args:
        ----
            None

        Returns:
        -------
            Logarithm of self as a Scalar

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Exponential function

        Args:
        ----
            None

        Returns:
        -------
            Exponential of self as a Scalar

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Sigmoid function

        Args:
        ----
            None

        Returns:
        -------
            Sigmoid of self as a Scalar

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """ReLU function

        Args:
        ----
            None

        Returns:
        -------
            ReLU of self as a Scalar

        """
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is constant (no `history`)

        Args:
        ----
            None

        Returns:
        -------
            True if this variable is constant

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Parents of this variable

        Args:
        ----
            None

        Returns:
        -------
            Parents of this variable

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Chain rule for backpropagation

        Args:
        ----
            d_output: Any

        Returns:
        -------
            Iterable of tuples of Variable and Any

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        derivatives = h.last_fn._backward(h.ctx, d_output)
        return list(zip(h.inputs, derivatives))

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        Returns:
        -------
            None

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.

    Returns:
    -------
        None

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
