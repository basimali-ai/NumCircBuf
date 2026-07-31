from typing import TypeVar, Union

import numpy as np

Scalar = Union[np.floating, np.integer]
ConcreteFloating = Union[np.float64, np.float32]
ConcreteInt = Union[np.int64, np.int32]
ConcreteUint = Union[np.uint64, np.uint32]
ConcreteInteger = Union[ConcreteInt, ConcreteUint]
ConcreteScalar = Union[ConcreteFloating, ConcreteInteger]

ShapeT = TypeVar("ShapeT", bound=tuple[int, ...])
ScalarT = TypeVar("ScalarT", bound=Scalar)

ConcreteFloatingT = TypeVar("ConcreteFloatingT", bound=ConcreteFloating)
ConcreteIntT = TypeVar("ConcreteIntT", bound=ConcreteInt)
ConcreteUintT = TypeVar("ConcreteUintT", bound=ConcreteUint)
ConcreteIntegerT = TypeVar("ConcreteIntegerT", bound=ConcreteInteger)
ConcreteScalarT = TypeVar("ConcreteScalarT", bound=ConcreteScalar)
ConcreteScalarT_co = TypeVar("ConcreteScalarT_co", bound=ConcreteScalar, covariant=True)
