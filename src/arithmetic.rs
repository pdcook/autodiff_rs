use core::ops::{Add, Div, Mul, Neg, Sub};
use num::traits::identities::{One, Zero};
use num::traits::pow::Pow;

// supertrait for self arithmetic operations
// T + T -> T
pub trait Arithmetic:
    Add<Self, Output = Self> // T + T -> T
    + Sub<Self, Output = Self> // T - T -> T
    + Mul<Self, Output = Self> // T * T -> T
    + Div<Self, Output = Self> // T / T -> T
    + Neg<Output = Self> // -T -> T
    + Sized // T: Sized
    + One // T::one()
    + Zero // T::zero()
    + Copy // T: Copy
    + Clone // T: Clone
    + PartialEq // T == T
{
}

// trait for strong associated arithmetic operations
// e.g. T + A -> T
pub trait StrongAssociatedArithmetic<A>:
    Arithmetic
    + Add<A, Output = Self> // T + A -> T
    + Sub<A, Output = Self> // T - A -> T
    + Mul<A, Output = Self> // T * A -> T
    + Div<A, Output = Self> // T / A -> T
{
}

// trait for weak associated arithmetic operations
// e.g. T + A -> A
pub trait WeakAssociatedArithmetic<A>:
    Arithmetic
    + Add<A, Output = A> // T + A -> A
    + Sub<A, Output = A> // T - A -> A
    + Mul<A, Output = A> // T * A -> A
    + Div<A, Output = A> // T / A -> A
{
}

// supertraits for more advanced arithmetic operations
pub trait ExtendedArithmetic:
    Arithmetic
    + Pow<Self, Output = Self> // T.pow(T) -> T
{
}

// trait for strong extended arithmetic operations
pub trait StrongAssociatedExtendedArithmetic<A>:
    ExtendedArithmetic + StrongAssociatedArithmetic<A> + Pow<A, Output = Self>
{
}

// trait for weak extended arithmetic operations
pub trait WeakAssociatedExtendedArithmetic<A>:
    ExtendedArithmetic + WeakAssociatedArithmetic<A> + Pow<A, Output = A>
{
}

// blanket implementation for arithmetic operations
// in order to use the supertraits like trait aliases

impl<
        T: Add<T, Output = T>
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Div<T, Output = T>
            + Neg<Output = T>
            + Sized
            + One
            + Zero
            + Copy
            + Clone
            + PartialEq,
    > Arithmetic for T
{
}

impl<
        T: Arithmetic
            + Add<A, Output = T>
            + Sub<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>,
        A,
    > StrongAssociatedArithmetic<A> for T
{
}

impl<
        T: Arithmetic
            + Add<A, Output = A>
            + Sub<A, Output = A>
            + Mul<A, Output = A>
            + Div<A, Output = A>,
        A,
    > WeakAssociatedArithmetic<A> for T
{
}

impl<
        T: Arithmetic
            + Pow<T, Output = T>
            + Add<T, Output = T>
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Div<T, Output = T>
            + Neg<Output = T>,
    > ExtendedArithmetic for T
{
}

impl<
        T: ExtendedArithmetic
            + Add<A, Output = T>
            + Sub<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Pow<A, Output = T>,
        A,
    > StrongAssociatedExtendedArithmetic<A> for T
{
}

impl<
        T: ExtendedArithmetic
            + Add<A, Output = A>
            + Sub<A, Output = A>
            + Mul<A, Output = A>
            + Div<A, Output = A>
            + Pow<A, Output = A>,
        A,
    > WeakAssociatedExtendedArithmetic<A> for T
{
}
