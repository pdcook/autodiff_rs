use crate::traits::{InstOne, InstZero};
use core::ops::{Add, Div, Mul, Neg, Sub};
use num::traits::pow::Pow;

// supertrait for self arithmetic operations
// T + T -> T
pub trait Arithmetic<'a>:
    'a
    + Add<Self, Output = Self> // T + T -> T
    + Sub<Self, Output = Self> // T - T -> T
    + Mul<Self, Output = Self> // T * T -> T
    + Div<Self, Output = Self> // T / T -> T
    + Neg<Output = Self> // -T -> T
    + Sized // T: Sized
    + InstOne // T::one()
    + InstZero // T::zero()
    + Clone // T: Clone
    + PartialEq // T == T
    + Add<&'a Self, Output = Self> // T + &T -> T
    + Sub<&'a Self, Output = Self> // T - &T -> T
    + Mul<&'a Self, Output = Self> // T * &T -> T
    + Div<&'a Self, Output = Self> // T / &T -> T
where
    &'a Self: CastingArithmetic<'a, Self, Self> // &T + T -> T and &T + &T -> T
{
}

// trait for strong associated arithmetic operations
// e.g. T + A -> T
pub trait StrongAssociatedArithmetic<'a, A: 'a>:
    Arithmetic<'a>
    + Add<A, Output = Self> // T + A -> T
    + Sub<A, Output = Self> // T - A -> T
    + Mul<A, Output = Self> // T * A -> T
    + Div<A, Output = Self> // T / A -> T
    + Add<&'a A, Output = Self> // T + &A -> T
    + Sub<&'a A, Output = Self> // T - &A -> T
    + Mul<&'a A, Output = Self> // T * &A -> T
    + Div<&'a A, Output = Self> // T / &A -> T
where
    &'a Self: CastingArithmetic<'a, Self, Self> + CastingArithmetic<'a, A, Self> // &T + A -> T and &T + &A -> T
{
}

// trait for weak associated arithmetic operations
// e.g. T + A -> A
pub trait WeakAssociatedArithmetic<'a, A: 'a>:
    Arithmetic<'a>
    + Add<A, Output = A> // T + A -> A
    + Sub<A, Output = A> // T - A -> A
    + Mul<A, Output = A> // T * A -> A
    + Div<A, Output = A> // T / A -> A
    + Add<&'a A, Output = A> // T + &A -> A
    + Sub<&'a A, Output = A> // T - &A -> A
    + Mul<&'a A, Output = A> // T * &A -> A
    + Div<&'a A, Output = A> // T / &A -> A
where
    &'a Self: CastingArithmetic<'a, Self, Self> + CastingArithmetic<'a, A, A> // &T + A -> A and &T + &A -> A
{
}

// trait for casting arithmetic operations
// e.g. T + A -> B and T + &A -> B
pub trait CastingArithmetic<'a, A: 'a, B: 'a>:
    Add<A, Output = B> // T + A -> B
    + Sub<A, Output = B> // T - A -> B
    + Mul<A, Output = B> // T * A -> B
    + Div<A, Output = B> // T / A -> B
    + Add<&'a A, Output = B> // T + &A -> B
    + Sub<&'a A, Output = B> // T - &A -> B
    + Mul<&'a A, Output = B> // T * &A -> B
    + Div<&'a A, Output = B> // T / &A -> B
{
}

// supertraits for more advanced arithmetic operations
pub trait ExtendedArithmetic<'a>:
    Arithmetic<'a>
    + Pow<Self, Output = Self> // T.pow(T) -> T
    + Pow<&'a Self, Output = Self> // T.pow(&T) -> T
where
    &'a Self: CastingArithmetic<'a, Self, Self>, // &T + T -> T and &T + &T -> T
{
}

// trait for strong extended arithmetic operations
pub trait StrongAssociatedExtendedArithmetic<'a, A: 'a>:
    ExtendedArithmetic<'a>
    + StrongAssociatedArithmetic<'a, A>
    + Pow<A, Output = Self>
    + Pow<&'a A, Output = Self>
where
    &'a Self: CastingArithmetic<'a, Self, Self> + CastingArithmetic<'a, A, Self>,
{
}

// trait for weak extended arithmetic operations
pub trait WeakAssociatedExtendedArithmetic<'a, A: 'a>:
    ExtendedArithmetic<'a>
    + WeakAssociatedArithmetic<'a, A>
    + Pow<A, Output = A>
    + Pow<&'a A, Output = A>
where
    &'a Self: CastingArithmetic<'a, Self, Self> + CastingArithmetic<'a, A, A>,
{
}

// blanket implementation for arithmetic operations
// in order to use the supertraits like trait aliases

impl<
        'a,
        T: 'a
            + Add<T, Output = T>
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Div<T, Output = T>
            + Neg<Output = T>
            + Sized
            + InstOne
            + InstZero
            + Clone
            + PartialEq
            + Add<&'a T, Output = T>
            + Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    > Arithmetic<'a> for T
where
    &'a T: CastingArithmetic<'a, T, T>,
{
}

impl<
        'a,
        T: Arithmetic<'a>
            + Add<A, Output = T>
            + Sub<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Add<&'a A, Output = T>
            + Sub<&'a A, Output = T>
            + Mul<&'a A, Output = T>
            + Div<&'a A, Output = T>,
        A: 'a,
    > StrongAssociatedArithmetic<'a, A> for T
where
    &'a T: CastingArithmetic<'a, A, T> + CastingArithmetic<'a, T, T>,
{
}

impl<
        'a,
        T: Arithmetic<'a>
            + Add<A, Output = A>
            + Sub<A, Output = A>
            + Mul<A, Output = A>
            + Div<A, Output = A>
            + Add<&'a A, Output = A>
            + Sub<&'a A, Output = A>
            + Mul<&'a A, Output = A>
            + Div<&'a A, Output = A>,
        A: 'a,
    > WeakAssociatedArithmetic<'a, A> for T
where
    &'a T: CastingArithmetic<'a, A, A> + CastingArithmetic<'a, T, T>,
{
}

impl<
        'a,
        T: 'a
            + Add<A, Output = B>
            + Sub<A, Output = B>
            + Mul<A, Output = B>
            + Div<A, Output = B>
            + Add<&'a A, Output = B>
            + Sub<&'a A, Output = B>
            + Mul<&'a A, Output = B>
            + Div<&'a A, Output = B>,
        A: 'a,
        B: 'a,
    > CastingArithmetic<'a, A, B> for T
{
}

impl<'a, T: Arithmetic<'a> + Pow<T, Output = T> + Pow<&'a T, Output = T>> ExtendedArithmetic<'a>
    for T
where
    &'a T: CastingArithmetic<'a, T, T>,
{
}

impl<
        'a,
        T: ExtendedArithmetic<'a>
            + StrongAssociatedArithmetic<'a, A>
            + Pow<A, Output = T>
            + Pow<&'a A, Output = T>,
        A: 'a,
    > StrongAssociatedExtendedArithmetic<'a, A> for T
where
    &'a T: CastingArithmetic<'a, A, T> + CastingArithmetic<'a, T, T>,
{
}

impl<
        'a,
        T: ExtendedArithmetic<'a>
            + WeakAssociatedArithmetic<'a, A>
            + Pow<A, Output = A>
            + Pow<&'a A, Output = A>,
        A: 'a,
    > WeakAssociatedExtendedArithmetic<'a, A> for T
where
    &'a T: CastingArithmetic<'a, A, A> + CastingArithmetic<'a, T, T>,
{
}
