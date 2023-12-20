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
    + Clone // T: Clone
    + PartialEq // T == T
    + for<'a> Add<&'a Self, Output = Self> // T + &T -> T
    + for<'a> Sub<&'a Self, Output = Self> // T - &T -> T
    + for<'a> Mul<&'a Self, Output = Self> // T * &T -> T
    + for<'a> Div<&'a Self, Output = Self> // T / &T -> T
where
    for<'a> &'a Self: CastingArithmetic<Self, Self> // &T + T -> T and &T + &T -> T
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
    + for<'a> Add<&'a A, Output = Self> // T + &A -> T
    + for<'a> Sub<&'a A, Output = Self> // T - &A -> T
    + for<'a> Mul<&'a A, Output = Self> // T * &A -> T
    + for<'a> Div<&'a A, Output = Self> // T / &A -> T
where
    for<'a> &'a Self: CastingArithmetic<Self, Self> + CastingArithmetic<A, Self> // &T + A -> T and &T + &A -> T
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
    + for<'a> Add<&'a A, Output = A> // T + &A -> A
    + for<'a> Sub<&'a A, Output = A> // T - &A -> A
    + for<'a> Mul<&'a A, Output = A> // T * &A -> A
    + for<'a> Div<&'a A, Output = A> // T / &A -> A
where
    for<'a> &'a Self: CastingArithmetic<Self, Self> + CastingArithmetic<A, A> // &T + A -> A and &T + &A -> A
{
}

// trait for casting arithmetic operations
// e.g. T + A -> B and T + &A -> B
pub trait CastingArithmetic<A, B>:
    Add<A, Output = B> // T + A -> B
    + Sub<A, Output = B> // T - A -> B
    + Mul<A, Output = B> // T * A -> B
    + Div<A, Output = B> // T / A -> B
    + for<'a> Add<&'a A, Output = B> // T + &A -> B
    + for<'a> Sub<&'a A, Output = B> // T - &A -> B
    + for<'a> Mul<&'a A, Output = B> // T * &A -> B
    + for<'a> Div<&'a A, Output = B> // T / &A -> B
{
}

// supertraits for more advanced arithmetic operations
pub trait ExtendedArithmetic:
    Arithmetic
    + Pow<Self, Output = Self> // T.pow(T) -> T
    + for<'a> Pow<&'a Self, Output = Self> // T.pow(&T) -> T
where
    for<'a> &'a Self: CastingArithmetic<Self, Self>, // &T + T -> T and &T + &T -> T
{
}

// trait for strong extended arithmetic operations
pub trait StrongAssociatedExtendedArithmetic<A>:
    ExtendedArithmetic
    + StrongAssociatedArithmetic<A>
    + Pow<A, Output = Self>
    + for<'a> Pow<&'a A, Output = Self>
where
    for<'a> &'a Self: CastingArithmetic<Self, Self> + CastingArithmetic<A, Self>,
{
}

// trait for weak extended arithmetic operations
pub trait WeakAssociatedExtendedArithmetic<A>:
    ExtendedArithmetic
    + WeakAssociatedArithmetic<A>
    + Pow<A, Output = A>
    + for<'a> Pow<&'a A, Output = A>
where
    for<'a> &'a Self: CastingArithmetic<Self, Self> + CastingArithmetic<A, A>,
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
            + Clone
            + PartialEq
            + for<'a> Add<&'a T, Output = T>
            + for<'a> Sub<&'a T, Output = T>
            + for<'a> Mul<&'a T, Output = T>
            + for<'a> Div<&'a T, Output = T>,
    > Arithmetic for T
where
    for<'a> &'a T: CastingArithmetic<T, T>,
{
}

impl<
        T: Arithmetic
            + Add<A, Output = T>
            + Sub<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + for<'a> Add<&'a A, Output = T>
            + for<'a> Sub<&'a A, Output = T>
            + for<'a> Mul<&'a A, Output = T>
            + for<'a> Div<&'a A, Output = T>,
        A,
    > StrongAssociatedArithmetic<A> for T
where
    for<'a> &'a T: CastingArithmetic<A, T> + CastingArithmetic<T, T>,
{
}

impl<
        T: Arithmetic
            + Add<A, Output = A>
            + Sub<A, Output = A>
            + Mul<A, Output = A>
            + Div<A, Output = A>
            + for<'a> Add<&'a A, Output = A>
            + for<'a> Sub<&'a A, Output = A>
            + for<'a> Mul<&'a A, Output = A>
            + for<'a> Div<&'a A, Output = A>,
        A,
    > WeakAssociatedArithmetic<A> for T
where
    for<'a> &'a T: CastingArithmetic<A, A> + CastingArithmetic<T, T>,
{
}

impl<
        T: Add<A, Output = B>
            + Sub<A, Output = B>
            + Mul<A, Output = B>
            + Div<A, Output = B>
            + for<'a> Add<&'a A, Output = B>
            + for<'a> Sub<&'a A, Output = B>
            + for<'a> Mul<&'a A, Output = B>
            + for<'a> Div<&'a A, Output = B>,
        A: Arithmetic,
        B: Arithmetic,
    > CastingArithmetic<A, B> for T
where
    for<'a> &'a A: CastingArithmetic<A, A>,
    for<'a> &'a B: CastingArithmetic<B, B>,
{
}

impl<T: Arithmetic + Pow<T, Output = T> + for<'a> Pow<&'a T, Output = T>> ExtendedArithmetic for T where
    for<'a> &'a T: CastingArithmetic<T, T>
{
}

impl<
        T: ExtendedArithmetic
            + StrongAssociatedArithmetic<A>
            + Pow<A, Output = T>
            + for<'a> Pow<&'a A, Output = T>,
        A,
    > StrongAssociatedExtendedArithmetic<A> for T
where
    for<'a> &'a T: CastingArithmetic<A, T> + CastingArithmetic<T, T>,
{
}

impl<
        T: ExtendedArithmetic
            + WeakAssociatedArithmetic<A>
            + Pow<A, Output = A>
            + for<'a> Pow<&'a A, Output = A>,
        A,
    > WeakAssociatedExtendedArithmetic<A> for T
where
    for<'a> &'a T: CastingArithmetic<A, A> + CastingArithmetic<T, T>,
{
}
