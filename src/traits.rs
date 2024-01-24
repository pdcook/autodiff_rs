use num::complex::Complex;
use num::rational::Ratio;
use num::{Integer, Num, One, Zero};
use std::num::Wrapping;
use std::ops::{Add, Mul};
use crate::gradienttype::GradientType;

pub trait InstZero: Sized + Add<Self, Output = Self> {
    // required methods
    fn zero(&self) -> Self;

    fn is_zero(&self) -> bool;

    // provided methods
    fn set_zero(&mut self) {
        *self = self.zero();
    }
}

pub trait InstOne: Sized + Mul<Self, Output = Self> {
    // required methods
    /// Returns the multiplicative identity of Self, 1.
    /// i.e. self * self.one() == self
    fn one(&self) -> Self;


    // provided methods
    fn set_one(&mut self) {
        *self = self.one();
    }

    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        *self == self.one()
    }
}

pub trait GradientIdentity: Sized
where
    Self: GradientType<Self>
{
    /// Returns the gradient identity for a function with input Self,
    /// output Self, and gradient type `<Self as GradientType<Self>>::GradientType`
    /// for primitive types, this is the same as one()
    /// for Array types, this is more complicated
    ///
    /// This cannot be implemented for complex numbers
    fn grad_identity(&self) -> <Self as GradientType<Self>>::GradientType;
}

// implementation for InstZero for all the types that implement Zero from num
// u32, i128, i16, u128, f64, usize, i32, i8, f32, i64, u16, Wrapping<T: Zero>, isize, u8, u64,
// BigInt, BigUint, Ratio<T: Integer>, Complex<T: Num>

// a macro to do this for any type that can call ::zero()
macro_rules! impl_zero {
    ($($t:ty),*) => ($(
        impl InstZero for $t {
            fn zero(&self) -> $t {
                <$t as Zero>::zero()
            }

            fn is_zero(&self) -> bool {
                <$t as Zero>::is_zero(self)
            }
        }
    )*)
}

impl_zero!(u32, i128, i16, u128, f64, usize, i32, i8, f32, i64, u16, isize, u8, u64);
impl_zero!(num::BigInt, num::BigUint);

// generic implementations done here
impl<T> InstZero for Wrapping<T>
where
    T: Zero,
    Wrapping<T>: Add<Wrapping<T>, Output = Wrapping<T>>,
{
    fn zero(&self) -> Wrapping<T> {
        <Wrapping<T> as Zero>::zero()
    }

    fn is_zero(&self) -> bool {
        <Wrapping<T> as Zero>::is_zero(self)
    }
}
impl<T> InstZero for Ratio<T>
where
    T: Clone + Integer,
{
    fn zero(&self) -> Ratio<T> {
        <Ratio<T> as Zero>::zero()
    }

    fn is_zero(&self) -> bool {
        <Ratio<T> as Zero>::is_zero(self)
    }
}
impl<T> InstZero for Complex<T>
where
    T: Clone + Num,
{
    fn zero(&self) -> Complex<T> {
        <Complex<T> as Zero>::zero()
    }

    fn is_zero(&self) -> bool {
        <Complex<T> as Zero>::is_zero(self)
    }
}

// implementation for InstOne for all the types that implement One from num
// Wrapping<T: One>, i64, u128, f32, u16, u32, i16, f64, isize, i32, u8, u64, usize, i128, i8
// BigInt, BigUint, Ratio<T: Integer>, Complex<T: Num>

// a macro to do this for any type that can call ::one()
macro_rules! impl_one {
    ($($t:ty),*) => ($(
        impl InstOne for $t {
            fn one(&self) -> $t {
                <$t as One>::one()
            }
        }
    )*)
}

impl_one!(i64, u128, f32, u16, u32, i16, f64, isize, i32, u8, u64, usize, i128, i8);
impl_one!(num::BigInt, num::BigUint);

// generic implementations done here
impl<T> InstOne for Wrapping<T>
where
    T: One,
    Wrapping<T>: Mul<Wrapping<T>, Output = Wrapping<T>>,
{
    fn one(&self) -> Wrapping<T> {
        <Wrapping<T> as One>::one()
    }
}

impl<T> InstOne for Ratio<T>
where
    T: Clone + Integer,
{
    fn one(&self) -> Ratio<T> {
        <Ratio<T> as One>::one()
    }
}

impl<T> InstOne for Complex<T>
where
    T: Clone + Num,
{
    fn one(&self) -> Complex<T> {
        <Complex<T> as One>::one()
    }
}

// macro for implementing gradient identity for primitive types that implement InstOne
macro_rules! impl_grad_identity {
    ($($t:ty),*) => ($(
        impl<G> GradientIdentity for $t
        where
            Self: GradientType<Self, GradientType = G>,
            G: One,
        {
            fn grad_identity(&self) -> G
            {
                G::one()
            }
        }
    )*)
}

impl_grad_identity!(i64, u128, f32, u16, u32, i16, f64, isize, i32, u8, u64, usize, i128, i8);
impl_grad_identity!(num::BigInt, num::BigUint);

// generic implementations done here
impl<T, G> GradientIdentity for Wrapping<T>
where
    Self: GradientType<Self, GradientType = G>,
    G: One,
{
    fn grad_identity(&self) -> G {
        G::one()
    }
}

impl<T, G> GradientIdentity for Ratio<T>
where
    Self: GradientType<Self, GradientType = G>,
    T: Clone + Integer,
    G: One,
{
    fn grad_identity(&self) -> G {
        G::one()
    }
}

impl<T, G> GradientIdentity for Complex<T>
where
    Self: GradientType<Self, GradientType = G>,
    T: Clone + Num,
    G: One,
{
    fn grad_identity(&self) -> G {
        G::one()
    }
}
