use crate::gradienttype::GradientType;
use num::complex::Complex;
use num::rational::Ratio;
use num::{Float, Integer, Num, One, Zero};
use std::num::Wrapping;
use std::ops::{Add, Mul, Neg};

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
    Self: GradientType<Self>,
{
    /// Returns the gradient identity for a function with input Self,
    /// output Self, and gradient type `<Self as GradientType<Self>>::GradientType`
    /// for primitive types, this is the same as one()
    /// for Array types, this is more complicated
    ///
    /// This cannot be implemented for complex numbers
    fn grad_identity(&self) -> <Self as GradientType<Self>>::GradientType;
}

pub trait PossiblyComplex {
    fn is_always_real() -> bool;
}

pub trait Conjugate {
    type Output;
    fn conj(&self) -> Self::Output;
}

pub trait Abs {
    type Output;
    fn abs(self) -> Self::Output;
}

pub trait AbsSqr {
    type Output;
    fn abs_sqr(self) -> Self::Output;
}

pub trait Arg {
    type Output;
    fn arg(self) -> Self::Output;
}

pub trait Signum {
    type Output;
    fn signum(self) -> Self::Output;
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

// implementation of Complex traits for all real number types
macro_rules! impl_complex_traits_real_signed_copy {
    ($t:ty, $pi:expr) => {
        impl PossiblyComplex for $t {
            fn is_always_real() -> bool {
                true
            }
        }

        impl Conjugate for $t {
            type Output = Self;
            fn conj(&self) -> Self::Output {
                *self
            }
        }

        impl Abs for $t {
            type Output = Self;
            fn abs(self) -> Self::Output {
                <$t>::abs(self)
            }
        }

        impl AbsSqr for $t {
            type Output = Self;
            fn abs_sqr(self) -> Self::Output {
                self * self
            }
        }

        impl Signum for $t {
            type Output = Self;
            fn signum(self) -> Self::Output {
                <$t>::signum(self)
            }
        }

        impl Arg for $t {
            type Output = Self;
            fn arg(self) -> Self::Output {
                if self >= self.zero() {
                    self.zero() // ln(1) = 0
                } else {
                    $pi
                }
            }
        }
    };
}

macro_rules! impl_complex_traits_real_unsigned_copy {
    ($t:ty) => {
        impl PossiblyComplex for $t {
            fn is_always_real() -> bool {
                true
            }
        }

        impl Conjugate for $t {
            type Output = Self;
            fn conj(&self) -> Self::Output {
                *self
            }
        }

        impl Abs for $t {
            type Output = Self;
            fn abs(self) -> Self::Output {
                self
            }
        }

        impl AbsSqr for $t {
            type Output = Self;
            fn abs_sqr(self) -> Self::Output {
                self * self
            }
        }

        impl Signum for $t {
            type Output = Self;
            fn signum(self) -> Self::Output {
                if <Self as InstZero>::is_zero(&self) {
                    self
                } else {
                    self.one()
                }
            }
        }

        impl Arg for $t {
            type Output = Self;
            fn arg(self) -> Self::Output {
                self.zero()
            }
        }
    };
}
impl_complex_traits_real_signed_copy!(f32, std::f32::consts::PI);
impl_complex_traits_real_signed_copy!(f64, std::f64::consts::PI);
impl_complex_traits_real_signed_copy!(i64, 3_i64); // I don't even want to think about the implications of this
impl_complex_traits_real_signed_copy!(i32, 3_i32); //
impl_complex_traits_real_signed_copy!(i16, 3_i16); //
impl_complex_traits_real_signed_copy!(i8, 3_i8); //
impl_complex_traits_real_signed_copy!(isize, 3_isize); //
impl_complex_traits_real_unsigned_copy!(u64);
impl_complex_traits_real_unsigned_copy!(u32);
impl_complex_traits_real_unsigned_copy!(u16);
impl_complex_traits_real_unsigned_copy!(u8);
impl_complex_traits_real_unsigned_copy!(usize);

impl PossiblyComplex for num::BigInt {
    fn is_always_real() -> bool {
        true
    }
}

impl Conjugate for num::BigInt {
    type Output = Self;
    fn conj(&self) -> Self::Output {
        self.clone()
    }
}

impl Abs for num::BigInt {
    type Output = Self;
    fn abs(self) -> Self::Output {
        num::BigInt::magnitude(&self).clone().into()
    }
}

impl AbsSqr for num::BigInt {
    type Output = Self;
    fn abs_sqr(self) -> Self::Output {
        self.clone() * self.clone()
    }
}

impl Signum for num::BigInt {
    type Output = Self;
    fn signum(self) -> Self::Output {
        match num::BigInt::sign(&self) {
            num::bigint::Sign::Minus => (-1).into(),
            num::bigint::Sign::NoSign => 0.into(),
            num::bigint::Sign::Plus => 1.into(),
        }
    }
}

impl Arg for num::BigInt {
    type Output = Self;
    fn arg(self) -> Self::Output {
        if self >= self.zero() {
            self.zero() // ln(1) = 0
        } else {
            3.into()
        }
    }
}

impl PossiblyComplex for num::BigUint {
    fn is_always_real() -> bool {
        true
    }
}

impl Conjugate for num::BigUint {
    type Output = Self;
    fn conj(&self) -> Self::Output {
        self.clone()
    }
}

impl Abs for num::BigUint {
    type Output = Self;
    fn abs(self) -> Self::Output {
        self.clone()
    }
}

impl AbsSqr for num::BigUint {
    type Output = Self;
    fn abs_sqr(self) -> Self::Output {
        self.clone() * self.clone()
    }
}

impl Signum for num::BigUint {
    type Output = Self;
    fn signum(self) -> Self::Output {
        if <Self as InstZero>::is_zero(&self) {
            self.clone()
        } else {
            self.one()
        }
    }
}

impl Arg for num::BigUint {
    type Output = Self;
    fn arg(self) -> Self::Output {
        self.zero()
    }
}

// generic implementations done here
impl<T> PossiblyComplex for Wrapping<T>
where
    T: PossiblyComplex,
{
    fn is_always_real() -> bool {
        T::is_always_real()
    }
}
impl<T> Conjugate for Wrapping<T>
where
    T: Conjugate,
{
    type Output = Wrapping<T::Output>;
    fn conj(&self) -> Self::Output {
        Wrapping(self.0.conj())
    }
}
impl<T> Abs for Wrapping<T>
where
    T: Abs,
{
    type Output = Wrapping<T::Output>;
    fn abs(self) -> Self::Output {
        Wrapping(self.0.abs())
    }
}
impl<T> AbsSqr for Wrapping<T>
where
    T: AbsSqr,
{
    type Output = Wrapping<T::Output>;
    fn abs_sqr(self) -> Self::Output {
        Wrapping(self.0.abs_sqr())
    }
}
impl<T> Signum for Wrapping<T>
where
    T: Signum,
{
    type Output = Wrapping<T::Output>;
    fn signum(self) -> Self::Output {
        Wrapping(self.0.signum())
    }
}
impl<T> Arg for Wrapping<T>
where
    T: Arg,
{
    type Output = Wrapping<T::Output>;
    fn arg(self) -> Self::Output {
        Wrapping(self.0.arg())
    }
}

impl<T> PossiblyComplex for Ratio<T>
where
    T: PossiblyComplex,
{
    fn is_always_real() -> bool {
        T::is_always_real()
    }
}
impl<T, O> Conjugate for Ratio<T>
where
    T: Clone + Integer + Conjugate<Output = O>,
    O: Clone + Integer,
{
    type Output = Ratio<<T as Conjugate>::Output>;
    fn conj(&self) -> Self::Output {
        Ratio::new(self.numer().clone().conj(), self.denom().clone().conj())
    }
}
impl<T, O> Abs for Ratio<T>
where
    T: Clone + Integer + Abs<Output = O>,
    O: Clone + Integer,
{
    type Output = Ratio<<T as Abs>::Output>;
    fn abs(self) -> Self::Output {
        Ratio::new(self.numer().clone().abs(), self.denom().clone().abs())
    }
}
impl<T, O> AbsSqr for Ratio<T>
where
    T: Clone + Integer + AbsSqr<Output = O>,
    O: Clone + Integer,
{
    type Output = Ratio<<T as AbsSqr>::Output>;
    fn abs_sqr(self) -> Self::Output {
        Ratio::new(
            self.numer().clone().abs_sqr(),
            self.denom().clone().abs_sqr(),
        )
    }
}
impl<T, O> Signum for Ratio<T>
where
    T: Clone + Integer + Signum<Output = O>,
    O: Clone + Integer,
{
    type Output = Ratio<<T as Signum>::Output>;
    fn signum(self) -> Self::Output {
        Ratio::new(self.numer().clone().signum(), self.denom().clone().signum())
    }
}
impl<T> Arg for Ratio<T>
where
    T: Clone + Integer + Arg<Output = T> + InstOne,
{
    type Output = Ratio<<T as Arg>::Output>;
    fn arg(self) -> Self::Output {
        Ratio::new(
            self.numer().clone().arg() - self.denom().clone().arg(),
            self.denom().clone().one(),
        )
    }
}

impl<T> PossiblyComplex for Complex<T>
where
    T: Clone + Num,
{
    fn is_always_real() -> bool {
        false
    }
}
impl<T> Conjugate for Complex<T>
where
    T: Clone + Num + Neg<Output = T>,
{
    type Output = Self;
    fn conj(&self) -> Self::Output {
        Complex::<T>::conj(self)
    }
}
impl<T> Abs for Complex<T>
where
    T: Clone + Num + Float,
{
    type Output = Self;
    fn abs(self) -> Self::Output {
        self.norm().into()
    }
}
impl<T> AbsSqr for Complex<T>
where
    T: Clone + Num,
{
    type Output = Self;
    fn abs_sqr(self) -> Self::Output {
        self.norm_sqr().into()
    }
}
impl<T> Signum for Complex<T>
where
    T: Clone + Num + Float,
{
    type Output = Self;
    fn signum(self) -> Self::Output {
        self.clone() / self.abs()
    }
}
impl<T> Arg for Complex<T>
where
    T: Clone + Num + Float,
{
    type Output = Self;
    fn arg(self) -> Self::Output {
        Complex::<T>::arg(self).into()
    }
}
