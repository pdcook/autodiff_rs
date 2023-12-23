use std::ops::{Add, Sub, Mul, Div, Neg, Deref};
use num::traits::Pow;
use crate::traits::{InstOne, InstZero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoTuple<T>(T);

impl<T> AutoTuple<T> {
    pub fn new(t: T) -> Self {
        Self(t)
    }
}

impl<T> Deref for AutoTuple<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// implement Add for AutoTuples of size 1, 2, 3
impl<A1, A2> Add<AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Add<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn add(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0,))
    }
}

impl<A1, A2, B1, B2> Add<AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Add<A2, Output=A1>,
    B1: Add<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn add(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0, self.0.1 + rhs.0.1))
    }
}

impl<A1, A2, B1, B2, C1, C2> Add<AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Add<A2, Output=A1>,
    B1: Add<B2, Output=B1>,
    C1: Add<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn add(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0, self.0.1 + rhs.0.1, self.0.2 + rhs.0.2))
    }
}

// implement Sub for AutoTuples of size 1, 2, 3
impl<A1, A2> Sub<AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Sub<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn sub(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0,))
    }
}

impl<A1, A2, B1, B2> Sub<AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Sub<A2, Output=A1>,
    B1: Sub<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn sub(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0, self.0.1 - rhs.0.1))
    }
}

impl<A1, A2, B1, B2, C1, C2> Sub<AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Sub<A2, Output=A1>,
    B1: Sub<B2, Output=B1>,
    C1: Sub<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn sub(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0, self.0.1 - rhs.0.1, self.0.2 - rhs.0.2))
    }
}

// implement Mul for AutoTuples of size 1, 2, 3
impl<A1, A2> Mul<AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Mul<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn mul(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0,))
    }
}

impl<A1, A2, B1, B2> Mul<AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Mul<A2, Output=A1>,
    B1: Mul<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn mul(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0, self.0.1 * rhs.0.1))
    }
}

impl<A1, A2, B1, B2, C1, C2> Mul<AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Mul<A2, Output=A1>,
    B1: Mul<B2, Output=B1>,
    C1: Mul<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn mul(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0, self.0.1 * rhs.0.1, self.0.2 * rhs.0.2))
    }
}

// implement Div for AutoTuples of size 1, 2, 3
impl<A1, A2> Div<AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Div<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn div(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0,))
    }
}

impl<A1, A2, B1, B2> Div<AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Div<A2, Output=A1>,
    B1: Div<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn div(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0, self.0.1 / rhs.0.1))
    }
}

impl<A1, A2, B1, B2, C1, C2> Div<AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Div<A2, Output=A1>,
    B1: Div<B2, Output=B1>,
    C1: Div<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn div(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0, self.0.1 / rhs.0.1, self.0.2 / rhs.0.2))
    }
}

// implement Neg for AutoTuples of size 1, 2, 3
impl<A1> Neg for AutoTuple<(A1,)>
where
    A1: Neg<Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn neg(self) -> Self::Output {
        AutoTuple::new((-self.0.0,))
    }
}

impl<A1, B1> Neg for AutoTuple<(A1, B1)>
where
    A1: Neg<Output=A1>,
    B1: Neg<Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn neg(self) -> Self::Output {
        AutoTuple::new((-self.0.0, -self.0.1))
    }
}

impl<A1, B1, C1> Neg for AutoTuple<(A1, B1, C1)>
where
    A1: Neg<Output=A1>,
    B1: Neg<Output=B1>,
    C1: Neg<Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn neg(self) -> Self::Output {
        AutoTuple::new((-self.0.0, -self.0.1, -self.0.2))
    }
}

// implement Pow for AutoTuples of size 1, 2, 3
impl<A1, A2> Pow<AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Pow<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn pow(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0),))
    }
}

impl<A1, A2, B1, B2> Pow<AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Pow<A2, Output=A1>,
    B1: Pow<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn pow(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0), self.0.1.pow(rhs.0.1)))
    }
}

impl<A1, A2, B1, B2, C1, C2> Pow<AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Pow<A2, Output=A1>,
    B1: Pow<B2, Output=B1>,
    C1: Pow<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn pow(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0), self.0.1.pow(rhs.0.1), self.0.2.pow(rhs.0.2)))
    }
}

// implement InstOne for AutoTuples of size 1, 2, 3
impl<A1> InstOne for AutoTuple<(A1,)>
where
    A1: InstOne,
{
    fn one(&self) -> Self {
        AutoTuple::new((self.0.0.one(),))
    }
}

impl<A1, B1> InstOne for AutoTuple<(A1, B1)>
where
    A1: InstOne,
    B1: InstOne,
{
    fn one(&self) -> Self {
        AutoTuple::new((self.0.0.one(), self.0.1.one()))
    }
}

impl<A1, B1, C1> InstOne for AutoTuple<(A1, B1, C1)>
where
    A1: InstOne,
    B1: InstOne,
    C1: InstOne,
{
    fn one(&self) -> Self {
        AutoTuple::new((self.0.0.one(), self.0.1.one(), self.0.2.one()))
    }
}

// implement InstZero for AutoTuples of size 1, 2, 3
impl<A1> InstZero for AutoTuple<(A1,)>
where
    A1: InstZero,
{
    fn zero(&self) -> Self {
        AutoTuple::new((self.0.0.zero(),))
    }

    fn is_zero(&self) -> bool {
        self.0.0.is_zero()
    }
}

impl<A1, B1> InstZero for AutoTuple<(A1, B1)>
where
    A1: InstZero,
    B1: InstZero,
{
    fn zero(&self) -> Self {
        AutoTuple::new((self.0.0.zero(), self.0.1.zero()))
    }

    fn is_zero(&self) -> bool {
        self.0.0.is_zero() && self.0.1.is_zero()
    }
}

impl<A1, B1, C1> InstZero for AutoTuple<(A1, B1, C1)>
where
    A1: InstZero,
    B1: InstZero,
    C1: InstZero,
{
    fn zero(&self) -> Self {
        AutoTuple::new((self.0.0.zero(), self.0.1.zero(), self.0.2.zero()))
    }

    fn is_zero(&self) -> bool {
        self.0.0.is_zero() && self.0.1.is_zero() && self.0.2.is_zero()
    }
}

// repeat all implementations for &AutoTuple
// binary ops with &AutoTuple and AutoTuple

impl<'a, A1, A2> Add<&'a AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Add<A2, Output=A1> + Clone,
    A2: Clone,
{
    type Output = AutoTuple<(A1,)>;

    fn add(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0.clone() + rhs.0.0.clone(),))
    }
}

impl<'a, A1, A2, B1, B2> Add<&'a AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Add<A2, Output=A1> + Clone,
    A2: Clone,
    B1: Add<B2, Output=B1> + Clone,
    B2: Clone,
{
    type Output = AutoTuple<(A1, B1)>;

    fn add(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0.clone() + rhs.0.0.clone(), self.0.1.clone() + rhs.0.1.clone()))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Add<&'a AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Add<A2, Output=A1>,
    B1: Add<B2, Output=B1>,
    C1: Add<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn add(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0, self.0.1 + rhs.0.1, self.0.2 + rhs.0.2))
    }
}

impl<'a, A1, A2> Add<AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Add<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn add(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Add<AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Add<A2, Output=A1>,
    B1: Add<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn add(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0, self.0.1 + rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Add<AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Add<A2, Output=A1>,
    B1: Add<B2, Output=B1>,
    C1: Add<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn add(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0, self.0.1 + rhs.0.1, self.0.2 + rhs.0.2))
    }
}

impl<'a, A1, A2> Add<&'a AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Add<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn add(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Add<&'a AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Add<A2, Output=A1>,
    B1: Add<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn add(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0, self.0.1 + rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Add<&'a AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Add<A2, Output=A1>,
    B1: Add<B2, Output=B1>,
    C1: Add<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn add(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 + rhs.0.0, self.0.1 + rhs.0.1, self.0.2 + rhs.0.2))
    }
}

// sub

impl<'a, A1, A2> Sub<&'a AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Sub<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn sub(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Sub<&'a AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Sub<A2, Output=A1>,
    B1: Sub<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn sub(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0, self.0.1 - rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Sub<&'a AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Sub<A2, Output=A1>,
    B1: Sub<B2, Output=B1>,
    C1: Sub<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn sub(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0, self.0.1 - rhs.0.1, self.0.2 - rhs.0.2))
    }
}

impl<'a, A1, A2> Sub<AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Sub<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn sub(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Sub<AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Sub<A2, Output=A1>,
    B1: Sub<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn sub(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0, self.0.1 - rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Sub<AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Sub<A2, Output=A1>,
    B1: Sub<B2, Output=B1>,
    C1: Sub<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn sub(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0, self.0.1 - rhs.0.1, self.0.2 - rhs.0.2))
    }
}

impl<'a, A1, A2> Sub<&'a AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Sub<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn sub(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Sub<&'a AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Sub<A2, Output=A1>,
    B1: Sub<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn sub(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0, self.0.1 - rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Sub<&'a AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Sub<A2, Output=A1>,
    B1: Sub<B2, Output=B1>,
    C1: Sub<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn sub(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 - rhs.0.0, self.0.1 - rhs.0.1, self.0.2 - rhs.0.2))
    }
}

// mul

impl<'a, A1, A2> Mul<&'a AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Mul<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn mul(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Mul<&'a AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Mul<A2, Output=A1>,
    B1: Mul<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn mul(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0, self.0.1 * rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Mul<&'a AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Mul<A2, Output=A1>,
    B1: Mul<B2, Output=B1>,
    C1: Mul<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn mul(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0, self.0.1 * rhs.0.1, self.0.2 * rhs.0.2))
    }
}

impl<'a, A1, A2> Mul<AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Mul<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn mul(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Mul<AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Mul<A2, Output=A1>,
    B1: Mul<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn mul(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0, self.0.1 * rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Mul<AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Mul<A2, Output=A1>,
    B1: Mul<B2, Output=B1>,
    C1: Mul<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn mul(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0, self.0.1 * rhs.0.1, self.0.2 * rhs.0.2))
    }
}

impl<'a, A1, A2> Mul<&'a AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Mul<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn mul(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Mul<&'a AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Mul<A2, Output=A1>,
    B1: Mul<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn mul(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0, self.0.1 * rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Mul<&'a AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Mul<A2, Output=A1>,
    B1: Mul<B2, Output=B1>,
    C1: Mul<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn mul(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 * rhs.0.0, self.0.1 * rhs.0.1, self.0.2 * rhs.0.2))
    }
}

// div

impl<'a, A1, A2> Div<&'a AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Div<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn div(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Div<&'a AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Div<A2, Output=A1>,
    B1: Div<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn div(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0, self.0.1 / rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Div<&'a AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Div<A2, Output=A1>,
    B1: Div<B2, Output=B1>,
    C1: Div<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn div(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0, self.0.1 / rhs.0.1, self.0.2 / rhs.0.2))
    }
}

impl<'a, A1, A2> Div<AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Div<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn div(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Div<AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Div<A2, Output=A1>,
    B1: Div<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn div(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0, self.0.1 / rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Div<AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Div<A2, Output=A1>,
    B1: Div<B2, Output=B1>,
    C1: Div<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn div(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0, self.0.1 / rhs.0.1, self.0.2 / rhs.0.2))
    }
}

impl<'a, A1, A2> Div<&'a AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Div<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn div(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0,))
    }
}

impl<'a, A1, A2, B1, B2> Div<&'a AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Div<A2, Output=A1>,
    B1: Div<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn div(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0, self.0.1 / rhs.0.1))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Div<&'a AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Div<A2, Output=A1>,
    B1: Div<B2, Output=B1>,
    C1: Div<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn div(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0 / rhs.0.0, self.0.1 / rhs.0.1, self.0.2 / rhs.0.2))
    }
}

// pow

impl<'a, A1, A2> Pow<&'a AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Pow<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn pow(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0),))
    }
}

impl<'a, A1, A2, B1, B2> Pow<&'a AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Pow<A2, Output=A1>,
    B1: Pow<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn pow(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0), self.0.1.pow(rhs.0.1)))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Pow<&'a AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Pow<A2, Output=A1>,
    B1: Pow<B2, Output=B1>,
    C1: Pow<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn pow(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0), self.0.1.pow(rhs.0.1), self.0.2.pow(rhs.0.2)))
    }
}

impl<'a, A1, A2> Pow<AutoTuple<(A2,)>> for &'a AutoTuple<(A1,)>
where
    A1: Pow<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn pow(self, rhs: AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0),))
    }
}

impl<'a, A1, A2, B1, B2> Pow<AutoTuple<(A2, B2)>> for &'a AutoTuple<(A1, B1)>
where
    A1: Pow<A2, Output=A1>,
    B1: Pow<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn pow(self, rhs: AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0), self.0.1.pow(rhs.0.1)))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Pow<AutoTuple<(A2, B2, C2)>> for &'a AutoTuple<(A1, B1, C1)>
where
    A1: Pow<A2, Output=A1>,
    B1: Pow<B2, Output=B1>,
    C1: Pow<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn pow(self, rhs: AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0), self.0.1.pow(rhs.0.1), self.0.2.pow(rhs.0.2)))
    }
}

impl<'a, A1, A2> Pow<&'a AutoTuple<(A2,)>> for AutoTuple<(A1,)>
where
    A1: Pow<A2, Output=A1>,
{
    type Output = AutoTuple<(A1,)>;

    fn pow(self, rhs: &'a AutoTuple<(A2,)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0),))
    }
}

impl<'a, A1, A2, B1, B2> Pow<&'a AutoTuple<(A2, B2)>> for AutoTuple<(A1, B1)>
where
    A1: Pow<A2, Output=A1>,
    B1: Pow<B2, Output=B1>,
{
    type Output = AutoTuple<(A1, B1)>;

    fn pow(self, rhs: &'a AutoTuple<(A2, B2)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0), self.0.1.pow(rhs.0.1)))
    }
}

impl<'a, A1, A2, B1, B2, C1, C2> Pow<&'a AutoTuple<(A2, B2, C2)>> for AutoTuple<(A1, B1, C1)>
where
    A1: Pow<A2, Output=A1>,
    B1: Pow<B2, Output=B1>,
    C1: Pow<C2, Output=C1>,
{
    type Output = AutoTuple<(A1, B1, C1)>;

    fn pow(self, rhs: &'a AutoTuple<(A2, B2, C2)>) -> Self::Output {
        AutoTuple::new((self.0.0.pow(rhs.0.0), self.0.1.pow(rhs.0.1), self.0.2.pow(rhs.0.2)))
    }
}

// neg

impl<'a, A> Neg for &'a AutoTuple<(A,)>
where
    A: Neg<Output=A> + Clone,
{
    type Output = AutoTuple<(A,)>;

    fn neg(self) -> Self::Output {
        AutoTuple::new((-self.0.0.clone(),))
    }
}

impl<'a, A, B> Neg for &'a AutoTuple<(A, B)>
where
    A: Neg<Output=A> + Clone,
    B: Neg<Output=B> + Clone,
{
    type Output = AutoTuple<(A, B)>;

    fn neg(self) -> Self::Output {
        AutoTuple::new((-self.0.0.clone(), -self.0.1.clone()))
    }
}

impl<'a, A, B, C> Neg for &'a AutoTuple<(A, B, C)>
where
    A: Neg<Output=A> + Clone,
    B: Neg<Output=B> + Clone,
    C: Neg<Output=C> + Clone,
{
    type Output = AutoTuple<(A, B, C)>;

    fn neg(self) -> Self::Output {
        AutoTuple::new((-self.0.0.clone(), -self.0.1.clone(), -self.0.2.clone()))
    }
}
