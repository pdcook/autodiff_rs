use std::ops::{Add, Sub, Mul, Div};

pub trait InstZero: Sized + Add<Self, Output = Self>
{
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
    fn one(&self) -> Self;

    // provided methods
    fn set_one(&mut self) {
        *self = self.one();
    }

    fn is_one(&self) -> bool
        where Self: PartialEq
    {
        *self == self.one()
    }
}

// implement InstZero for all types that implement Sized, Add, and PartialEq
// as well as Sub on their references
// Sub and PartialEq on their references
impl<T> InstZero for T
    where T: Sized + Add<T, Output = T> + PartialEq,
          for<'a> &'a T: Sub<&'a T, Output = T>
{
    fn zero(&self) -> T {
        self - self
    }

    fn is_zero(&self) -> bool {
        *self == self.zero()
    }
}

// implement InstOne for all types that implement Sized, Mul, and InstZero
// as well as Div on their references
impl<T> InstOne for T
    where T: Mul<T, Output = T> + InstZero,
          for<'a> &'a T: Div<&'a T, Output = T>
{
    fn one(&self) -> T {
        // TODO: is there a way to still get 1 without possibly panicking?
        if self.is_zero() {
            panic!("divide by zero");
        }
        self / self
    }
}
