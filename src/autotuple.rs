use std::ops::{Add, Sub, Mul, Div, Deref, Neg, Rem};
use crate::traits::{InstZero, InstOne};
use num::traits::{Pow, Signed, Num, NumOps, Zero, One};
use paste::paste;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoTuple<Tuple>(pub Tuple)
where
    Tuple: Clone + PartialEq;

impl<Tuple> AutoTuple<Tuple>
where
    Tuple: Clone + PartialEq,
{
    pub fn new(tuple: Tuple) -> Self {
        Self(tuple)
    }
}

impl<Tuple> Deref for AutoTuple<Tuple>
where
    Tuple: Clone + PartialEq,
{
    type Target = Tuple;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// impl Num to always return an error
impl<Tuple> Num for AutoTuple<Tuple>
where
    Tuple: Clone + PartialEq,
    AutoTuple<Tuple>: NumOps + One + Zero + PartialEq,
{
    type FromStrRadixErr = ();

    fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err(())
    }
}

// impl From for autotuple
macro_rules! autotuple_from {
    ($($idx:literal),+) =>
    {
        paste! {
            impl<$([<T $idx>],)+> From<($([<T $idx>],)+)> for AutoTuple<($([<T $idx>],)+)>
            where
                ($([<T $idx>],)+): Clone + PartialEq,
            {
                fn from(tuple: ($([<T $idx>],)+)) -> Self {
                    AutoTuple::new(tuple)
                }
            }
        }
    }
}

// implement From for autotuples up to length 16
autotuple_from!(0);
autotuple_from!(0, 1);
autotuple_from!(0, 1, 2);
autotuple_from!(0, 1, 2, 3);
autotuple_from!(0, 1, 2, 3, 4);
autotuple_from!(0, 1, 2, 3, 4, 5);
autotuple_from!(0, 1, 2, 3, 4, 5, 6);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15);

// macro for implementing binary operations between autotuples with
// the same number of elements
// example: autotuple_binary_op!(Add, add, 0, 1, 2);
// implements Add<AutoTuple<(U0, U1, U2)>> for AutoTuple<(T0, T1, T2)>
// using paste! to make new types T0, T1, T2, etc

macro_rules! autotuple_binary_op {
    ($trt:ident, $mth:ident, $($idx:literal),+) =>
    {
        paste! {
            // AutoTuple op AutoTuple
            impl<$([<T $idx>],)+ $([<U $idx>],)+> $trt<AutoTuple<($([<U $idx>],)+)>> for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: $trt<[<U $idx>], Output=[<T $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                ($([<U $idx>],)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($([<T $idx>],)+)>;

                fn $mth(self, rhs: AutoTuple<($([<U $idx>],)+)>) -> Self::Output {
                    AutoTuple::new(($( self.0.$idx.$mth(rhs.0.$idx), )+))
                }
            }
            // AutoTuple op (U0, U1, U2)
            impl<$([<T $idx>],)+ $([<U $idx>],)+> $trt<($([<U $idx>],)+)> for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: $trt<[<U $idx>], Output=[<T $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                $([<U $idx>]: Clone + PartialEq,)+
            {
                type Output = AutoTuple<($([<T $idx>],)+)>;

                fn $mth(self, rhs: ($([<U $idx>],)+)) -> Self::Output {
                    AutoTuple::new(($( self.0.$idx.$mth(rhs.$idx.clone()), )+))
                }
            }
        }
    }
}

// macro for implementing all binary ops between autotuples with
// specified length
macro_rules! autotuple_binary_ops {
    ($($idx:literal),+) =>
    {
        autotuple_binary_op!(Add, add, $($idx),+);
        autotuple_binary_op!(Sub, sub, $($idx),+);
        autotuple_binary_op!(Mul, mul, $($idx),+);
        autotuple_binary_op!(Div, div, $($idx),+);
        autotuple_binary_op!(Rem, rem, $($idx),+);
        autotuple_binary_op!(Pow, pow, $($idx),+);
    }
}

// implement all binary ops for autotuples up to length 16
autotuple_binary_ops!(0);
autotuple_binary_ops!(0, 1);
autotuple_binary_ops!(0, 1, 2);
autotuple_binary_ops!(0, 1, 2, 3);
autotuple_binary_ops!(0, 1, 2, 3, 4);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15);

// macro for implementing unary operations on autotuples
macro_rules! autotuple_unary_ops {
    ($($idx:literal),+) =>
    {
        paste! {
            impl<$([<T $idx>],)+> Neg for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: Neg<Output=[<T $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($([<T $idx>],)+)>;

                fn neg(self) -> Self::Output {
                    AutoTuple::new(($( self.0.$idx.neg(), )+))
                }
            }
            impl<$([<T $idx>],)+> Zero for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: Zero + Add<[<T $idx>], Output=[<T $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                AutoTuple<($([<T $idx>],)+)>: Add<AutoTuple<($([<T $idx>],)+)>, Output=AutoTuple<($([<T $idx>],)+)>>,
            {
                fn zero() -> Self {
                    AutoTuple::new(($([<T $idx>]::zero(), )+))
                }
                fn is_zero(&self) -> bool {
                    $(self.0.$idx.is_zero() && )+ true
                }
            }
            impl<$([<T $idx>],)+> InstZero for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: InstZero,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                AutoTuple<($([<T $idx>],)+)>: Add<AutoTuple<($([<T $idx>],)+)>, Output=AutoTuple<($([<T $idx>],)+)>>,
            {
                fn zero(&self) -> Self {
                    AutoTuple::new(($( self.0.$idx.zero(), )+))
                }
                fn is_zero(&self) -> bool {
                    $(self.0.$idx.is_zero() && )+ true
                }
            }
            impl<$([<T $idx>],)+> One for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: One + Mul<[<T $idx>], Output=[<T $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                AutoTuple<($([<T $idx>],)+)>: Mul<AutoTuple<($([<T $idx>],)+)>, Output=AutoTuple<($([<T $idx>],)+)>>,
            {
                fn one() -> Self {
                    AutoTuple::new(($( [<T $idx>]::one(), )+))
                }
            }
            impl<$([<T $idx>],)+> InstOne for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: InstOne + Num + Mul<[<T $idx>], Output=[<T $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                AutoTuple<($([<T $idx>],)+)>: Mul<AutoTuple<($([<T $idx>],)+)>, Output=AutoTuple<($([<T $idx>],)+)>>,
            {
                fn one(&self) -> Self {
                    AutoTuple::new(($( self.0.$idx.one(), )+))
                }
            }
            impl<$([<T $idx>],)+> Signed for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: Signed + Num + Neg<Output = [<T $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                AutoTuple<($([<T $idx>],)+)>: NumOps,
            {
                fn abs(&self) -> Self {
                    AutoTuple::new(($( self.0.$idx.abs(), )+))
                }
                fn abs_sub(&self, other: &Self) -> Self {
                    AutoTuple::new(($( self.0.$idx.abs_sub(&other.0.$idx), )+))
                }
                fn signum(&self) -> Self {
                    AutoTuple::new(($( self.0.$idx.signum(), )+))
                }
                fn is_positive(&self) -> bool {
                    $(self.0.$idx.is_positive() && )+ true
                }
                fn is_negative(&self) -> bool {
                    $(self.0.$idx.is_negative() && )+ true
                }
            }
        }
    }
}

// implement for tuples of length 1-16
autotuple_unary_ops!(0);
autotuple_unary_ops!(0, 1);
autotuple_unary_ops!(0, 1, 2);
autotuple_unary_ops!(0, 1, 2, 3);
autotuple_unary_ops!(0, 1, 2, 3, 4);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15);

#[test]
fn test_autotuple() {
    let a = AutoTuple::new((1u32, 1.0_f64));
    let b_tup = (2u32, -1.0_f64);
    let b = AutoTuple::new(b_tup);
    let c1 = a + b;
    let c2 = a + b_tup;
    assert_eq!(c1, AutoTuple::new((3, 0.0)));
    assert_eq!(c2, AutoTuple::new((3, 0.0)));
}
