use crate::traits::{ComposedGradMul, InstOne, InstZero};
use num::complex::Complex;
use num::traits::{Num, NumOps, One, Pow, Signed, Zero};
use paste::paste;
use std::ops::{Add, Deref, Div, Mul, Neg, Rem, Sub};

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
autotuple_from!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

// macro for implementing AutoTuple From<T> for T, used for constants
// so AutoTuple::From(f32) -> AutoTuple<(f32,)>
// i.e. AutoTuple::From(f32) = AutoTuple::From((f32,))

macro_rules! autotuple_from_primitive {
    ($($type:ty),+) =>
    {
        $(
            impl From<$type> for AutoTuple<($type,)> {
                fn from(t: $type) -> Self {
                    AutoTuple::new((t,))
                }
            }
        )+
    }
}

autotuple_from_primitive!(
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    Complex<f32>,
    Complex<f64>
);

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
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
autotuple_binary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

// macro to implement ComposedGradMul for autotuples of any length
// whose elements implement ComposedGradMul

macro_rules! autotuple_composed_grad_mul {
    ($($idx:literal),+) =>
    {
        paste! {
            impl<$([<T $idx>],)+ $([<IIT $idx>],)+ $([<OOT $idx>],)+ $([<IGT $idx>],)+> ComposedGradMul<
                AutoTuple<($([<IIT $idx>],)+)>,
                AutoTuple<($([<OOT $idx>],)+)>,
                AutoTuple<($([<IGT $idx>],)+)>,
                > for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: ComposedGradMul<[<IIT $idx>], [<OOT $idx>], [<IGT $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                ($([<IIT $idx>],)+): Clone + PartialEq,
                ($([<OOT $idx>],)+): Clone + PartialEq,
                ($([<IGT $idx>],)+): Clone + PartialEq,
                ($(<[<T $idx>] as ComposedGradMul<[<IIT $idx>], [<OOT $idx>], [<IGT $idx>]>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<[<T $idx>] as ComposedGradMul<[<IIT $idx>], [<OOT $idx>], [<IGT $idx>]>>
                    ::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<($([<IIT $idx>],)+)>,
                    f_of_g: &AutoTuple<($([<OOT $idx>],)+)>,
                    dg: &AutoTuple<($([<IGT $idx>],)+)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.$idx.compose_mul(&x.0.$idx, &f_of_g.0.$idx, &dg.0.$idx),)+))
                }
            }
        }
    }
}

// implement ComposedGradMul for autotuples up to length 16
autotuple_composed_grad_mul!(0);
autotuple_composed_grad_mul!(0, 1);
autotuple_composed_grad_mul!(0, 1, 2);
autotuple_composed_grad_mul!(0, 1, 2, 3);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
autotuple_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

// macro for operations between autotuples of any length and
// autotuples of length 1
// so that it can do constant operations on AutoTuples of any length

macro_rules! autotuple_const_op {
    ($trt:ident, $mth:ident, $($idx:literal),+) =>
    {
        paste! {
            // AutoTuple op AutoTuple
            impl<$([<T $idx>],)+ U> $trt<AutoTuple<(U,)>> for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: $trt<U, Output=[<T $idx>]>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                U: Clone + PartialEq,
            {
                type Output = AutoTuple<($([<T $idx>],)+)>;

                fn $mth(self, rhs: AutoTuple<(U,)>) -> Self::Output {
                    AutoTuple::new(($( self.0.$idx.$mth(rhs.0.0.clone()), )+))
                }
            }
        }
    }
}

// macro for implementing all binary ops between autotuples with
// specified length
macro_rules! autotuple_const_ops {
    ($($idx:literal),+) =>
    {
        autotuple_const_op!(Add, add, $($idx),+);
        autotuple_const_op!(Sub, sub, $($idx),+);
        autotuple_const_op!(Mul, mul, $($idx),+);
        autotuple_const_op!(Div, div, $($idx),+);
        autotuple_const_op!(Rem, rem, $($idx),+);
        autotuple_const_op!(Pow, pow, $($idx),+);
    }
}

// implement all binary ops for autotuples up to length 16
//autotuple_const_ops!(0);
autotuple_const_ops!(0, 1);
autotuple_const_ops!(0, 1, 2);
autotuple_const_ops!(0, 1, 2, 3);
autotuple_const_ops!(0, 1, 2, 3, 4);
autotuple_const_ops!(0, 1, 2, 3, 4, 5);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
autotuple_const_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

// macro to implement ComposedGradMul for autotuples of any length
// whose elements implement ComposedGradMul
// with AutoTuples of length 1

macro_rules! autotuple_const_composed_grad_mul {
    ($($idx:literal),+) =>
    {
        paste! {
            /// <len n>.compose_mul(<len n>, <len n>, <len 1>)
            impl<$([<T $idx>],)+ $([<IIT $idx>],)+ $([<OOT $idx>],)+ IGT> ComposedGradMul<
                AutoTuple<($([<IIT $idx>],)+)>,
                AutoTuple<($([<OOT $idx>],)+)>,
                AutoTuple<(IGT,)>,
                > for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: ComposedGradMul<[<IIT $idx>], [<OOT $idx>], IGT>,)+
                ($([<IIT $idx>],)+): Clone + PartialEq,
                ($([<OOT $idx>],)+): Clone + PartialEq,
                ($([<T $idx>],)+): Clone + PartialEq,
                IGT: Clone + PartialEq,
                ($(<[<T $idx>] as ComposedGradMul<[<IIT $idx>], [<OOT $idx>], IGT>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<[<T $idx>] as ComposedGradMul<[<IIT $idx>], [<OOT $idx>], IGT>>::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<($([<IIT $idx>],)+)>,
                    f_of_g: &AutoTuple<($([<OOT $idx>],)+)>,
                    dg: &AutoTuple<(IGT,)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.$idx.compose_mul(&x.0.$idx, &f_of_g.0.$idx, &dg.0.0),)+))
                }
            }
            /// <len n>.compose_mul(<len n>, <len 1>, <len n>)
            impl<$([<T $idx>],)+ $([<IIT $idx>],)+ OOT, $([<IGT $idx>],)+> ComposedGradMul<
                AutoTuple<($([<IIT $idx>],)+)>,
                AutoTuple<(OOT,)>,
                AutoTuple<($([<IGT $idx>],)+)>,
                > for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: ComposedGradMul<[<IIT $idx>], OOT, [<IGT $idx>]>,)+
                ($([<IIT $idx>],)+): Clone + PartialEq,
                OOT: Clone + PartialEq,
                ($([<T $idx>],)+): Clone + PartialEq,
                ($([<IGT $idx>],)+): Clone + PartialEq,
                ($(<[<T $idx>] as ComposedGradMul<[<IIT $idx>], OOT, [<IGT $idx>]>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<[<T $idx>] as ComposedGradMul<[<IIT $idx>], OOT, [<IGT $idx>]>>::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<($([<IIT $idx>],)+)>,
                    f_of_g: &AutoTuple<(OOT,)>,
                    dg: &AutoTuple<($([<IGT $idx>],)+)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.$idx.compose_mul(&x.0.$idx, &f_of_g.0.0, &dg.0.$idx),)+))
                }
            }
            /// <len n>.compose_mul(<len n>, <len 1>, <len 1>)
            impl<$([<T $idx>],)+ $([<IIT $idx>],)+ OOT, IGT> ComposedGradMul<
                AutoTuple<($([<IIT $idx>],)+)>,
                AutoTuple<(OOT,)>,
                AutoTuple<(IGT,)>,
                > for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: ComposedGradMul<[<IIT $idx>], OOT, IGT>,)+
                ($([<IIT $idx>],)+): Clone + PartialEq,
                OOT: Clone + PartialEq,
                ($([<T $idx>],)+): Clone + PartialEq,
                IGT: Clone + PartialEq,
                ($(<[<T $idx>] as ComposedGradMul<[<IIT $idx>], OOT, IGT>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<[<T $idx>] as ComposedGradMul<[<IIT $idx>], OOT, IGT>>::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<($([<IIT $idx>],)+)>,
                    f_of_g: &AutoTuple<(OOT,)>,
                    dg: &AutoTuple<(IGT,)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.$idx.compose_mul(&x.0.$idx, &f_of_g.0.0, &dg.0.0),)+))
                }
            }
            /// <len n>.compose_mul(<len 1>, <len n>, <len n>)
            impl<$([<T $idx>],)+ IIT, $([<OOT $idx>],)+ $([<IGT $idx>],)+> ComposedGradMul<
                AutoTuple<(IIT,)>,
                AutoTuple<($([<OOT $idx>],)+)>,
                AutoTuple<($([<IGT $idx>],)+)>,
                > for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: ComposedGradMul<IIT, [<OOT $idx>], [<IGT $idx>]>,)+
                ($([<OOT $idx>],)+): Clone + PartialEq,
                ($([<T $idx>],)+): Clone + PartialEq,
                IIT: Clone + PartialEq,
                ($([<IGT $idx>],)+): Clone + PartialEq,
                ($(<[<T $idx>] as ComposedGradMul<IIT, [<OOT $idx>], [<IGT $idx>]>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<[<T $idx>] as ComposedGradMul<IIT, [<OOT $idx>], [<IGT $idx>]>>::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<(IIT,)>,
                    f_of_g: &AutoTuple<($([<OOT $idx>],)+)>,
                    dg: &AutoTuple<($([<IGT $idx>],)+)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.$idx.compose_mul(&x.0.0, &f_of_g.0.$idx, &dg.0.$idx),)+))
                }
            }
            /// <len n>.compose_mul(<len 1>, <len n>, <len 1>)
            impl<$([<T $idx>],)+ IIT, $([<OOT $idx>],)+ IGT> ComposedGradMul<
                AutoTuple<(IIT,)>,
                AutoTuple<($([<OOT $idx>],)+)>,
                AutoTuple<(IGT,)>,
                > for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: ComposedGradMul<IIT, [<OOT $idx>], IGT>,)+
                ($([<OOT $idx>],)+): Clone + PartialEq,
                ($([<T $idx>],)+): Clone + PartialEq,
                IIT: Clone + PartialEq,
                IGT: Clone + PartialEq,
                ($(<[<T $idx>] as ComposedGradMul<IIT, [<OOT $idx>], IGT>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<[<T $idx>] as ComposedGradMul<IIT, [<OOT $idx>], IGT>>::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<(IIT,)>,
                    f_of_g: &AutoTuple<($([<OOT $idx>],)+)>,
                    dg: &AutoTuple<(IGT,)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.$idx.compose_mul(&x.0.0, &f_of_g.0.$idx, &dg.0.0),)+))
                }
            }
            /// <len n>.compose_mul(<len 1>, <len 1>, <len n>)
            impl<$([<T $idx>],)+ IIT, OOT, $([<IGT $idx>],)+> ComposedGradMul<
                AutoTuple<(IIT,)>,
                AutoTuple<(OOT,)>,
                AutoTuple<($([<IGT $idx>],)+)>,
                > for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: ComposedGradMul<IIT, OOT, [<IGT $idx>]>,)+
                OOT: Clone + PartialEq,
                ($([<T $idx>],)+): Clone + PartialEq,
                IIT: Clone + PartialEq,
                ($([<IGT $idx>],)+): Clone + PartialEq,
                ($(<[<T $idx>] as ComposedGradMul<IIT, OOT, [<IGT $idx>]>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<[<T $idx>] as ComposedGradMul<IIT, OOT, [<IGT $idx>]>>::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<(IIT,)>,
                    f_of_g: &AutoTuple<(OOT,)>,
                    dg: &AutoTuple<($([<IGT $idx>],)+)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.$idx.compose_mul(&x.0.0, &f_of_g.0.0, &dg.0.$idx),)+))
                }
            }
            /// <len n>.compose_mul(<len 1>, <len 1>, <len 1>)
            impl<$([<T $idx>],)+ IIT, OOT, IGT> ComposedGradMul<
                AutoTuple<(IIT,)>,
                AutoTuple<(OOT,)>,
                AutoTuple<(IGT,)>,
                > for AutoTuple<($([<T $idx>],)+)>
            where
                $([<T $idx>]: ComposedGradMul<IIT, OOT, IGT>,)+
                ($([<T $idx>],)+): Clone + PartialEq,
                IIT: Clone + PartialEq,
                OOT: Clone + PartialEq,
                IGT: Clone + PartialEq,
                ($(<[<T $idx>] as ComposedGradMul<IIT, OOT, IGT>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<[<T $idx>] as ComposedGradMul<IIT, OOT, IGT>>::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<(IIT,)>,
                    f_of_g: &AutoTuple<(OOT,)>,
                    dg: &AutoTuple<(IGT,)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.$idx.compose_mul(&x.0.0, &f_of_g.0.0, &dg.0.0),)+))
                }
            }
            /*
            /// <length 1 autotuples>.compose_mul(<length n autotuples>)
            impl<T, $([<IIT $idx>],)+ $([<OOT $idx>],)+ $([<IGT $idx>],)+> ComposedGradMul<
                AutoTuple<($([<IIT $idx>],)+)>,
                AutoTuple<($([<OOT $idx>],)+)>,
                AutoTuple<($([<IGT $idx>],)+)>,
                > for AutoTuple<(T,)>
            where
                T: Clone + PartialEq,
                $([<IIT $idx>]: Clone + PartialEq,)+
                $([<OOT $idx>]: Clone + PartialEq,)+
                $([<IGT $idx>]: Clone + PartialEq,)+
                ($([<IIT $idx>],)+): Clone + PartialEq,
                ($([<OOT $idx>],)+): Clone + PartialEq,
                ($([<IGT $idx>],)+): Clone + PartialEq,
                $(
                    T: ComposedGradMul<[<IIT $idx>], [<OOT $idx>], [<IGT $idx>]>,
                )+
                ($(<T as ComposedGradMul<[<IIT $idx>], [<OOT $idx>], [<IGT $idx>]>>::Output,)+): Clone + PartialEq,
            {
                type Output = AutoTuple<($(<T as ComposedGradMul<[<IIT $idx>], [<OOT $idx>], [<IGT $idx>]>>::Output,)+)>;

                fn compose_mul(
                    &self,
                    x: &AutoTuple<($([<IIT $idx>],)+)>,
                    f_of_g: &AutoTuple<($([<OOT $idx>],)+)>,
                    dg: &AutoTuple<($([<IGT $idx>],)+)>,
                ) -> Self::Output {
                    AutoTuple::new(($(self.0.0.compose_mul(&x.0.$idx, &f_of_g.0.$idx, &dg.0.$idx),)+))
                }
            }
            */
        }
    }
}

// implement const ComposedGradMul for autotuples up to length 16
//autotuple_const_composed_grad_mul!(0);
autotuple_const_composed_grad_mul!(0, 1);
autotuple_const_composed_grad_mul!(0, 1, 2);
autotuple_const_composed_grad_mul!(0, 1, 2, 3);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
autotuple_const_composed_grad_mul!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

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
                $([<T $idx>]: InstOne + Mul<[<T $idx>], Output=[<T $idx>]>,)+
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
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
autotuple_unary_ops!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

#[test]
fn test_autotuple() {
    let a = AutoTuple::new((1u32, 1.0_f64));
    let b_tup = (2u32, -1.0_f64);
    let b = AutoTuple::new(b_tup);
    let c1 = a + b;
    let c2 = a + b_tup;
    assert_eq!(c1, AutoTuple::new((3, 0.0)));
    assert_eq!(c2, AutoTuple::new((3, 0.0)));
    assert_eq!((*c1).0, 3);
    assert_eq!((*c1).1, 0.0);

    let d = AutoTuple::new((-1.0_f64, 1.0_f64));
    let cnst: AutoTuple<(f64,)> = 2.0_f64.into();
    let e1 = d * cnst;
    let e2 = d + cnst;
    assert_eq!(e1, AutoTuple::new((-2.0, 2.0)));
    assert_eq!(e2, AutoTuple::new((1.0, 3.0)));
}
