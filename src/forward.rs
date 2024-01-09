use std::ops::Mul;
use crate::gradienttype::GradientType;

/// Multiplication used for df/dx * dx -> df
/// s:
/// `x: Input`
/// `f(x): Output`
/// `df/dx(x): Self`
/// `dx: OtherGrad`
/// `df: Result`
///
/// This is also the multiplication used in the chain rule
/// `df(g(x)) = df/dg(g(x)).forward_mul<x's type, f(x)'s type, dg's type>(dg/dx(x))`
/// where
/// `x: Input`,
/// `f(g(x)): Output`,
/// `df/dg(g(x)): Self`,
/// `dg/dx(x): OtherGrad`
/// `df/dx(x): Result`
pub trait ForwardMul<SelfInput, OtherGrad>
{
    type ResultGrad;
    fn forward_mul(&self, other: &OtherGrad) -> Self::ResultGrad;
}

// impl forward for simple types (commutative multiplication)

macro_rules! impl_forward_mul {
    ($($t:ty),*) => {
        $(
            impl<SelfInput, OtherGrad, ResultGrad> ForwardMul<SelfInput, OtherGrad> for $t
            where
                for<'a,'b> &'a $t: Mul<&'b OtherGrad, Output = ResultGrad>,
            {
                type ResultGrad = ResultGrad;
                fn forward_mul(
                    &self,
                    other: &OtherGrad,
                ) ->  Self::ResultGrad {
                    self * other
                }
            }
        )*
    };
}

impl_forward_mul!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, isize, usize);
