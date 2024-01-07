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
pub trait ForwardMul<Input, Output, OtherGrad, ResultGrad>
where
    Input: GradientType<Output, GradientType = Self>,
{
    fn forward_mul(self, other: &OtherGrad) -> ResultGrad;
}

// impl forward for simple types (commutative multiplication)

macro_rules! impl_forward_mul {
    ($($t:ty),*) => {
        $(
            impl<Input, Output, OtherGrad> ForwardMul<Input, Output, OtherGrad, Output> for $t
            where
                Input: GradientType<Output, GradientType = $t>,
                $t: Mul<OtherGrad, Output = Output>,
                for<'a> $t: Mul<&'a OtherGrad, Output = Output>,
                OtherGrad: Mul<$t, Output = Output>,
                for<'a> OtherGrad: Mul<&'a $t, Output = Output>,
            {
                fn forward_mul(
                    self,
                    other: &OtherGrad,
                ) ->  Output {
                    self * other
                }
            }
        )*
    };
}

impl_forward_mul!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, isize, usize);

/*
// impl forward mul for all simple types that implement Mul
// simple types are those such that <Input as GradientType<Output>>::GradientType = T
// and T * OtherGrad = Output
impl<T, Input, Output, OtherGrad> ForwardMul<Input, Output, OtherGrad, Output> for T
where
    Input: GradientType<Output, GradientType = T>,
    T: Mul<OtherGrad, Output = Output>,
    for<'a> T: Mul<&'a OtherGrad, Output = Output>,
    OtherGrad: Mul<T, Output = Output>,
    for<'a> OtherGrad: Mul<&'a T, Output = Output>,
{
    fn forward_mul(
        self,
        other: &OtherGrad,
    ) ->  Output {
        self * other
    }
}*/
