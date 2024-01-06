use std::ops::Mul;
use crate::gradienttype::GradientType;

/// Multiplication used for df/dx * dx -> df
/// s:
/// `x: SelfInput`
/// `f(x): SelfOutput`
/// `df/dx(x): Self`
/// `dx: OtherGrad`
/// `df: Result`
///
/// This is also the multiplication used in the chain rule
/// `df(g(x)) = df/dg(g(x)).forward_mul<x's type, f(x)'s type, dg's type>(dg/dx(x))`
/// where
/// `x: SelfInput`,
/// `f(g(x)): SelfOutput`,
/// `df/dg(g(x)): Self`,
/// `dg/dx(x)*dx: OtherGrad`
/// `df(x): Result`
pub trait ForwardMul<SelfInput, SelfOutput, OtherGrad>
where
    SelfInput: GradientType<SelfOutput>,
{
    type ResultGrad;

    fn forward_mul(self, other: &OtherGrad) -> Self::ResultGrad;
}

// impl forward mul for all simple types that implement Mul
// simple types are those such that <Input as GradientType<Output>>::GradientType = T
// and T * OtherGrad = Output
impl<T, SelfInput, SelfOutput, OtherGrad> ForwardMul<SelfInput, SelfOutput, OtherGrad> for T
where
    SelfInput: GradientType<SelfOutput, GradientType = T>,
    for<'a> T: Mul<&'a OtherGrad, Output = SelfOutput>,
{
    type ResultGrad = SelfOutput;

    fn forward_mul(
        self,
        other: &OtherGrad,
    ) ->  Self::ResultGrad {
        self * other
    }
}
