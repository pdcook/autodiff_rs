use std::ops::Mul;
use crate::gradienttype::GradientType;

/// Multiplication used for df/dx * dx -> df
/// Types:
/// `x: Input`
/// `f(x): Output`
/// `dx: Grad`
///
/// This is also the multiplication used in the chain rule
/// `d/dx f(g(x)) = df/dx(g(x)).forward_mul<x's type, f(x)'s type, dg's type>(dg/dx(x))`
/// where
/// `x: Input`,
/// `f(x): Output`,
/// `dg/dx(x): Grad`
pub trait ForwardMul<InputType, OutputType, GradType>
where InputType: GradientType<OutputType>,
{
    fn forward_mul(self, _other: GradType) -> <InputType as GradientType<OutputType>>::GradientType;
}

// impl forward mul for all simple types that implement Mul
// simple types are those such that <InputType as GradientType<OutputType>>::GradientType = OutputType
impl<T, InputType, OutputType, GradType> ForwardMul<InputType, OutputType, GradType> for T
where
    InputType: GradientType<OutputType, GradientType = OutputType>,
    T: Mul<GradType, Output = OutputType>,
{
    fn forward_mul(
        self,
        other: GradType,
    ) ->  OutputType {
        self * other
    }
}
