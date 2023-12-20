use crate::complex::adops::*;
use crate::complex::arithmetic::*;
use crate::autodiffable::AutoDiffable;
use crate::complex::func_traits;
use std::marker::PhantomData;
use num::traits::Float;
use crate::autodiff::AutoDiff;

/// Impl of ComplexFunc for AutoDiff
impl<
        InputF: Float,
        OutputF: Float,
        GradF: Float,
        StaticArgsType,
        InputType: ComplexArithmetic<InputF>,
        OutputType: ComplexWeakAssociatedArithmetic<OutputF, GradF, GradType>,
        GradType: ComplexStrongAssociatedArithmetic<GradF, OutputF, OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > func_traits::ComplexFunc for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type ReType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADRe<A, InputF, OutputF, GradF>>;
    type ImType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADIm<A, InputF, OutputF, GradF>>;
    type NormSqrType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADNormSqr<A, InputF, OutputF, GradF>>;
    type NormType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADNorm<A, InputF, OutputF, GradF>>;
    type ArgType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADArg<A, InputF, OutputF, GradF>>;

    fn re(self) -> Self::ReType {
        AutoDiff(ADRe(self.0, PhantomData), PhantomData)
    }

    fn im(self) -> Self::ImType {
        AutoDiff(ADIm(self.0, PhantomData), PhantomData)
    }

    fn norm_sqr(self) -> Self::NormSqrType {
        AutoDiff(ADNormSqr(self.0, PhantomData), PhantomData)
    }

    fn norm(self) -> Self::NormType {
        AutoDiff(ADNorm(self.0, PhantomData), PhantomData)
    }

    fn arg(self) -> Self::ArgType {
        AutoDiff(ADArg(self.0, PhantomData), PhantomData)
    }
}
