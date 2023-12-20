use num::traits::Float;
use num::complex::Complex;
use crate::complex::arithmetic::*;
use crate::autodiffable::AutoDiffable;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct ADRe<A,F1,F2,F3>(pub A, pub PhantomData<(F1, F2, F3)>);

impl<
        InputF: Float,
        OutputF: Float,
        GradF: Float,
        StaticArgsType,
        InputType: ComplexArithmetic<InputF>,
        OutputType: ComplexWeakAssociatedArithmetic<OutputF, GradF, GradType>,
        GradType: ComplexStrongAssociatedArithmetic<GradF, OutputF, OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADRe<A, InputF, OutputF, GradF>
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).re()
    }

    fn grad(&self, x: InputType, static_args: &StaticArgsType) -> GradType {
        // d/dz = d/da - i d/db
        // d/dz Re(f) = (d/da Re(x) - i d/db Re(x))[x = f(z)] * f'
        //            = f'
        self.0.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADIm<A, F1, F2, F3>(pub A, pub PhantomData<(F1, F2, F3)>);

impl<
        InputF: Float,
        OutputF: Float,
        GradF: Float,
        StaticArgsType,
        InputType: ComplexArithmetic<InputF>,
        OutputType: ComplexWeakAssociatedArithmetic<OutputF, GradF, GradType>,
        GradType: ComplexStrongAssociatedArithmetic<GradF, OutputF, OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADIm<A, InputF, OutputF, GradF>
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).im()
    }

    fn grad(&self, x: InputType, static_args: &StaticArgsType) -> GradType {
        // d/dz = d/da - i d/db
        // d/dz Im(f) = (d/da Im(x) - i d/db Im(x))[x = f(z)] * f'
        //            = -i f'
        -OutputType::i() * self.0.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADNormSqr<A, F1, F2, F3>(pub A, pub PhantomData<(F1, F2, F3)>);

impl<
        InputF: Float,
        OutputF: Float,
        GradF: Float,
        StaticArgsType,
        InputType: ComplexArithmetic<InputF>,
        OutputType: ComplexWeakAssociatedArithmetic<OutputF, GradF, GradType>,
        GradType: ComplexStrongAssociatedArithmetic<GradF, OutputF, OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADNormSqr<A, InputF, OutputF, GradF>
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).norm_sqr()
    }

    fn grad(&self, x: InputType, static_args: &StaticArgsType) -> GradType {
        // d/dz = d/da - i d/db
        // d/dz |f|^2 = (d/da a^2 - i d/db b^2)[a + ib = f(z)] * f'
        //            = 2a f' - 2i b f'
        //            = 2 f.conj() f'
        OutputType::new(OutputF::from(2.0).unwrap(), OutputF::from(0.0).unwrap()) * self.0.eval(x, static_args).conj() * self.0.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADNorm<A, F1, F2, F3>(pub A, pub PhantomData<(F1, F2, F3)>);

impl<
        InputF: Float,
        OutputF: Float,
        GradF: Float,
        StaticArgsType,
        InputType: ComplexArithmetic<InputF>,
        OutputType: ComplexWeakAssociatedArithmetic<OutputF, GradF, GradType>,
        GradType: ComplexStrongAssociatedArithmetic<GradF, OutputF, OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADNorm<A, InputF, OutputF, GradF>
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).norm()
    }

    fn grad(&self, x: InputType, static_args: &StaticArgsType) -> GradType {
        // d/dz = d/da - i d/db
        // d/dz |f| = (d/da sqrt(a^2 + b^2) - i d/db sqrt(a^2 + b^2))[a + ib = f(z)] * f'
        //          = 0.5 * (a / |f|) f' - i (b / |f|) f'
        //          = 0.5 * f / |f| f'
        OutputType::new(OutputF::from(0.5).unwrap(), OutputF::from(0.0).unwrap()) * self.0.eval(x, static_args) * self.0.grad(x, static_args) / self.0.eval(x, static_args).norm()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADArg<A, F1, F2, F3>(pub A, pub PhantomData<(F1, F2, F3)>);

impl<
        InputF: Float,
        OutputF: Float,
        GradF: Float,
        StaticArgsType,
        InputType: ComplexArithmetic<InputF>,
        OutputType: ComplexWeakAssociatedArithmetic<OutputF, GradF, GradType>,
        GradType: ComplexStrongAssociatedArithmetic<GradF, OutputF, OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADArg<A, InputF, OutputF, GradF>
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).arg()
    }

    fn grad(&self, x: InputType, static_args: &StaticArgsType) -> GradType {
        // d/dz = d/da - i d/db
        // d/dz arg(f) = (d/da arg(a + ib) - i d/db arg(a + ib))[a + ib = f(z)] * f'
        //             = (d/da atan2(b, a) - i d/db atan2(b, a))[a + ib = f(z)] * f'
        //             = (-b / (a^2 + b^2) - i (a / (a^2 + b^2)))[a + ib = f(z)] * f'
        //             = (-b - i a) / (a^2 + b^2) * f'
        //             = -i * f.conj() / |f|^2 * f'
        //             = -i * f' / f
        -OutputType::i() * self.0.grad(x, static_args) / self.0.eval(x, static_args)
    }
}
