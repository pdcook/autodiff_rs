use crate::arithmetic::*;
use crate::autodiffable::AutoDiffable;
use num::traits::bounds::UpperBounded;
use num::traits::Signed;

#[derive(Debug, Clone, Copy)]
pub struct ADAdd<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
        B: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADAdd<A, B>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) + self.1.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args) + self.1.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSub<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
        B: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADSub<A, B>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) - self.1.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args) - self.1.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADMul<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
        B: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADMul<A, B>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) * self.1.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        // product rule, (fg)' = f'g + fg'

        self.0.grad(x, static_args) * self.1.eval(x, static_args)
            + self.0.eval(x, static_args) * self.1.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADDiv<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
        B: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADDiv<A, B>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) / self.1.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        // quotient rule, (f/g)' = (f'g - fg')/g^2

        let f = self.0.eval(x, static_args);
        let g = self.1.eval(x, static_args);

        let f_prime = self.0.grad(x, static_args);
        let g_prime = self.1.grad(x, static_args);

        //(f_prime * g - f * g_prime) / (g * g)
        f_prime / &g - f * g_prime / (&g * g)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADNeg<A>(pub A);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADNeg<A>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        -self.0.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        -self.0.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADCompose<Outer, Inner>(pub Outer, pub Inner);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        InnerOutputType: WeakAssociatedArithmetic<InnerGradType>,
        OuterOutputType: WeakAssociatedArithmetic<OuterGradType>,
        InnerGradType: StrongAssociatedArithmetic<InnerOutputType> + WeakAssociatedArithmetic<OuterGradType>,
        OuterGradType: StrongAssociatedArithmetic<OuterOutputType> + StrongAssociatedArithmetic<InnerGradType>,
        Outer: AutoDiffable<
            StaticArgsType,
            InType = InnerOutputType,
            OutType = OuterOutputType,
            GradType = OuterGradType,
        >,
        Inner: AutoDiffable<
            StaticArgsType,
            InType = InputType,
            OutType = InnerOutputType,
            GradType = InnerGradType,
        >,
    > AutoDiffable<StaticArgsType> for ADCompose<Outer, Inner>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a InnerOutputType: CastingArithmetic<InnerOutputType, InnerOutputType>
        + CastingArithmetic<InnerGradType, InnerGradType>,
    for<'a> &'a OuterOutputType: CastingArithmetic<OuterOutputType, OuterOutputType>
        + CastingArithmetic<OuterGradType, OuterGradType>,
    for<'a> &'a InnerGradType: CastingArithmetic<InnerGradType, InnerGradType>
        + CastingArithmetic<InnerOutputType, InnerGradType>
        + CastingArithmetic<OuterGradType, OuterGradType>,
    for<'a> &'a OuterGradType: CastingArithmetic<OuterGradType, OuterGradType>
        + CastingArithmetic<OuterOutputType, OuterGradType>
        + CastingArithmetic<InnerGradType, OuterGradType>,
{
    type InType = InputType;
    type OutType = OuterOutputType;
    type GradType = OuterGradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OuterOutputType {
        self.0.eval(&self.1.eval(x, static_args), static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> OuterGradType {
        // chain rule, (f(g(x)))' = f'(g(x)) * g'(x)
        let inner_output = self.1.eval(x, static_args);
        let inner_grad = self.1.grad(x, static_args);

        let outer_grad = self.0.grad(&inner_output, static_args);

        outer_grad * inner_grad
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantAdd<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADConstantAdd<A, OutputType>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) + self.1
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantSub<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADConstantSub<A, OutputType>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) - self.1
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantMul<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADConstantMul<A, OutputType>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) * self.1
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args) * self.1
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantDiv<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADConstantDiv<A, OutputType>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        if self.1.is_zero() {
            panic!("Division by zero");
        }
        self.0.eval(x, static_args) / self.1
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        if self.1.is_zero() {
            panic!("Division by zero");
        }
        self.0.grad(x, static_args) / self.1
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantPow<A, B>(pub A, pub B);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedExtendedArithmetic<GradType>,
        GradType: StrongAssociatedExtendedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADConstantPow<A, OutputType>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).pow(self.1)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        // chain rule on power, (f(x)^g)' = f(x)^(p-1) * p * f'(x)

        if self.1.is_zero() {
            return GradType::zero();
        } else if self.1.is_one() {
            return self.0.grad(x, static_args);
        }

        let f = self.0.eval(x, static_args);
        let f_prime = self.0.grad(x, static_args);

        let g = self.1;

        let f_prime_of_g = f.pow(g - OutputType::one()) * g * f_prime;

        f_prime_of_g
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADAbs<A>(pub A);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedExtendedArithmetic<GradType> + Signed,
        GradType: StrongAssociatedExtendedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADAbs<A>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).abs()
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        // chain rule on abs, (|f(x)|)' = f'(x) * sign(f(x))

        let f = self.0.eval(x, static_args);
        let f_prime = self.0.grad(x, static_args);

        let f_prime_of_abs = f_prime * f.signum();

        f_prime_of_abs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSignum<A>(pub A);

impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedExtendedArithmetic<GradType> + Signed,
        GradType: StrongAssociatedExtendedArithmetic<OutputType> + UpperBounded,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType> for ADSignum<A>
where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).signum()
    }

    fn grad(&self, x: &InputType, _: &StaticArgsType) -> GradType {
        // chain rule on signum, (sign(f(x)))' = 2 delta(x)
        // we approximate delta(x) as
        // delta(x) = GradType::MAX if x == 0, 0 otherwise
        if x.is_zero() {
            GradType::max_value()
        } else {
            GradType::zero()
        }
    }
}
