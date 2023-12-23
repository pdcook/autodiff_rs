use crate::arithmetic::*;
use crate::autodiffable::AutoDiffable;
use crate::diffable::Diffable;
use num::traits::bounds::UpperBounded;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct ADAdd<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A, B>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADAdd<A, B>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
    for<'b> B: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) + self.1.eval(x, static_args)
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        (f + g, df + dg)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSub<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A, B>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADSub<A, B>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
    for<'b> B: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) - self.1.eval(x, static_args)
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        (f - g, df - dg)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADMul<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A, B>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADMul<A, B>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
    for<'b> B: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) * self.1.eval(x, static_args)
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        // product rule, (fg)' = f'g + fg'
        (&f * &g, &df * &g + &f * &dg)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADDiv<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A, B>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADDiv<A, B>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
    for<'b> B: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) / self.1.eval(x, static_args)
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        // quotient rule, (f/g)' = (f'g - fg')/g^2
        // = (df * g - f * dg) / (g * g)
        (&f / &g, &df / &g - &f * &dg / (&g * &g))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADNeg<A>(pub A);

impl<StaticArgsType, InputType, OutputType, GradType, A>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADNeg<A>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        -self.0.eval(x, static_args)
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (-f, -df)
    }
}

#[derive(Debug, Clone)]
pub struct ADCompose<
    Outer,
    Inner,
    StaticArgsType,
    InputType,
    InnerOutputType,
    OuterOutputType,
    InnerGradType,
    OuterGradType,
>(
    pub Outer,
    pub Inner,
    pub  PhantomData<(
        StaticArgsType,
        InputType,
        InnerOutputType,
        OuterOutputType,
        InnerGradType,
        OuterGradType,
    )>,
)
where
    for<'b> Outer:
        AutoDiffable<'b, StaticArgsType, InnerOutputType, OuterOutputType, OuterGradType>,
    for<'b> Inner: AutoDiffable<'b, StaticArgsType, InputType, InnerOutputType, InnerGradType>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> InnerOutputType: WeakAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b InnerOutputType: CastingArithmetic<'b, InnerOutputType, InnerOutputType>
        + CastingArithmetic<'b, InnerGradType, InnerGradType>,
    for<'b> OuterOutputType: WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b OuterOutputType: CastingArithmetic<'b, OuterOutputType, OuterOutputType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> InnerGradType: StrongAssociatedArithmetic<'b, InnerOutputType>
        + WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b InnerGradType: CastingArithmetic<'b, InnerGradType, InnerGradType>
        + CastingArithmetic<'b, InnerOutputType, InnerGradType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> OuterGradType: StrongAssociatedArithmetic<'b, OuterOutputType>
        + StrongAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b OuterGradType: CastingArithmetic<'b, OuterGradType, OuterGradType>
        + CastingArithmetic<'b, OuterOutputType, OuterGradType>
        + CastingArithmetic<'b, InnerGradType, OuterGradType>;

// Implement Copy for ADCompose if Outer and Inner are Copy and
// all other types are Clone
impl<
        Outer: Copy,
        Inner: Copy,
        StaticArgsType: Clone,
        InputType: Clone,
        InnerOutputType: Clone,
        OuterOutputType: Clone,
        InnerGradType: Clone,
        OuterGradType: Clone,
    > Copy
    for ADCompose<
        Outer,
        Inner,
        StaticArgsType,
        InputType,
        InnerOutputType,
        OuterOutputType,
        InnerGradType,
        OuterGradType,
    >
where
    for<'b> Outer:
        AutoDiffable<'b, StaticArgsType, InnerOutputType, OuterOutputType, OuterGradType>,
    for<'b> Inner: AutoDiffable<'b, StaticArgsType, InputType, InnerOutputType, InnerGradType>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> InnerOutputType: WeakAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b InnerOutputType: CastingArithmetic<'b, InnerOutputType, InnerOutputType>
        + CastingArithmetic<'b, InnerGradType, InnerGradType>,
    for<'b> OuterOutputType: WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b OuterOutputType: CastingArithmetic<'b, OuterOutputType, OuterOutputType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> InnerGradType: StrongAssociatedArithmetic<'b, InnerOutputType>
        + WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b InnerGradType: CastingArithmetic<'b, InnerGradType, InnerGradType>
        + CastingArithmetic<'b, InnerOutputType, InnerGradType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> OuterGradType: StrongAssociatedArithmetic<'b, OuterOutputType>
        + StrongAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b OuterGradType: CastingArithmetic<'b, OuterGradType, OuterGradType>
        + CastingArithmetic<'b, OuterOutputType, OuterGradType>
        + CastingArithmetic<'b, InnerGradType, OuterGradType>,
{
}

impl<
        StaticArgsType,
        InputType,
        InnerOutputType,
        OuterOutputType,
        InnerGradType,
        OuterGradType,
        Outer,
        Inner,
    > Diffable<StaticArgsType, InputType, OuterOutputType, OuterGradType>
    for ADCompose<
        Outer,
        Inner,
        StaticArgsType,
        InputType,
        InnerOutputType,
        OuterOutputType,
        InnerGradType,
        OuterGradType,
    >
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> InnerOutputType: WeakAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b InnerOutputType: CastingArithmetic<'b, InnerOutputType, InnerOutputType>
        + CastingArithmetic<'b, InnerGradType, InnerGradType>,
    for<'b> OuterOutputType: WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b OuterOutputType: CastingArithmetic<'b, OuterOutputType, OuterOutputType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> InnerGradType: StrongAssociatedArithmetic<'b, InnerOutputType>
        + WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b InnerGradType: CastingArithmetic<'b, InnerGradType, InnerGradType>
        + CastingArithmetic<'b, InnerOutputType, InnerGradType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> OuterGradType: StrongAssociatedArithmetic<'b, OuterOutputType>
        + StrongAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b OuterGradType: CastingArithmetic<'b, OuterGradType, OuterGradType>
        + CastingArithmetic<'b, OuterOutputType, OuterGradType>
        + CastingArithmetic<'b, InnerGradType, OuterGradType>,
    for<'b> Outer:
        AutoDiffable<'b, StaticArgsType, InnerOutputType, OuterOutputType, OuterGradType>,
    for<'b> Inner: AutoDiffable<'b, StaticArgsType, InputType, InnerOutputType, InnerGradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OuterOutputType {
        self.0.eval(&self.1.eval(x, static_args), static_args)
    }

    fn eval_grad(
        &self,
        x: &InputType,
        static_args: &StaticArgsType,
    ) -> (OuterOutputType, OuterGradType) {
        let (inner, d_inner) = self.1.eval_grad(x, static_args);
        let (outer, d_outer) = self.0.eval_grad(&inner, static_args);

        // chain rule, (f(g(x)))' = f'(g(x)) * g'(x)
        (outer, d_outer * d_inner)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantAdd<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADConstantAdd<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) + &self.1
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);
        (f + &self.1, df)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantSub<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADConstantSub<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) - &self.1
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);
        (f - &self.1, df)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantMul<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADConstantMul<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) * &self.1
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);
        (f * &self.1, df * &self.1)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantDiv<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADConstantDiv<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        if self.1.is_zero() {
            panic!("Division by zero");
        }
        self.0.eval(x, static_args) / &self.1
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        if self.1.is_zero() {
            panic!("Division by zero");
        }
        let (f, df) = self.0.eval_grad(x, static_args);
        (f / &self.1, df / &self.1)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantPow<A, B>(pub A, pub B);

impl<StaticArgsType, InputType, OutputType, GradType, A>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADConstantPow<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedExtendedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedExtendedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).pow(&self.1)
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        let (f, df) = self.0.eval_grad(x, static_args);

        // chain rule on power, (f(x)^g)' = f(x)^(p-1) * p * f'(x)

        if self.1.is_zero() {
            return (f.one(), df.zero());
        } else if self.1.is_one() {
            return (f, df);
        }

        let g = &self.1;

        // g.one() provided by InstOne, in Arithmetic
        (f.clone().pow(&self.1), f.pow(g - g.one()) * g * df)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADAbs<A>(pub A);

impl<StaticArgsType, InputType, OutputType: num::traits::Signed, GradType, A>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADAbs<A>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedExtendedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedExtendedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).abs()
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        // chain rule on abs, (|f(x)|)' = f'(x) * sign(f(x))

        let (f, df) = self.0.eval_grad(x, static_args);

        (f.abs(), df * f.signum())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSignum<A>(pub A);

impl<StaticArgsType, InputType, OutputType: num::traits::Signed, GradType: UpperBounded, A>
    Diffable<StaticArgsType, InputType, OutputType, GradType> for ADSignum<A>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedExtendedArithmetic<'b, GradType>,
    for<'b> &'b OutputType:
        CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedExtendedArithmetic<'b, OutputType>,
    for<'b> &'b GradType:
        CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).signum()
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        // chain rule on signum, (sign(f(x)))' = 2 delta(x)
        // we approximate delta(x) as
        // delta(x) = GradType::MAX if x == 0, 0 otherwise

        let (f, df) = self.0.eval_grad(x, static_args);

        if x.is_zero() {
            (f.signum(), df * GradType::max_value())
        } else {
            (f.signum(), df.zero())
        }
    }
}
