use crate::arithmetic::*;
use crate::autodiffable::AutoDiffable;
use num::traits::bounds::UpperBounded;
use num::traits::Signed;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct ADAdd<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
        B,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADAdd<A, B>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
    for<'b> B: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) + self.1.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args) + self.1.grad(x, static_args)
    }
}

#[derive(Debug, Clone)]
pub struct ADSub<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
        B,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADSub<A, B>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
    for<'b> B: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) - self.1.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args) - self.1.grad(x, static_args)
    }
}

#[derive(Debug, Clone)]
pub struct ADMul<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
        B,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADMul<A, B>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
    for<'b> B: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) * self.1.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        // product rule, (fg)' = f'g + fg'

        self.0.grad(x, static_args) * self.1.eval(x, static_args)
            + self.0.eval(x, static_args) * self.1.grad(x, static_args)
    }
}

#[derive(Debug, Clone)]
pub struct ADDiv<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
        B,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADDiv<A, B>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
    for<'b> B: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
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
        f_prime / &g - f * g_prime / (&g * &g)
    }
}

#[derive(Debug, Clone)]
pub struct ADNeg<A>(pub A);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADNeg<A>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        -self.0.eval(x, static_args)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        -self.0.grad(x, static_args)
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
    for<'b> Outer: AutoDiffable<'b, StaticArgsType, InnerOutputType, OuterOutputType, OuterGradType>,
    for<'b> Inner: AutoDiffable<'b, StaticArgsType, InputType, InnerOutputType, InnerGradType>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> InnerOutputType: WeakAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b InnerOutputType: CastingArithmetic<'b, InnerOutputType, InnerOutputType>
        + CastingArithmetic<'b, InnerGradType, InnerGradType>,
    for<'b> OuterOutputType: WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b OuterOutputType: CastingArithmetic<'b, OuterOutputType, OuterOutputType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> InnerGradType: StrongAssociatedArithmetic<'b, InnerOutputType> + WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b InnerGradType: CastingArithmetic<'b, InnerGradType, InnerGradType>
        + CastingArithmetic<'b, InnerOutputType, InnerGradType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> OuterGradType: StrongAssociatedArithmetic<'b, OuterOutputType> + StrongAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b OuterGradType: CastingArithmetic<'b, OuterGradType, OuterGradType>
        + CastingArithmetic<'b, OuterOutputType, OuterGradType>
        + CastingArithmetic<'b, InnerGradType, OuterGradType>;

impl<'a,
        StaticArgsType,
        InputType,
        InnerOutputType,
        OuterOutputType,
        InnerGradType,
        OuterGradType,
        Outer,
        Inner,
    > AutoDiffable<'a, StaticArgsType, InputType, OuterOutputType, OuterGradType>
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
    for<'b> InnerGradType: StrongAssociatedArithmetic<'b, InnerOutputType> + WeakAssociatedArithmetic<'b, OuterGradType>,
    for<'b> &'b InnerGradType: CastingArithmetic<'b, InnerGradType, InnerGradType>
        + CastingArithmetic<'b, InnerOutputType, InnerGradType>
        + CastingArithmetic<'b, OuterGradType, OuterGradType>,
    for<'b> OuterGradType: StrongAssociatedArithmetic<'b, OuterOutputType> + StrongAssociatedArithmetic<'b, InnerGradType>,
    for<'b> &'b OuterGradType: CastingArithmetic<'b, OuterGradType, OuterGradType>
        + CastingArithmetic<'b, OuterOutputType, OuterGradType>
        + CastingArithmetic<'b, InnerGradType, OuterGradType>,
    for<'b> Outer: AutoDiffable<'b, StaticArgsType, InnerOutputType, OuterOutputType, OuterGradType>,
    for<'b> Inner: AutoDiffable<'b, StaticArgsType, InputType, InnerOutputType, InnerGradType>,
{
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

#[derive(Debug, Clone)]
pub struct ADConstantAdd<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADConstantAdd<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) + &self.1
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args)
    }
}

#[derive(Debug, Clone)]
pub struct ADConstantSub<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADConstantSub<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) - &self.1
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args)
    }
}

#[derive(Debug, Clone)]
pub struct ADConstantMul<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADConstantMul<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args) * &self.1
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args) * &self.1
    }
}

#[derive(Debug, Clone)]
pub struct ADConstantDiv<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADConstantDiv<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        if self.1.is_zero() {
            panic!("Division by zero");
        }
        self.0.eval(x, static_args) / &self.1
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        if self.1.is_zero() {
            panic!("Division by zero");
        }
        self.0.grad(x, static_args) / &self.1
    }
}

#[derive(Debug, Clone)]
pub struct ADConstantPow<A, B>(pub A, pub B);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType,
        GradType,
        A,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADConstantPow<A, OutputType>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedExtendedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedExtendedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).pow(&self.1)
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        // chain rule on power, (f(x)^g)' = f(x)^(p-1) * p * f'(x)

        let f_prime = self.0.grad(x, static_args);

        if self.1.is_zero() {
            return f_prime.zero();
        } else if self.1.is_one() {
            return f_prime;
        }

        let f = self.0.eval(x, static_args);

        let g = &self.1;

        // g.one() provided by InstOne, in Arithmetic
        let f_prime_of_g = f.pow(g - g.one()) * g * f_prime;

        f_prime_of_g
    }
}

#[derive(Debug, Clone)]
pub struct ADAbs<A>(pub A);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType: Signed,
        GradType,
        A,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADAbs<A>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedExtendedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedExtendedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
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

#[derive(Debug, Clone)]
pub struct ADSignum<A>(pub A);

impl<'a,
        StaticArgsType,
        InputType,
        OutputType: Signed,
        GradType: UpperBounded,
        A,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for ADSignum<A>
where
    for<'b> InputType: Arithmetic<'b>,
    for<'b> &'b InputType: CastingArithmetic<'b, InputType, InputType>,
    for<'b> OutputType: WeakAssociatedExtendedArithmetic<'b, GradType>,
    for<'b> &'b OutputType: CastingArithmetic<'b, OutputType, OutputType> + CastingArithmetic<'b, GradType, GradType>,
    for<'b> GradType: StrongAssociatedExtendedArithmetic<'b, OutputType>,
    for<'b> &'b GradType: CastingArithmetic<'b, GradType, GradType> + CastingArithmetic<'b, OutputType, GradType>,
    for<'b> A: AutoDiffable<'b, StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).signum()
    }

    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        // chain rule on signum, (sign(f(x)))' = 2 delta(x)
        // we approximate delta(x) as
        // delta(x) = GradType::MAX if x == 0, 0 otherwise
        if x.is_zero() {
            GradType::max_value()
        } else {
            self.0.grad(x, static_args).zero()
        }
    }
}
