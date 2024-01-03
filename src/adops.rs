use crate::autodiffable::AutoDiffable;
use crate::traits::{InstOne, InstZero};
use num::traits::bounds::UpperBounded;
use num::traits::{Pow, Signed};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy)]
pub struct ADAdd<A, B>(pub A, pub B);

impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> AutoDiffable<StaticArgs> for ADAdd<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Add<BOutput, Output = Output>,
{
    type Input = Input;
    type Output = Output;

    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        // use .add instead of + to allow for newtypes which implement Deref
        self.0.eval(x, static_args).add(self.1.eval(x, static_args))
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_grad(x, dx, static_args);

        (f.add(g), df.add(dg))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSub<A, B>(pub A, pub B);

impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> AutoDiffable<StaticArgs> for ADSub<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Sub<BOutput, Output = Output>,
{
    type Input = Input;
    type Output = Output;

    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        // use .sub instead of + to allow for newtypes which implement Deref
        self.0.eval(x, static_args).sub(self.1.eval(x, static_args))
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_grad(x, dx, static_args);

        (f.sub(g), df.sub(dg))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADMul<A, B>(pub A, pub B);

impl<StaticArgs, Input, Output, AOutput, BOutput, A, B> AutoDiffable<StaticArgs> for ADMul<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    // make sure A and B are both Clone
    AOutput: Clone,
    BOutput: Clone,
    // make sure A * B is defined and Output = A * B
    AOutput: Mul<BOutput, Output = Output>,
    // make sure A * B + A * B is defined and Output = A * B + A * B
    Output: Add<Output, Output = Output>,
{
    type Input = Input;
    type Output = Output;

    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        // use .mul instead of * to allow for newtypes which implement Deref
        self.0.eval(x, static_args).mul(self.1.eval(x, static_args))
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_grad(x, dx, static_args);

        // f * g : AOutput: Mul<BOutput>
        //
        // df * g : AOutput: Mul<BOutput>
        // dg * f : BOutput: Mul<AOutput>
        // df * g + dg * f : <AOutput as Mul<BOutput>>::Output: Add<<BOutput as Mul<AOutput>>::Output>
        //

        (f.clone().mul(g.clone()), df.mul(g).add(f.mul(dg)))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADDiv<A, B>(pub A, pub B);

impl<StaticArgs, Input, Output, AOutput, BOutput, BB, AB, ABOvBB, A, B> AutoDiffable<StaticArgs>
    for ADDiv<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    // ensure f and g are both Clone
    AOutput: Clone,
    BOutput: Clone,
    // ensure A/B is defined and Output = A/B
    AOutput: Div<BOutput, Output = Output>,
    // ensure B^2 is defined
    BOutput: Mul<BOutput, Output = BB>,
    // ensure A*B is defined (f * dg)
    AOutput: Mul<BOutput, Output = AB>,
    // ensure AB/B^2 is defined (f * dg) / g^2
    AB: Div<BB, Output = ABOvBB>,
    // ensure A/B - AB/B^2 is defined (df/g - f dg/g^2) and is Output
    Output: Sub<ABOvBB, Output = Output>,
{
    type Input = Input;
    type Output = Output;

    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        // use .div instead of / to allow for newtypes which implement Deref
        self.0.eval(x, static_args).div(self.1.eval(x, static_args))
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_grad(x, dx, static_args);

        // d(f/g) = (df*g - f*dg)/g^2 = df/g - f*dg/g^2
        // = (df/g - (f*dg)/(g*g))

        (
            f.clone().div(g.clone()),
            df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g))),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADNeg<A>(pub A);

impl<StaticArgs, Input, A> AutoDiffable<StaticArgs> for ADNeg<A>
where
    // ensure A has Neg
    <A as AutoDiffable<StaticArgs>>::Output: Neg,
    A: AutoDiffable<StaticArgs, Input = Input>,
{
    type Input = Input;
    type Output = <<A as AutoDiffable<StaticArgs>>::Output as Neg>::Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        // use .neg instead of - to allow for newtypes which implement Deref
        self.0.eval(x, static_args).neg()
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);

        (f.neg(), df.neg())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADCompose<Outer, Inner>(pub Outer, pub Inner);

impl<StaticArgs, InnerInput, InnerOutput, OuterInput, OuterOutput, Outer, Inner>
    AutoDiffable<StaticArgs> for ADCompose<Outer, Inner>
where
    Outer: AutoDiffable<StaticArgs, Input = OuterInput, Output = OuterOutput>,
    Inner: AutoDiffable<StaticArgs, Input = InnerInput, Output = InnerOutput>,
    OuterInput: From<InnerOutput>,
{
    type Input = InnerInput;
    type Output = OuterOutput;

    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0
            .eval(&self.1.eval(x, static_args).into(), static_args)
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (g, dg) = self.1.eval_grad(x, dx, static_args);
        self.0.eval_grad(&g.into(), &dg.into(), static_args)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantAdd<A, B>(pub A, pub B);

impl<StaticArgs, Input, Output, AOutput, A, B> AutoDiffable<StaticArgs> for ADConstantAdd<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A + B is defined and Output = A + B
    AOutput: Add<B, Output = Output>,
    // ensure B is Clone and B.zero is defined
    B: Clone + InstZero,
{
    type Input = Input;
    type Output = Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args).add(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);

        (f.add(self.1.clone()), df.add(self.1.zero()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantSub<A, B>(pub A, pub B);

impl<StaticArgs, Input, Output, AOutput, A, B> AutoDiffable<StaticArgs> for ADConstantSub<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A + B is defined and Output = A + B
    AOutput: Sub<B, Output = Output>,
    // ensure B is Clone and B.zero is defined
    B: Clone + InstZero,
{
    type Input = Input;
    type Output = Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args).sub(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);

        (f.sub(self.1.clone()), df.sub(self.1.zero()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantMul<A, B>(pub A, pub B);

impl<StaticArgs, Input, Output, AOutput, A, B> AutoDiffable<StaticArgs> for ADConstantMul<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A * B is defined and Output = A * B
    AOutput: Mul<B, Output = Output>,
    // ensure B is Clone
    B: Clone,
{
    type Input = Input;
    type Output = Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args).mul(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);

        (f.mul(self.1.clone()), df.mul(self.1.clone()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantDiv<A, B>(pub A, pub B);

impl<StaticArgs, Input, Output, AOutput, A, B> AutoDiffable<StaticArgs> for ADConstantDiv<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A / B is defined and Output = A / B
    AOutput: Div<B, Output = Output>,
    // ensure B is Clone
    B: Clone,
{
    type Input = Input;
    type Output = Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args).div(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);

        (f.div(self.1.clone()), df.div(self.1.clone()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantPow<A, B>(pub A, pub B);

impl<StaticArgs, Input, Output, AOutput, A, B> AutoDiffable<StaticArgs> for ADConstantPow<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A is Clone and A^B is defined and is Output
    AOutput: Clone + Pow<B, Output = Output>,
    // ensure B is Clone and B.one is defined and B-1 is B
    B: Clone + InstOne + Sub<B, Output = B>,
    // ensure A * A^B * B is defined and is Output
    Output: Mul<B>,
    AOutput: Mul<<Output as Mul<B>>::Output, Output = Output>,
{
    type Input = Input;
    type Output = Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args).pow(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);

        // d(f^p) = p * f^(p-1) * df
        // = df * ((f^(p-1)) * p)

        (
            f.clone().pow(self.1.clone()),
            (df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone()))),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADAbs<A>(pub A);

impl<StaticArgs, Input, Output, A> AutoDiffable<StaticArgs> for ADAbs<A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Output: Signed + Mul<Output, Output = Output>,
{
    type Input = Input;
    type Output = Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args).abs()
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        let (f, df) = self.0.eval_grad(x, dx, static_args);

        (f.abs(), df.mul(f.signum()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSignum<A>(pub A);

impl<StaticArgs, Input, Output, A> AutoDiffable<StaticArgs> for ADSignum<A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Output: InstZero + Signed + UpperBounded,
{
    type Input = Input;
    type Output = Output;

    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args).signum()
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        // chain rule on signum, (sign(f(x)))' = 2 delta(f(x))
        // we approximate delta(x) as
        // delta(x) = Grad::MAX if x == 0, 0 otherwise

        let (f, df) = self.0.eval_grad(x, dx, static_args);

        if InstZero::is_zero(&f) {
            (f.signum(), Self::Output::max_value())
        } else {
            (f.signum(), df.zero())
        }
    }
}
