use crate::autodiffable::{AutoDiffable, ForwardDiffable};
use crate::traits::{InstOne, InstZero};
use num::traits::bounds::UpperBounded;
use num::traits::{Pow, Signed};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::marker::PhantomData;
use crate::forward::ForwardMul;
use crate::gradienttype::GradientType;
use crate::diffable::Diffable;

use crate as autodiff;
use forwarddiffable_derive::*;

#[derive(Debug, Clone, Copy)]
pub struct ADCoerce<A, NewInput, NewOutput>(pub A, pub PhantomData<(NewInput, NewOutput)>);

impl<A: Diffable<StaticArgs>, NewInput, NewOutput, StaticArgs> Diffable<StaticArgs> for ADCoerce<A, NewInput, NewOutput> {
    type Input = NewInput;
    type Output = NewOutput;
}

// coerce A's input and output to match NewInput and NewOutput
impl<StaticArgs, Input, Output, Grad, NewInput, NewOutput, NewGradient, A> AutoDiffable<StaticArgs>
for ADCoerce<A, NewInput, NewOutput>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: GradientType<Output, GradientType = Grad>,
    NewInput: Clone + GradientType<NewOutput, GradientType = NewGradient>,
    Input: From<NewInput>,
    NewOutput: From<Output>,
    NewGradient: From<Grad>,
{
    ////type Input = NewInput;
    ////type Output = NewOutput;

    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(&x.clone().into(), static_args).into()
    }

    fn eval_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, NewGradient) {
        let (f, df) = self.0.eval_grad(&x.clone().into(), static_args);
        (f.into(), df.into())
    }
}

// impl ForwardDiffable for ADCoerce<A, NewInput, NewOutput>
impl<StaticArgs, Input, Output, NewInput, NewOutput, A> ForwardDiffable<StaticArgs> for ADCoerce<A, NewInput, NewOutput>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
    NewInput: Clone,
    Input: From<NewInput>,
    NewOutput: From<Output>,
{
    //type Input = NewInput;
    //type Output = NewOutput;

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(&x.clone().into(), &dx.clone().into(), static_args);
        (f.into(), df.into())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADAppendStaticArgs<A, NewStaticArgs>(pub A, pub PhantomData<NewStaticArgs>);

impl<A: Diffable<StaticArgs>, StaticArgs, NewStaticArgs> Diffable<(StaticArgs, NewStaticArgs)> for ADAppendStaticArgs<A, NewStaticArgs> {
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, NewStaticArgs, Input, Output, Gradient, A> AutoDiffable<(StaticArgs, NewStaticArgs)>
for ADAppendStaticArgs<A, NewStaticArgs>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: GradientType<Output, GradientType = Gradient>,
{
    //type Input = Input;
    //type Output = Output;

    fn eval(&self, x: &Self::Input, static_args: &(StaticArgs, NewStaticArgs)) -> Self::Output {
        self.0.eval(x, &static_args.0)
    }

    fn eval_grad(&self, x: &Self::Input, static_args: &(StaticArgs, NewStaticArgs)) -> (Self::Output, Gradient) {
        self.0.eval_grad(x, &static_args.0)
    }
}

impl<StaticArgs, NewStaticArgs, Input, Output, A> ForwardDiffable<(StaticArgs, NewStaticArgs)>
for ADAppendStaticArgs<A, NewStaticArgs>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &Self::Input, dx: &Self::Input, static_args: &(StaticArgs, NewStaticArgs)) -> (Self::Output, Self::Output) {
        self.0.eval_forward_grad(x, dx, &static_args.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADPrependStaticArgs<A, NewStaticArgs>(pub A, pub PhantomData<NewStaticArgs>);

impl<A: Diffable<StaticArgs>, StaticArgs, NewStaticArgs> Diffable<(NewStaticArgs, StaticArgs)> for ADPrependStaticArgs<A, NewStaticArgs> {
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, NewStaticArgs, Input, Output, Gradient, A> AutoDiffable<(NewStaticArgs, StaticArgs)>
for ADPrependStaticArgs<A, NewStaticArgs>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: GradientType<Output, GradientType = Gradient>,
{
    //type Input = Input;
    //type Output = Output;

    fn eval(&self, x: &Self::Input, static_args: &(NewStaticArgs, StaticArgs)) -> Self::Output {
        self.0.eval(x, &static_args.1)
    }

    fn eval_grad(&self, x: &Self::Input, static_args: &(NewStaticArgs, StaticArgs)) -> (Self::Output, Gradient) {
        self.0.eval_grad(x, &static_args.1)
    }
}

impl<StaticArgs, NewStaticArgs, Input, Output, A> ForwardDiffable<(NewStaticArgs, StaticArgs)>
for ADPrependStaticArgs<A, NewStaticArgs>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
{
    //type Input = Input;
    //type Output = Output;

    fn eval_forward_grad(&self, x: &Self::Input, dx: &Self::Input, static_args: &(NewStaticArgs, StaticArgs)) -> (Self::Output, Self::Output) {
        self.0.eval_forward_grad(x, dx, &static_args.1)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADAdd<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADAdd<A, B>
where
    A::Output: Add<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Add<B::Output>>::Output;
}

impl<StaticArgs, Input, AOutput, BOutput, AGrad, BGrad, Output, Grad, A, B> AutoDiffable<StaticArgs> for ADAdd<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    AOutput: Add<BOutput, Output = Output>,
    AGrad: Add<BGrad, Output = Grad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;

    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        // use .add instead of + to allow for newtypes which implement Deref
        self.0.eval(x, static_args).add(self.1.eval(x, static_args))
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        (f.add(g), df.add(dg))
    }
}

impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> ForwardDiffable<StaticArgs> for ADAdd<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Add<BOutput, Output = Output>,
{
    //type Input = Input;
    //type Output = Output;

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (f.add(g), df.add(dg))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSub<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADSub<A, B>
where
    A::Output: Sub<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Sub<B::Output>>::Output;
}

impl<StaticArgs, Input, AOutput, BOutput, AGrad, BGrad, Output, Grad, A, B> AutoDiffable<StaticArgs> for ADSub<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    AOutput: Sub<BOutput, Output = Output>,
    AGrad: Sub<BGrad, Output = Grad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;

    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        // use .sub instead of - to allow for newtypes which implement Deref
        self.0.eval(x, static_args).sub(self.1.eval(x, static_args))
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        (f.sub(g), df.sub(dg))
    }
}

impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> ForwardDiffable<StaticArgs> for ADSub<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Sub<BOutput, Output = Output>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (f.sub(g), df.sub(dg))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADMul<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADMul<A, B>
where
    A::Output: Mul<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Mul<B::Output>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, BOutput, AGrad, BGrad, DAB, ADB, A, B> AutoDiffable<StaticArgs> for ADMul<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    // make sure A and B are both Clone
    AOutput: Clone,
    BOutput: Clone,
    // make sure A * B is defined and Output = Output
    AOutput: Mul<BOutput, Output = Output>,
    // make sure dA * B is defined and Output = DAB
    AGrad: Mul<BOutput, Output = DAB>,
    // make sure A * dB is defined and Output = ADB
    AOutput: Mul<BGrad, Output = ADB>,
    // make sure DAB + ADB is defined and Output = Grad
    DAB: Add<ADB, Output = Grad>,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;

    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        // use .mul instead of * to allow for newtypes which implement Deref
        self.0.eval(x, static_args).mul(self.1.eval(x, static_args))
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        // f * g : AOutput: Mul<BOutput, Output = Output>
        //
        // df * g : AGrad: Mul<BOutput, Output = DAB>
        // f * dg : BGrad: Mul<AOutput, Output = ADB>
        // df * g + f * dg : DAB: Add<ADB, Output = Grad>

        (f.clone().mul(g.clone()), df.mul(g).add(f.mul(dg)))
    }
}

impl<StaticArgs, Input, Output, AOutput, BOutput, A, B> ForwardDiffable<StaticArgs> for ADMul<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    // make sure A and B are both Clone
    AOutput: Clone,
    BOutput: Clone,
    // make sure A * B is defined and Output = Output
    AOutput: Mul<BOutput, Output = Output>,
    // make sure A * B + B * A is defined and Output = Output
    Output: Add<Output, Output = Output>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        // f * g : AOutput: Mul<BOutput, Output = Output>
        //
        // df * g : AGrad: Mul<BOutput, Output = DAB>
        // f * dg : BGrad: Mul<AOutput, Output = ADB>
        // df * g + f * dg : DAB: Add<ADB, Output = Grad>

        (f.clone().mul(g.clone()), df.mul(g).add(f.mul(dg)))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADDiv<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADDiv<A, B>
where
    A::Output: Div<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Div<B::Output>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, BOutput, AGrad, BGrad, BB, ADB, DAOVB, ADBOVBB, A, B> AutoDiffable<StaticArgs>
    for ADDiv<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    // ensure f and g are both Clone
    AOutput: Clone,
    BOutput: Clone,
    // ensure A/B is defined and Output = A/B
    AOutput: Div<BOutput, Output = Output>,
    // ensure B^2 is defined
    BOutput: Mul<BOutput, Output = BB>,
    // ensure A*dB is defined (f * dg)
    AOutput: Mul<BGrad, Output = ADB>,
    // ensure dA/B is defined (df/g)
    AGrad: Div<BOutput, Output = DAOVB>,
    // ensure AdB/B^2 is defined (f * dg/g^2)
    ADB: Div<BB, Output = ADBOVBB>,
    // ensure dA/B - AdB/B^2 is defined (df/g - f * dg/g^2)
    DAOVB: Sub<ADBOVBB, Output = Grad>,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;

    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        // use .div instead of / to allow for newtypes which implement Deref
        self.0.eval(x, static_args).div(self.1.eval(x, static_args))
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        // d(f/g) = (df*g - f*dg)/g^2 = df/g - f*dg/g^2
        // = (df/g - (f*dg)/(g*g))

        (
            f.clone().div(g.clone()),
            df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g))),
        )
    }
}

impl<StaticArgs, Input, Output, AOutput, BOutput, BB, AB, ABOVBB, A, B> ForwardDiffable<StaticArgs>
    for ADDiv<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    // ensure f and g are both Clone
    AOutput: Clone,
    BOutput: Clone,
    // ensure A/B is defined and Output = Output
    // ensure dA/B is defined (df/g)
    AOutput: Div<BOutput, Output = Output>,
    // ensure B^2 is defined
    BOutput: Mul<BOutput, Output = BB>,
    // ensure A*dB is defined (f * dg)
    AOutput: Mul<BOutput, Output = AB>,
    // ensure AdB/B^2 is defined (f * dg/g^2)
    AB: Div<BB, Output = ABOVBB>,
    // ensure dA/B - AdB/B^2 is defined (df/g - f * dg/g^2)
    Output: Sub<ABOVBB, Output = Output>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

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

impl<A: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADNeg<A>
where
    A::Output: Neg,
{
    type Input = A::Input;
    type Output = <A::Output as Neg>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A> AutoDiffable<StaticArgs> for ADNeg<A>
where
    // ensure A has Neg
    AOutput: Neg<Output = Output>,
    AGrad: Neg<Output = Grad>,
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        // use .neg instead of - to allow for newtypes which implement Deref
        self.0.eval(x, static_args).neg()
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.neg(), df.neg())
    }
}

impl<StaticArgs, Input, Output, AOutput, A> ForwardDiffable<StaticArgs> for ADNeg<A>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A has Neg
    AOutput: Neg<Output = Output>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.neg(), df.neg())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADCompose<Outer, Inner>(pub Outer, pub Inner);

impl<Outer, Inner, StaticArgs> Diffable<StaticArgs> for ADCompose<Outer, Inner>
where
    Outer: Diffable<StaticArgs>,
    Inner: Diffable<StaticArgs>,
{
    type Input = Inner::Input;
    type Output = Outer::Output;
}

impl<StaticArgs, InnerInput, InnerOutput, InnerGrad, OuterInput, OuterOutput, OuterGrad, Grad, Outer, Inner>
    AutoDiffable<StaticArgs> for ADCompose<Outer, Inner>
where
    Outer: AutoDiffable<StaticArgs, Input = OuterInput, Output = OuterOutput>,
    Inner: AutoDiffable<StaticArgs, Input = InnerInput, Output = InnerOutput>,
    OuterInput: From<InnerOutput> + GradientType<OuterOutput, GradientType = OuterGrad>,
    InnerInput: GradientType<InnerOutput, GradientType = InnerGrad> + GradientType<OuterOutput, GradientType = Grad>,
    OuterGrad: ForwardMul<OuterInput, InnerGrad, ResultGrad = Grad>
{
    //type Input = InnerInput;
    //type Output = OuterOutput;

    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .eval(&self.1.eval(x, static_args).into(), static_args)
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (g, dg) = self.1.eval_grad(x, static_args);
        let (f, df) = self.0.eval_grad(&g.into(), static_args);
        (f, df.forward_mul(&dg))
    }
}

impl<StaticArgs, InnerInput, InnerOutput, OuterInput, OuterOutput, Outer, Inner>
    ForwardDiffable<StaticArgs> for ADCompose<Outer, Inner>
where
    Outer: ForwardDiffable<StaticArgs, Input = OuterInput, Output = OuterOutput>,
    Inner: ForwardDiffable<StaticArgs, Input = InnerInput, Output = InnerOutput>,
    OuterInput: From<InnerOutput>,
{
    //type Input = InnerInput;
    //type Output = OuterOutput;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);
        let (f, df) = self.0.eval_forward_grad(&g.into(), &dg.into(), static_args);
        (f, df)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantAdd<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantAdd<A, B>
where
    A::Output: Add<B>,
    B: Clone
{
    type Input = A::Input;
    type Output = <A::Output as Add<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs> for ADConstantAdd<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    // ensure A + B is defined and Output = Output
    AOutput: Add<B, Output = Output>,
    AGrad: Add<B, Output = Grad>,
    // ensure B is Clone and B.zero is defined
    B: Clone + InstZero,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).add(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.add(self.1.clone()), df.add(self.1.zero()))
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantAdd<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A + B is defined and Output = Output
    AOutput: Add<B, Output = Output>,
    B: Clone + InstZero,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.add(self.1.clone()), df.add(self.1.zero()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantSub<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantSub<A, B>
where
    A::Output: Sub<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Sub<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs> for ADConstantSub<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    // ensure A - B is defined and Output = Output
    AOutput: Sub<B, Output = Output>,
    AGrad: Sub<B, Output = Grad>,
    // ensure B is Clone and B.zero is defined
    B: Clone + InstZero,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).sub(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.sub(self.1.clone()), df.sub(self.1.zero()))
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantSub<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A - B is defined and Output = Output
    AOutput: Sub<B, Output = Output>,
    B: Clone + InstZero,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.sub(self.1.clone()), df.sub(self.1.zero()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantMul<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantMul<A, B>
where
    A::Output: Mul<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Mul<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs> for ADConstantMul<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    // ensure A * B is defined and Output = A * B
    AOutput: Mul<B, Output = Output>,
    AGrad: Mul<B, Output = Grad>,
    // ensure B is Clone
    B: Clone,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).mul(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.mul(self.1.clone()), df.mul(self.1.clone()))
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantMul<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A * B is defined and Output = A * B
    AOutput: Mul<B, Output = Output>,
    B: Clone,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.mul(self.1.clone()), df.mul(self.1.clone()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantDiv<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantDiv<A, B>
where
    A::Output: Div<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Div<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs> for ADConstantDiv<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    // ensure A / B is defined and Output = A * B
    AOutput: Div<B, Output = Output>,
    AGrad: Div<B, Output = Grad>,
    // ensure B is Clone
    B: Clone,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).div(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.div(self.1.clone()), df.div(self.1.clone()))
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantDiv<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A / B is defined and Output = A * B
    AOutput: Div<B, Output = Output>,
    B: Clone,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.div(self.1.clone()), df.div(self.1.clone()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADConstantPow<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantPow<A, B>
where
    A::Output: Pow<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Pow<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, ADB, A, B> AutoDiffable<StaticArgs> for ADConstantPow<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    // ensure A is Clone and A^B is defined and is Output
    AOutput: Clone + Pow<B, Output = Output>,
    // ensure B is Clone and B.one is defined and B-1 is B
    B: Clone + InstOne + Sub<B, Output = B>,
    // ensure A^(B-1) * B is defined and is ADB
    Output: Mul<B, Output = ADB>,
    // ensure dA * A^(B-1) * B is defined and is Grad
    AGrad: Mul<ADB, Output = Grad>,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).pow(self.1.clone())
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);

        // d(f^p) = p * f^(p-1) * df
        // = df * ((f^(p-1)) * p)

        (
            f.clone().pow(self.1.clone()),
            (df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone()))),
        )
    }
}

impl<StaticArgs, Input, Output, AOutput, APBB, A, B> ForwardDiffable<StaticArgs> for ADConstantPow<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A is Clone and A^B is defined and is Output
    AOutput: Clone + Pow<B, Output = Output>,
    // ensure B is Clone and B.one is defined and B-1 is B
    B: Clone + InstOne + Sub<B, Output = B>,
    // ensure A^(B-1) * B is defined and is APBB
    Output: Mul<B, Output = APBB>,
    // ensure A * A^(B-1) * B is defined and is Output
    AOutput: Mul<APBB, Output = Output>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        // d(f^p) = p * f^(p-1) * df
        // = df * ((f^(p-1)) * p)

        (
            f.clone().pow(self.1.clone()),
            df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone())),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADAbs<A>(pub A);

impl<A: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADAbs<A> {
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, Input, Output, Grad, A> AutoDiffable<StaticArgs> for ADAbs<A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: GradientType<Output, GradientType = Grad>,
    Output: Signed,
    Grad: Mul<Output, Output = Grad>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).abs()
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.abs(), df.mul(f.signum()))
    }
}

impl<StaticArgs, Input, Output, A> ForwardDiffable<StaticArgs> for ADAbs<A>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
    Output: Signed + Mul<Output, Output = Output>,
{
    //type Input = Input;
    //type Output = Output;
    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.abs(), df.mul(f.signum()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSignum<A>(pub A);

impl<A: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADSignum<A> {
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, Input, Output, Grad, A> AutoDiffable<StaticArgs> for ADSignum<A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: GradientType<Output, GradientType = Grad>,
    Output: Signed + InstZero,
    Grad: InstZero + UpperBounded,
{
    //type Input = Input;
    //type Output = Output;

    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).signum()
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        // chain rule on signum, (sign(f(x)))' = 2 delta(f(x))
        // we approximate delta(x) as
        // delta(x) = Grad::MAX if x == 0, 0 otherwise

        let (f, df) = self.0.eval_grad(x, static_args);

        if InstZero::is_zero(&f) {
            (f.signum(), Grad::max_value())
        } else {
            (f.signum(), df.zero())
        }
    }
}

impl<StaticArgs, Input, Output, A> ForwardDiffable<StaticArgs> for ADSignum<A>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
    Output: Signed + InstZero + UpperBounded,
{
    //type Input = Input;
    //type Output = Output;

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> (<Self as Diffable<StaticArgs>>::Output, <Self as Diffable<StaticArgs>>::Output) {
        // chain rule on signum, (sign(f(x)))' = 2 delta(f(x))
        // we approximate delta(x) as
        // delta(x) = Grad::MAX if x == 0, 0 otherwise

        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        if InstZero::is_zero(&f) {
            (f.signum(), Output::max_value())
        } else {
            (f.signum(), df.zero())
        }
    }
}
