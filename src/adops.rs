use crate::autodiffable::{AutoDiffable, ForwardDiffable};
use crate::diffable::Diffable;
use crate::forward::ForwardMul;
use crate::gradienttype::GradientType;
use crate::traits::{Abs, AbsSqr, Conjugate, InstOne, InstZero, PossiblyComplex, Signum}; //, Arg};
use num::traits::Pow;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate as autodiff;
use autodiff_derive::*;

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADCoerce<A, NewInput, NewOutput>(pub A, pub PhantomData<(NewInput, NewOutput)>);

impl<A: Diffable<StaticArgs>, NewInput, NewOutput, StaticArgs> Diffable<StaticArgs>
    for ADCoerce<A, NewInput, NewOutput>
{
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
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(&x.clone().into(), static_args).into()
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, NewGradient) {
        let (f, df) = self.0.eval_grad(&x.clone().into(), static_args);
        (f.into(), df.into())
    }

    fn grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> NewGradient {
        self.0.grad(&x.clone().into(), static_args).into()
    }

    // Wirtinger derivative df/dconj(x)
    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, NewGradient) {
        let (f, df) = self.0.eval_conj_grad(&x.clone().into(), static_args);
        (f.into(), df.into())
    }

    // Wirtinger derivative df/dconj(x)
    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> NewGradient {
        self.0.conj_grad(&x.clone().into(), static_args).into()
    }
}

// impl ForwardDiffable for ADCoerce<A, NewInput, NewOutput>
impl<StaticArgs, Input, Output, NewInput, NewOutput, A> ForwardDiffable<StaticArgs>
    for ADCoerce<A, NewInput, NewOutput>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
    NewInput: Clone,
    Input: From<NewInput>,
    NewOutput: From<Output>,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(&x.clone().into(), static_args).into()
    }

    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self
            .0
            .eval_forward_grad(&x.clone().into(), &dx.clone().into(), static_args);
        (f.into(), df.into())
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) =
            self.0
                .eval_forward_conj_grad(&x.clone().into(), &dx.clone().into(), static_args);
        (f.into(), df.into())
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_grad(&x.clone().into(), &dx.clone().into(), static_args)
            .into()
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_conj_grad(&x.clone().into(), &dx.clone().into(), static_args)
            .into()
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADAppendStaticArgs<A, NewStaticArgs>(pub A, pub PhantomData<NewStaticArgs>);

impl<A: Diffable<StaticArgs>, StaticArgs, NewStaticArgs> Diffable<(StaticArgs, NewStaticArgs)>
    for ADAppendStaticArgs<A, NewStaticArgs>
{
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, NewStaticArgs, Input, Output, Gradient, A>
    AutoDiffable<(StaticArgs, NewStaticArgs)> for ADAppendStaticArgs<A, NewStaticArgs>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: GradientType<Output, GradientType = Gradient>,
{
    fn eval(&self, x: &Self::Input, static_args: &(StaticArgs, NewStaticArgs)) -> Self::Output {
        self.0.eval(x, &static_args.0)
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        static_args: &(StaticArgs, NewStaticArgs),
    ) -> (Self::Output, Gradient) {
        self.0.eval_grad(x, &static_args.0)
    }

    fn grad(&self, x: &Self::Input, static_args: &(StaticArgs, NewStaticArgs)) -> Gradient {
        self.0.grad(x, &static_args.0)
    }

    fn eval_conj_grad(
        &self,
        x: &Self::Input,
        static_args: &(StaticArgs, NewStaticArgs),
    ) -> (Self::Output, Gradient) {
        self.0.eval_conj_grad(x, &static_args.0)
    }

    fn conj_grad(&self, x: &Self::Input, static_args: &(StaticArgs, NewStaticArgs)) -> Gradient {
        self.0.conj_grad(x, &static_args.0)
    }
}

impl<StaticArgs, NewStaticArgs, Input, Output, A> ForwardDiffable<(StaticArgs, NewStaticArgs)>
    for ADAppendStaticArgs<A, NewStaticArgs>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
{
    fn eval_forward(
        &self,
        x: &Self::Input,
        static_args: &(StaticArgs, NewStaticArgs),
    ) -> Self::Output {
        self.0.eval_forward(x, &static_args.0)
    }

    fn eval_forward_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &(StaticArgs, NewStaticArgs),
    ) -> (Self::Output, Self::Output) {
        self.0.eval_forward_grad(x, dx, &static_args.0)
    }
    fn eval_forward_conj_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &(StaticArgs, NewStaticArgs),
    ) -> (Self::Output, Self::Output) {
        self.0.eval_forward_conj_grad(x, dx, &static_args.0)
    }
    fn forward_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &(StaticArgs, NewStaticArgs),
    ) -> Self::Output {
        self.0.forward_grad(x, dx, &static_args.0)
    }
    fn forward_conj_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &(StaticArgs, NewStaticArgs),
    ) -> Self::Output {
        self.0.forward_conj_grad(x, dx, &static_args.0)
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADPrependStaticArgs<A, NewStaticArgs>(pub A, pub PhantomData<NewStaticArgs>);

impl<A: Diffable<StaticArgs>, StaticArgs, NewStaticArgs> Diffable<(NewStaticArgs, StaticArgs)>
    for ADPrependStaticArgs<A, NewStaticArgs>
{
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, NewStaticArgs, Input, Output, Gradient, A>
    AutoDiffable<(NewStaticArgs, StaticArgs)> for ADPrependStaticArgs<A, NewStaticArgs>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: GradientType<Output, GradientType = Gradient>,
{
    fn eval(&self, x: &Self::Input, static_args: &(NewStaticArgs, StaticArgs)) -> Self::Output {
        self.0.eval(x, &static_args.1)
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        static_args: &(NewStaticArgs, StaticArgs),
    ) -> (Self::Output, Gradient) {
        self.0.eval_grad(x, &static_args.1)
    }

    fn grad(&self, x: &Self::Input, static_args: &(NewStaticArgs, StaticArgs)) -> Gradient {
        self.0.grad(x, &static_args.1)
    }

    fn eval_conj_grad(
        &self,
        x: &Self::Input,
        static_args: &(NewStaticArgs, StaticArgs),
    ) -> (Self::Output, Gradient) {
        self.0.eval_conj_grad(x, &static_args.1)
    }

    fn conj_grad(&self, x: &Self::Input, static_args: &(NewStaticArgs, StaticArgs)) -> Gradient {
        self.0.conj_grad(x, &static_args.1)
    }
}

impl<StaticArgs, NewStaticArgs, Input, Output, A> ForwardDiffable<(NewStaticArgs, StaticArgs)>
    for ADPrependStaticArgs<A, NewStaticArgs>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
{
    fn eval_forward(
        &self,
        x: &Self::Input,
        static_args: &(NewStaticArgs, StaticArgs),
    ) -> Self::Output {
        self.0.eval_forward(x, &static_args.1)
    }

    fn eval_forward_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &(NewStaticArgs, StaticArgs),
    ) -> (Self::Output, Self::Output) {
        self.0.eval_forward_grad(x, dx, &static_args.1)
    }
    fn eval_forward_conj_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &(NewStaticArgs, StaticArgs),
    ) -> (Self::Output, Self::Output) {
        self.0.eval_forward_conj_grad(x, dx, &static_args.1)
    }
    fn forward_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &(NewStaticArgs, StaticArgs),
    ) -> Self::Output {
        self.0.forward_grad(x, dx, &static_args.1)
    }
    fn forward_conj_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &(NewStaticArgs, StaticArgs),
    ) -> Self::Output {
        self.0.forward_conj_grad(x, dx, &static_args.1)
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADAdd<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs>
    for ADAdd<A, B>
where
    A::Output: Add<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Add<B::Output>>::Output;
}

impl<StaticArgs, Input, AOutput, BOutput, AGrad, BGrad, Output, Grad, A, B> AutoDiffable<StaticArgs>
    for ADAdd<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    AOutput: Add<BOutput, Output = Output>,
    AGrad: Add<BGrad, Output = Grad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        self.0.grad(x, static_args).add(self.1.grad(x, static_args))
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (f.add(g), df.add(dg))
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        self.0
            .conj_grad(x, static_args)
            .add(self.1.conj_grad(x, static_args))
    }
}

impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> ForwardDiffable<StaticArgs> for ADAdd<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Add<BOutput, Output = Output>,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let f = self.0.eval_forward(x, static_args);
        let g = self.1.eval_forward(x, static_args);

        f.add(g)
    }

    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (f.add(g), df.add(dg))
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        (f.add(g), df.add(dg))
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_grad(x, dx, static_args)
            .add(self.1.forward_grad(x, dx, static_args))
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_conj_grad(x, dx, static_args)
            .add(self.1.forward_conj_grad(x, dx, static_args))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADSub<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs>
    for ADSub<A, B>
where
    A::Output: Sub<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Sub<B::Output>>::Output;
}

impl<StaticArgs, Input, AOutput, BOutput, AGrad, BGrad, Output, Grad, A, B> AutoDiffable<StaticArgs>
    for ADSub<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    AOutput: Sub<BOutput, Output = Output>,
    AGrad: Sub<BGrad, Output = Grad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        self.0.grad(x, static_args).sub(self.1.grad(x, static_args))
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (f.sub(g), df.sub(dg))
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        self.0
            .conj_grad(x, static_args)
            .sub(self.1.conj_grad(x, static_args))
    }
}

impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> ForwardDiffable<StaticArgs> for ADSub<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Sub<BOutput, Output = Output>,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let f = self.0.eval_forward(x, static_args);
        let g = self.1.eval_forward(x, static_args);

        f.sub(g)
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (f.sub(g), df.sub(dg))
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        (f.sub(g), df.sub(dg))
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_grad(x, dx, static_args)
            .sub(self.1.forward_grad(x, dx, static_args))
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_conj_grad(x, dx, static_args)
            .sub(self.1.forward_conj_grad(x, dx, static_args))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADMul<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs>
    for ADMul<A, B>
where
    A::Output: Mul<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Mul<B::Output>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, BOutput, AGrad, BGrad, DAB, ADB, A, B>
    AutoDiffable<StaticArgs> for ADMul<A, B>
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
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        let f = self.0.eval(x, static_args);
        let g = self.1.eval(x, static_args);
        let df = self.0.grad(x, static_args);
        let dg = self.1.grad(x, static_args);

        df.mul(g).add(f.mul(dg))
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (f.clone().mul(g.clone()), df.mul(g).add(f.mul(dg)))
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        let f = self.0.eval(x, static_args);
        let g = self.1.eval(x, static_args);
        let df = self.0.conj_grad(x, static_args);
        let dg = self.1.conj_grad(x, static_args);

        df.mul(g).add(f.mul(dg))
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
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let f = self.0.eval_forward(x, static_args);
        let g = self.1.eval_forward(x, static_args);

        f.mul(g)
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        // f * g : AOutput: Mul<BOutput, Output = Output>
        //
        // df * g : AGrad: Mul<BOutput, Output = DAB>
        // f * dg : BGrad: Mul<AOutput, Output = ADB>
        // df * g + f * dg : DAB: Add<ADB, Output = Grad>

        (f.clone().mul(g.clone()), df.mul(g).add(f.mul(dg)))
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        // f * g : AOutput: Mul<BOutput, Output = Output>
        //
        // df * g : AGrad: Mul<BOutput, Output = DAB>
        // f * dg : BGrad: Mul<AOutput, Output = ADB>
        // df * g + f * dg : DAB: Add<ADB, Output = Grad>

        (f.clone().mul(g.clone()), df.mul(g).add(f.mul(dg)))
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        df.mul(g).add(f.mul(dg))
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        df.mul(g).add(f.mul(dg))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADDiv<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs>
    for ADDiv<A, B>
where
    A::Output: Div<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Div<B::Output>>::Output;
}

impl<
        StaticArgs,
        Input,
        Output,
        Grad,
        AOutput,
        BOutput,
        AGrad,
        BGrad,
        BB,
        ADB,
        DAOVB,
        ADBOVBB,
        A,
        B,
    > AutoDiffable<StaticArgs> for ADDiv<A, B>
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
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g)))
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        // d(f/g) = (df*g - f*dg)/g^2 = df/g - f*dg/g^2
        // = (df/g - (f*dg)/(g*g))

        (
            f.clone().div(g.clone()),
            df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g))),
        )
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g)))
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
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let f = self.0.eval_forward(x, static_args);
        let g = self.1.eval_forward(x, static_args);

        f.div(g)
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        // d(f/g) = (df*g - f*dg)/g^2 = df/g - f*dg/g^2
        // = (df/g - (f*dg)/(g*g))

        (
            f.clone().div(g.clone()),
            df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g))),
        )
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        // d(f/g) = (df*g - f*dg)/g^2 = df/g - f*dg/g^2
        // = (df/g - (f*dg)/(g*g))

        (
            f.clone().div(g.clone()),
            df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g))),
        )
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g)))
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        df.div(g.clone()).sub(f.mul(dg).div(g.clone().mul(g)))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
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
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        self.0.grad(x, static_args).neg()
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        (f.neg(), df.neg())
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        self.0.conj_grad(x, static_args).neg()
    }
}

impl<StaticArgs, Input, Output, AOutput, A> ForwardDiffable<StaticArgs> for ADNeg<A>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A has Neg
    AOutput: Neg<Output = Output>,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let f = self.0.eval_forward(x, static_args);

        f.neg()
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.neg(), df.neg())
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.neg(), df.neg())
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.forward_grad(x, dx, static_args).neg()
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.forward_conj_grad(x, dx, static_args).neg()
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADCompose<Outer, Inner>(pub Outer, pub Inner);

impl<Outer, Inner, StaticArgs> Diffable<StaticArgs> for ADCompose<Outer, Inner>
where
    Outer: Diffable<StaticArgs>,
    Inner: Diffable<StaticArgs>,
{
    type Input = Inner::Input;
    type Output = Outer::Output;
}

impl<
        StaticArgs,
        InnerInput,
        InnerOutput,
        InnerGrad,
        OuterInput,
        OuterOutput,
        OuterGrad,
        Grad,
        Outer,
        Inner,
    > AutoDiffable<StaticArgs> for ADCompose<Outer, Inner>
where
    Outer: AutoDiffable<StaticArgs, Input = OuterInput, Output = OuterOutput>,
    Inner: AutoDiffable<StaticArgs, Input = InnerInput, Output = InnerOutput>,
    OuterInput: From<InnerOutput> + GradientType<OuterOutput, GradientType = OuterGrad>,
    InnerInput: GradientType<InnerOutput, GradientType = InnerGrad>
        + GradientType<OuterOutput, GradientType = Grad>,
    OuterGrad: ForwardMul<OuterInput, InnerGrad, ResultGrad = Grad>,
    InnerInput: PossiblyComplex,
    InnerOutput: Clone,
    OuterInput: PossiblyComplex,
    InnerGrad: Conjugate<Output = InnerGrad>,
    OuterGrad: Conjugate<Output = OuterGrad>,
    Grad: Add<Output = Grad>,
{
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .eval(&self.1.eval(x, static_args).into(), static_args)
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        if InnerInput::is_always_real() && OuterInput::is_always_real() {
            let (g, dg) = self.1.eval_grad(x, static_args);
            let (f, df) = self.0.eval_grad(&g.into(), static_args);
            (f, df.forward_mul(&dg))
        } else {
            // in the Wirtinger calculus we have
            //
            // d/dz (f(g(z))) = df/dz(g(z)) * dg/dz + df/dconjz(g(z)) * dconjg/dz
            // and dconjg/dz = conj(dg/dconjz)

            let (g, dg) = self.1.eval_grad(x, static_args);
            let dconjg = self.1.conj_grad(x, static_args).conj();
            let (f, df) = self.0.eval_grad(&g.clone().into(), static_args);
            let dfdconjg = self.0.conj_grad(&g.into(), static_args);

            (f, df.forward_mul(&dg).add(dfdconjg.forward_mul(&dconjg)))
        }
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        if InnerInput::is_always_real() && OuterInput::is_always_real() {
            let (g, dg) = self.1.eval_grad(x, static_args);
            let df = self.0.grad(&g.into(), static_args);
            df.forward_mul(&dg)
        } else {
            // in the Wirtinger calculus we have
            //
            // d/dz (f(g(z))) = df/dz(g(z)) * dg/dz + df/dconjz(g(z)) * dconjg/dz
            // and dconjg/dz = conj(dg/dconjz)

            let (g, dg) = self.1.eval_grad(x, static_args);
            let dconjg = self.1.conj_grad(x, static_args).conj();
            let df = self.0.grad(&g.clone().into(), static_args);
            let dfdconjg = self.0.conj_grad(&g.into(), static_args);

            df.forward_mul(&dg).add(dfdconjg.forward_mul(&dconjg))
        }
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        if InnerInput::is_always_real() && OuterInput::is_always_real() {
            self.eval_grad(x, static_args)
        } else {
            // in the Wirtinger calculus we have
            //
            // d/dconjz (f(g(z))) = df/dz(g(z)) * dg/dconjz + df/dconjz(g(z)) * dgconj/dconjz
            // and dgconj/dconjz = conj(dg/dz)

            let (g, dgdconjz) = self.1.eval_conj_grad(x, static_args);
            let dconjgdconjz = self.1.grad(x, static_args).conj();
            let (f, df) = self.0.eval_grad(&g.clone().into(), static_args);
            let dfdconjg = self.0.conj_grad(&g.into(), static_args);

            (
                f,
                df.forward_mul(&dgdconjz)
                    .add(dfdconjg.forward_mul(&dconjgdconjz)),
            )
        }
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        if InnerInput::is_always_real() && OuterInput::is_always_real() {
            self.grad(x, static_args)
        } else {
            // in the Wirtinger calculus we have
            //
            // d/dconjz (f(g(z))) = df/dz(g(z)) * dg/dconjz + df/dconjz(g(z)) * dgconj/dconjz
            // and dgconj/dconjz = conj(dg/dz)

            let (g, dgdconjz) = self.1.eval_conj_grad(x, static_args);
            let dconjgdconjz = self.1.grad(x, static_args).conj();
            let df = self.0.grad(&g.clone().into(), static_args);
            let dfdconjg = self.0.conj_grad(&g.into(), static_args);

            df.forward_mul(&dgdconjz)
                .add(dfdconjg.forward_mul(&dconjgdconjz))
        }
    }
}

impl<StaticArgs, InnerInput, InnerOutput, OuterInput, OuterOutput, Outer, Inner>
    ForwardDiffable<StaticArgs> for ADCompose<Outer, Inner>
where
    Outer: ForwardDiffable<StaticArgs, Input = OuterInput, Output = OuterOutput>,
    Inner: ForwardDiffable<StaticArgs, Input = InnerInput, Output = InnerOutput>,
    OuterInput: From<InnerOutput>,
    InnerInput: PossiblyComplex,
    InnerOutput: Clone,
    OuterInput: PossiblyComplex,
    OuterOutput: Add<OuterOutput, Output = OuterOutput>,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .eval_forward(&self.1.eval_forward(x, static_args).into(), static_args)
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        if InnerInput::is_always_real() && OuterInput::is_always_real() {
            let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);
            let (f, df) = self.0.eval_forward_grad(&g.into(), &dg.into(), static_args);
            (f, df)
        } else {
            // in the Wirtinger calculus we have
            //
            // d/dz (f(g(z))) = df/dz(g(z)) * dg/dz + df/dconjz(g(z)) * dconjg/dz
            // and dconjg/dz = conj(dg/dconjz)

            // g and dg/dz * dz
            let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);
            // conj(dg/dconj(z) * dconj(z)) = dconj(g)/dz * dz
            let dgdconjz = self.1.forward_conj_grad(x, dx, static_args);
            // f and df/dg * dg
            let (f, df) = self
                .0
                .eval_forward_grad(&g.clone().into(), &dg.into(), static_args);
            // df/dconjg * dconjg
            let dfdconjg = self
                .0
                .forward_conj_grad(&g.into(), &dgdconjz.into(), static_args);

            (f, df.add(dfdconjg))
        }
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        if InnerInput::is_always_real() && OuterInput::is_always_real() {
            let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);
            let df = self.0.forward_grad(&g.into(), &dg.into(), static_args);
            df
        } else {
            // in the Wirtinger calculus we have
            //
            // d/dz (f(g(z))) = df/dz(g(z)) * dg/dz + df/dconjz(g(z)) * dconjg/dz
            // and dconjg/dz = conj(dg/dconjz)

            // g and dg/dz * dz
            let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);
            // conj(dg/dconj(z) * dconj(z)) = dconj(g)/dz * dz
            let dgdconjz = self.1.forward_conj_grad(x, dx, static_args);
            // f and df/dg * dg
            let df = self
                .0
                .forward_grad(&g.clone().into(), &dg.into(), static_args);
            // df/dconjg * dconjg
            let dfdconjg = self
                .0
                .forward_conj_grad(&g.into(), &dgdconjz.into(), static_args);

            df.add(dfdconjg)
        }
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        if InnerInput::is_always_real() && OuterInput::is_always_real() {
            self.eval_forward_grad(x, dx, static_args)
        } else {
            // in the Wirtinger calculus we have
            //
            // d/dconjz (f(g(z))) = df/dz(g(z)) * dg/dconjz + df/dconjz(g(z)) * dgconj/dconjz
            // and dgconj/dconjz = conj(dg/dz)

            // g and dg/dz * dz
            let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);
            // conj(dg/dconj(z) * dconj(z)) = dconj(g)/dz * dz
            let dgdconjz = self.1.forward_grad(x, dx, static_args);
            // f and df/dg * dg
            let (f, df) = self
                .0
                .eval_forward_conj_grad(&g.clone().into(), &dg.into(), static_args);
            // df/dconjg * dconjg
            let dfdconjg = self
                .0
                .forward_conj_grad(&g.into(), &dgdconjz.into(), static_args);

            (f, df.add(dfdconjg))
        }
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        if InnerInput::is_always_real() && OuterInput::is_always_real() {
            self.forward_grad(x, dx, static_args)
        } else {
            // in the Wirtinger calculus we have
            //
            // d/dconjz (f(g(z))) = df/dz(g(z)) * dg/dconjz + df/dconjz(g(z)) * dgconj/dconjz
            // and dgconj/dconjz = conj(dg/dz)

            // g and dg/dz * dz
            let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);
            // conj(dg/dconj(z) * dconj(z)) = dconj(g)/dz * dz
            let dgdconjz = self.1.forward_grad(x, dx, static_args);
            // f and df/dg * dg
            let df = self
                .0
                .forward_conj_grad(&g.clone().into(), &dg.into(), static_args);
            // df/dconjg * dconjg
            let dfdconjg = self
                .0
                .forward_conj_grad(&g.into(), &dgdconjz.into(), static_args);

            df.add(dfdconjg)
        }
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantAdd<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantAdd<A, B>
where
    A::Output: Add<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Add<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs>
    for ADConstantAdd<A, B>
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
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        self.0.grad(x, static_args).add(self.1.zero())
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        (f.add(self.1.clone()), df.add(self.1.zero()))
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        self.0.conj_grad(x, static_args).add(self.1.zero())
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantAdd<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A + B is defined and Output = Output
    AOutput: Add<B, Output = Output>,
    B: Clone + InstZero,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).add(self.1.clone())
    }

    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.add(self.1.clone()), df.add(self.1.zero()))
    }
    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.add(self.1.clone()), df.add(self.1.zero()))
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.forward_grad(x, dx, static_args).add(self.1.zero())
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_conj_grad(x, dx, static_args)
            .add(self.1.zero())
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantSub<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantSub<A, B>
where
    A::Output: Sub<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Sub<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs>
    for ADConstantSub<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    // ensure A + B is defined and Output = Output
    AOutput: Sub<B, Output = Output>,
    AGrad: Sub<B, Output = Grad>,
    // ensure B is Clone and B.zero is defined
    B: Clone + InstZero,
    // assign gradient type
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        self.0.grad(x, static_args).sub(self.1.zero())
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        (f.sub(self.1.clone()), df.sub(self.1.zero()))
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        self.0.conj_grad(x, static_args).sub(self.1.zero())
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantSub<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A + B is defined and Output = Output
    AOutput: Sub<B, Output = Output>,
    B: Clone + InstZero,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).sub(self.1.clone())
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.sub(self.1.clone()), df.sub(self.1.zero()))
    }
    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.sub(self.1.clone()), df.sub(self.1.zero()))
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.forward_grad(x, dx, static_args).sub(self.1.zero())
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_conj_grad(x, dx, static_args)
            .sub(self.1.zero())
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantMul<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantMul<A, B>
where
    A::Output: Mul<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Mul<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs>
    for ADConstantMul<A, B>
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
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        self.0.grad(x, static_args).mul(self.1.clone())
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        (f.mul(self.1.clone()), df.mul(self.1.clone()))
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        self.0.conj_grad(x, static_args).mul(self.1.clone())
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantMul<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A * B is defined and Output = A * B
    AOutput: Mul<B, Output = Output>,
    B: Clone,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).mul(self.1.clone())
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.mul(self.1.clone()), df.mul(self.1.clone()))
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.forward_grad(x, dx, static_args).mul(self.1.clone())
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.mul(self.1.clone()), df.mul(self.1.clone()))
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_conj_grad(x, dx, static_args)
            .mul(self.1.clone())
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantDiv<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantDiv<A, B>
where
    A::Output: Div<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Div<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs>
    for ADConstantDiv<A, B>
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
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        self.0.grad(x, static_args).div(self.1.clone())
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        (f.div(self.1.clone()), df.div(self.1.clone()))
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        self.0.conj_grad(x, static_args).div(self.1.clone())
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantDiv<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A / B is defined and Output = A * B
    AOutput: Div<B, Output = Output>,
    B: Clone,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).div(self.1.clone())
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.div(self.1.clone()), df.div(self.1.clone()))
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.forward_grad(x, dx, static_args).div(self.1.clone())
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.div(self.1.clone()), df.div(self.1.clone()))
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0
            .forward_conj_grad(x, dx, static_args)
            .div(self.1.clone())
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantPow<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantPow<A, B>
where
    A::Output: Pow<B>,
    B: Clone,
{
    type Input = A::Input;
    type Output = <A::Output as Pow<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, ADB, A, B> AutoDiffable<StaticArgs>
    for ADConstantPow<A, B>
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
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
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
        //
        // this is true for Wirtinger calculus as well, since the outer function in the chain rule
        // is z^p, which has conjugate derivative 0

        (
            f.clone().pow(self.1.clone()),
            (df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone()))),
        )
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        let df = self.0.grad(x, static_args);

        df.mul(
            self.0
                .eval(x, static_args)
                .pow(self.1.clone().sub(self.1.one()))
                .mul(self.1.clone()),
        )
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        // d(f^p) = p * f^(p-1) * df

        (
            f.clone().pow(self.1.clone()),
            (df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone()))),
        )
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        let df = self.0.conj_grad(x, static_args);

        df.mul(
            self.0
                .eval(x, static_args)
                .pow(self.1.clone().sub(self.1.one()))
                .mul(self.1.clone()),
        )
    }
}

impl<StaticArgs, Input, Output, AOutput, APBB, A, B> ForwardDiffable<StaticArgs>
    for ADConstantPow<A, B>
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
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).pow(self.1.clone())
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        // d(f^p) = p * f^(p-1) * df
        // = df * ((f^(p-1)) * p)

        (
            f.clone().pow(self.1.clone()),
            df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone())),
        )
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone()))
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        // d(f^p) = p * f^(p-1) * df

        (
            f.clone().pow(self.1.clone()),
            df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone())),
        )
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone()))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADAbs<A>(pub A);

impl<A: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADAbs<A> {
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, Input, Output, Grad, A> AutoDiffable<StaticArgs> for ADAbs<A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: PossiblyComplex + GradientType<Output, GradientType = Grad>,
    Output: Clone
        + PossiblyComplex
        + Abs<Output = Output>
        + Signum<Output = Output>
        + Conjugate<Output = Output>,
    Grad: Conjugate<Output = Grad> + Mul<Output, Output = Grad> + Add<Grad, Output = Grad>,
{
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).abs()
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_grad(x, static_args);

            (f.clone().abs(), df.mul(f.signum()))
        } else {
            // in the Wirtinger calculus we have
            // |z| = sqrt(z * conj(z))
            // d|z|/dz = 1/2 * conj(z) * (z * conj(z))^(-1/2)
            //         = 1/2 * conj(z) * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            // d|z|/dconjz = 1/2 * z * (z * conj(z))^(-1/2)
            //            = 1/2 * z * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            //
            // now for |f|, we have
            //
            // d|f|/dz = d|f|/df * df/dz + d|f|/dconjf * conj(df/dconjz)
            //        = f/|f| * df/dz + conj(f)/|f| * conj(df/dconjz)
            //
            // and
            //
            // d|f|/dconjz = d|f|/df * df/dconjz + d|f|/dconjf * conj(df/dz)
            //          = f/|f| * df/dconjz + conj(f)/|f| * conj(df/dz)

            // note that for purely real z, this reduces to the real case

            let (f, df) = self.0.eval_grad(x, static_args);
            let fconj = f.conj();
            let dconjfdz = self.0.conj_grad(x, static_args).conj();

            (
                f.clone().abs(),
                df.mul(f.signum()).add(dconjfdz.mul(fconj.signum())),
            )
        }
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        if Input::is_always_real() && Output::is_always_real() {
            let df = self.0.grad(x, static_args);

            df.mul(self.0.eval(x, static_args).signum())
        } else {
            // in the Wirtinger calculus we have
            // |z| = sqrt(z * conj(z))
            // d|z|/dz = 1/2 * conj(z) * (z * conj(z))^(-1/2)
            //         = 1/2 * conj(z) * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            // d|z|/dconjz = 1/2 * z * (z * conj(z))^(-1/2)
            //            = 1/2 * z * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            //
            // now for |f|, we have
            //
            // d|f|/dz = d|f|/df * df/dz + d|f|/dconjf * conj(df/dconjz)
            //        = f/|f| * df/dz + conj(f)/|f| * conj(df/dconjz)
            //
            // and
            //
            // d|f|/dconjz = d|f|/df * df/dconjz + d|f|/dconjf * conj(df/dz)
            //          = f/|f| * df/dconjz + conj(f)/|f| * conj(df/dz)

            // note that for purely real z, this reduces to the real case

            let (f, df) = self.0.eval_grad(x, static_args);
            let fconj = f.conj();
            let dconjfdz = self.0.conj_grad(x, static_args).conj();

            df.mul(f.signum()).add(dconjfdz.mul(fconj.signum()))
        }
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_conj_grad(x, static_args);

            (f.clone().abs(), df.mul(f.signum()))
        } else {
            // in the Wirtinger calculus we have
            // |z| = sqrt(z * conj(z))
            // d|z|/dz = 1/2 * conj(z) * (z * conj(z))^(-1/2)
            //         = 1/2 * conj(z) * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            // d|z|/dconjz = 1/2 * z * (z * conj(z))^(-1/2)
            //            = 1/2 * z * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            //
            // now for |f|, we have
            //
            // d|f|/dz = d|f|/df * df/dz + d|f|/dconjf * conj(df/dconjz)
            //        = f/|f| * df/dz + conj(f)/|f| * conj(df/dconjz)
            //
            // and
            //
            // d|f|/dconjz = d|f|/df * df/dconjz + d|f|/dconjf * conj(df/dz)
            //          = f/|f| * df/dconjz + conj(f)/|f| * conj(df/dz)

            // note that for purely real z, this reduces to the real case

            let (f, dfdconjz) = self.0.eval_conj_grad(x, static_args);
            let fconj = f.conj();
            let dconjfdconjz = self.0.grad(x, static_args).conj();

            (
                f.clone().abs(),
                dfdconjz
                    .mul(f.signum())
                    .add(dconjfdconjz.mul(fconj.signum())),
            )
        }
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        if Input::is_always_real() && Output::is_always_real() {
            let df = self.0.conj_grad(x, static_args);

            df.mul(self.0.eval(x, static_args).signum())
        } else {
            // in the Wirtinger calculus we have
            // |z| = sqrt(z * conj(z))
            // d|z|/dz = 1/2 * conj(z) * (z * conj(z))^(-1/2)
            //         = 1/2 * conj(z) * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            // d|z|/dconjz = 1/2 * z * (z * conj(z))^(-1/2)
            //            = 1/2 * z * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            //
            // now for |f|, we have
            //
            // d|f|/dz = d|f|/df * df/dz + d|f|/dconjf * conj(df/dconjz)
            //        = f/|f| * df/dz + conj(f)/|f| * conj(df/dconjz)
            //
            // and
            //
            // d|f|/dconjz = d|f|/df * df/dconjz + d|f|/dconjf * conj(df/dz)
            //          = f/|f| * df/dconjz + conj(f)/|f| * conj(df/dz)

            // note that for purely real z, this reduces to the real case

            let (f, dfdconjz) = self.0.eval_conj_grad(x, static_args);
            let fconj = f.conj();
            let dconjfdconjz = self.0.grad(x, static_args).conj();

            dfdconjz
                .mul(f.signum())
                .add(dconjfdconjz.mul(fconj.signum()))
        }
    }
}

impl<StaticArgs, Input, Output, A> ForwardDiffable<StaticArgs> for ADAbs<A>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: PossiblyComplex,
    Output: Clone
        + PossiblyComplex
        + Signum<Output = Output>
        + Abs<Output = Output>
        + Conjugate<Output = Output>
        + Mul<Output, Output = Output>
        + Add<Output, Output = Output>,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).abs()
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

            (f.clone().abs(), df.mul(f.signum()))
        } else {
            // in the Wirtinger calculus we have
            // |z| = sqrt(z * conj(z))
            // d|z|/dz = 1/2 * conj(z) * (z * conj(z))^(-1/2)
            //         = 1/2 * conj(z) * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            // d|z|/dconjz = 1/2 * z * (z * conj(z))^(-1/2)
            //            = 1/2 * z * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            //
            // now for |f|, we have
            //
            // d|f|/dz = d|f|/df * df/dz + d|f|/dconjf * conj(df/dconjz)
            //        = f/|f| * df/dz + conj(f)/|f| * conj(df/dconjz)
            //
            // and
            //
            // d|f|/dconjz = d|f|/df * df/dconjz + d|f|/dconjf * conj(df/dz)
            //          = f/|f| * df/dconjz + conj(f)/|f| * conj(df/dz)

            // note that for purely real z, this reduces to the real case

            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
            let fconj = f.conj();
            let dconjfdz = self.0.forward_conj_grad(x, dx, static_args).conj();

            (
                f.clone().abs(),
                df.mul(f.signum()).add(dconjfdz.mul(fconj.signum())),
            )
        }
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

            df.mul(f.signum())
        } else {
            // in the Wirtinger calculus we have
            // |z| = sqrt(z * conj(z))
            // d|z|/dz = 1/2 * conj(z) * (z * conj(z))^(-1/2)
            //         = 1/2 * conj(z) * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            // d|z|/dconjz = 1/2 * z * (z * conj(z))^(-1/2)
            //            = 1/2 * z * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            //
            // now for |f|, we have
            //
            // d|f|/dz = d|f|/df * df/dz + d|f|/dconjf * conj(df/dconjz)
            //        = f/|f| * df/dz + conj(f)/|f| * conj(df/dconjz)
            //
            // and
            //
            // d|f|/dconjz = d|f|/df * df/dconjz + d|f|/dconjf * conj(df/dz)
            //          = f/|f| * df/dconjz + conj(f)/|f| * conj(df/dz)

            // note that for purely real z, this reduces to the real case

            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
            let fconj = f.conj();
            let dconjfdz = self.0.forward_conj_grad(x, dx, static_args).conj();

            df.mul(f.signum()).add(dconjfdz.mul(fconj.signum()))
        }
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

            (f.clone().abs(), df.mul(f.signum()))
        } else {
            // in the Wirtinger calculus we have
            // |z| = sqrt(z * conj(z))
            // d|z|/dz = 1/2 * conj(z) * (z * conj(z))^(-1/2)
            //         = 1/2 * conj(z) * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            // d|z|/dconjz = 1/2 * z * (z * conj(z))^(-1/2)
            //            = 1/2 * z * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            //
            // now for |f|, we have
            //
            // d|f|/dz = d|f|/df * df/dz + d|f|/dconjf * conj(df/dconjz)
            //        = f/|f| * df/dz + conj(f)/|f| * conj(df/dconjz)
            //
            // and
            //
            // d|f|/dconjz = d|f|/df * df/dconjz + d|f|/dconjf * conj(df/dz)
            //          = f/|f| * df/dconjz + conj(f)/|f| * conj(df/dz)

            // note that for purely real z, this reduces to the real case

            let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
            let fconj = f.conj();
            let dconjfdconjz = self.0.forward_grad(x, dx, static_args).conj();

            (
                f.clone().abs(),
                df.mul(f.signum()).add(dconjfdconjz.mul(fconj.signum())),
            )
        }
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

            df.mul(f.signum())
        } else {
            // in the Wirtinger calculus we have
            // |z| = sqrt(z * conj(z))
            // d|z|/dz = 1/2 * conj(z) * (z * conj(z))^(-1/2)
            //         = 1/2 * conj(z) * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            // d|z|/dconjz = 1/2 * z * (z * conj(z))^(-1/2)
            //            = 1/2 * z * |z|^(-1)
            // cannot simplify further, since we would be assuming the "sign" of z
            //
            // now for |f|, we have
            //
            // d|f|/dz = d|f|/df * df/dz + d|f|/dconjf * conj(df/dconjz)
            //        = f/|f| * df/dz + conj(f)/|f| * conj(df/dconjz)
            //
            // and
            //
            // d|f|/dconjz = d|f|/df * df/dconjz + d|f|/dconjf * conj(df/dz)
            //          = f/|f| * df/dconjz + conj(f)/|f| * conj(df/dz)

            // note that for purely real z, this reduces to the real case

            let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
            let fconj = f.conj();
            let dconjfdconjz = self.0.forward_grad(x, dx, static_args).conj();

            df.mul(f.signum()).add(dconjfdconjz.mul(fconj.signum()))
        }
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADAbsSqr<A>(pub A);

impl<A: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADAbsSqr<A> {
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, Input, Output, Grad, A> AutoDiffable<StaticArgs> for ADAbsSqr<A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: PossiblyComplex + GradientType<Output, GradientType = Grad>,
    Output: PossiblyComplex
        + AbsSqr<Output = Output>
        + Conjugate<Output = Output>
        + Add<Output, Output = Output>
        + Clone
        + Mul<Grad, Output = Grad>,
    Grad: Conjugate<Output = Grad> + Mul<Output, Output = Grad> + Add<Grad, Output = Grad>,
{
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).abs_sqr()
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        if Input::is_always_real() && Output::is_always_real() {
            // for real z, |z|^2 -> 2 |z| * sign(z) = 2z
            // |f|^2 -> 2f * df/dz

            let (f, df) = self.0.eval_grad(x, static_args);

            (f.clone().abs_sqr(), df.mul(f.clone().add(f)))
        } else {
            // in the Wirtinger calculus we have
            // |z|^2 = z * conj(z)
            // d|z|^2/dz = conj(z)
            //
            // now for |f|^2, we have
            //
            // d|f|^2/dz = d|f|^2/df * df/dz + d|f|^2/dconjf * conj(df/dconjz)
            //           = conj(f) * df/dz + f * conj(df/dconjz)

            let (f, df) = self.0.eval_grad(x, static_args);
            let fconj = f.conj();
            let dconjfdz = self.0.conj_grad(x, static_args).conj();

            (f.clone().abs_sqr(), df.mul(fconj).add(f.mul(dconjfdz)))
        }
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        if Input::is_always_real() && Output::is_always_real() {
            // for real z, |z|^2 -> 2 |z| * sign(z) = 2z
            // |f|^2 -> 2f * df/dz

            let (f, df) = self.0.eval_grad(x, static_args);

            df.mul(f.clone().add(f))
        } else {
            // in the Wirtinger calculus we have
            // |z|^2 = z * conj(z)
            // d|z|^2/dz = conj(z)
            //
            // now for |f|^2, we have
            //
            // d|f|^2/dz = d|f|^2/df * df/dz + d|f|^2/dconjf * conj(df/dconjz)
            //           = conj(f) * df/dz + f * conj(df/dconjz)

            let (f, df) = self.0.eval_grad(x, static_args);
            let fconj = f.conj();
            let dconjfdz = self.0.conj_grad(x, static_args).conj();

            df.mul(fconj).add(f.mul(dconjfdz))
        }
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        if Input::is_always_real() && Output::is_always_real() {
            // for real z, |z|^2 -> 2 |z| * sign(z) = 2z
            // |f|^2 -> 2f * df/dz

            let (f, df) = self.0.eval_conj_grad(x, static_args);

            (f.clone().abs_sqr(), df.mul(f.clone().add(f)))
        } else {
            // in the Wirtinger calculus we have
            // |z|^2 = z * conj(z)
            // d|z|^2/dconjz = z
            //
            // now for |f|^2, we have
            //
            // d|f|^2/dconjz = d|f|^2/df * df/dconjz + d|f|^2/dconjf * conj(df/dz)
            //           = conj(f) * df/dconjz + f * conj(df/dz)

            let (f, dfdconjz) = self.0.eval_conj_grad(x, static_args);
            let fconj = f.conj();
            let dconjfdconjz = self.0.grad(x, static_args).conj();

            (
                f.clone().abs_sqr(),
                dfdconjz.mul(fconj).add(f.mul(dconjfdconjz)),
            )
        }
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        if Input::is_always_real() && Output::is_always_real() {
            // for real z, |z|^2 -> 2 |z| * sign(z) = 2z
            // |f|^2 -> 2f * df/dz

            let (f, df) = self.0.eval_conj_grad(x, static_args);

            df.mul(f.clone().add(f))
        } else {
            // in the Wirtinger calculus we have
            // |z|^2 = z * conj(z)
            // d|z|^2/dconjz = z
            //
            // now for |f|^2, we have
            //
            // d|f|^2/dconjz = d|f|^2/df * df/dconjz + d|f|^2/dconjf * conj(df/dz)
            //           = conj(f) * df/dconjz + f * conj(df/dz)

            let (f, dfdconjz) = self.0.eval_conj_grad(x, static_args);
            let fconj = f.conj();
            let dconjfdconjz = self.0.grad(x, static_args).conj();

            dfdconjz.mul(fconj).add(f.mul(dconjfdconjz))
        }
    }
}

impl<StaticArgs, Input, Output, A> ForwardDiffable<StaticArgs> for ADAbsSqr<A>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: PossiblyComplex,
    Output: PossiblyComplex
        + AbsSqr<Output = Output>
        + Conjugate<Output = Output>
        + Mul<Output, Output = Output>
        + Add<Output, Output = Output>
        + Clone,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).abs_sqr()
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

            (f.clone().abs_sqr(), df.mul(f.clone().add(f)))
        } else {
            // in the Wirtinger calculus we have
            // |z|^2 = z * conj(z)
            // d|z|^2/dz = conj(z)
            //
            // now for |f|^2, we have
            //
            // d|f|^2/dz = d|f|^2/df * df/dz + d|f|^2/dconjf * conj(df/dconjz)
            //           = conj(f) * df/dz + f * conj(df/dconjz)

            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
            let fconj = f.conj();
            let dconjf = self.0.forward_conj_grad(x, dx, static_args).conj();

            (f.clone().abs_sqr(), df.mul(fconj).add(f.mul(dconjf)))
        }
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

            df.mul(f.clone().add(f))
        } else {
            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
            let fconj = f.conj();
            let dconjf = self.0.forward_conj_grad(x, dx, static_args).conj();

            df.mul(fconj).add(f.mul(dconjf))
        }
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

            (f.clone().abs_sqr(), df.mul(f.clone().add(f)))
        } else {
            // in the Wirtinger calculus we have
            // |z|^2 = z * conj(z)
            // d|z|^2/dconjz = z
            //
            // now for |f|^2, we have
            //
            // d|f|^2/dconjz = d|f|^2/df * df/dconjz + d|f|^2/dconjf * conj(df/dz)
            //           = conj(f) * df/dconjz + f * conj(df/dz)

            let (f, dfdconjz) = self.0.eval_forward_conj_grad(x, dx, static_args);
            let fconj = f.conj();
            let dconjfdconjz = self.0.forward_grad(x, dx, static_args).conj();

            (
                f.clone().abs_sqr(),
                dfdconjz.mul(fconj).add(f.mul(dconjfdconjz)),
            )
        }
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

            df.mul(f.clone().add(f))
        } else {
            let (f, dfdconjz) = self.0.eval_forward_conj_grad(x, dx, static_args);
            let fconj = f.conj();
            let dconjfdconjz = self.0.forward_grad(x, dx, static_args).conj();

            dfdconjz.mul(fconj).add(f.mul(dconjfdconjz))
        }
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADSignum<A>(pub A);

impl<A: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADSignum<A> {
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, Input, Output, Grad, A> AutoDiffable<StaticArgs> for ADSignum<A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: PossiblyComplex + GradientType<Output, GradientType = Grad>,
    Output: Clone
        + PossiblyComplex
        + Signum<Output = Output>
        + Abs<Output = Output>
        + Conjugate<Output = Output>
        + InstOne
        + InstZero
        + Mul<Grad, Output = Grad>
        + Mul<Output, Output = Output>
        + Add<Output, Output = Output>
        + Neg<Output = Output>
        + Div<Output, Output = Output>,
    Grad: Conjugate<Output = Grad> + InstZero,
{
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).signum()
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        // defining signum(f) = f/|f| if f != 0, 0 otherwise
        // this works for both purely real and complex f
        //
        // in the real case:
        // d/dx (f/|f|) = (|f| * df/dx - f * d|f|/dx) / |f|^2
        //
        // and
        //
        // d/dx |f| = f * df/dx / |f|
        //
        // so d/dx signum(f) = (|f| * df/dx - f^2 * df/dx / |f|) / |f|^2
        //                   = (|f| * df/dx - |f| * df/dx) / |f|^2
        //                   = 0
        // as expected
        //
        // in the complex case:
        //
        // d/dz (f/|f|)
        //
        // d/dz (z/|z|) = d/dz sqrt(z / conj(z)) = 1/(2 sqrt(z * conj(z)))
        //            = 1/(2 |z|)
        // d/dconjz (z/|z|) = d/dconjz sqrt(z / conj(z))
        //                 = sqrt(z) * d/dconjz conj(z)^(-1/2)
        //                 = sqrt(z) * -1/2 * conj(z)^(-3/2)
        //                 = -1/2 (z/conj(z)^3)^(1/2)
        //                 = -1/2 (z^4/|z|^6)^(1/2)
        //                 = -1/2 (z^2/|z|^3)
        //
        // so from the chain rule then
        //
        // d/dz (f/|f|) = (1/(2 |f|)) * df/dz - 1/2 (f^2/|f|^3) * conj(df/dconjz)
        // which would be 0 in the real case as expected

        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_grad(x, static_args);

            (f.signum(), df.zero())
        } else {
            let (f, df) = self.0.eval_grad(x, static_args);
            let dconjfdz = self.0.conj_grad(x, static_args).conj();
            let fabs = f.clone().abs();
            let dsdf = f.one().div(fabs.clone().add(fabs.clone()));
            let dsdconjf_half_denom = fabs.clone().mul(fabs.clone().mul(fabs));
            let dsdconjf = f
                .clone()
                .mul(f.clone())
                .div(dsdconjf_half_denom.clone().add(dsdconjf_half_denom))
                .neg();

            (f.signum(), dsdf.mul(df).add(dsdconjf.mul(dconjfdz)))
        }
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> Grad {
        if Input::is_always_real() && Output::is_always_real() {
            self.0.grad(x, static_args).zero()
        } else {
            let (f, df) = self.0.eval_grad(x, static_args);
            let dconjfdz = self.0.conj_grad(x, static_args).conj();
            let fabs = f.clone().abs();
            let dsdf = f.one().div(fabs.clone().add(fabs.clone()));
            let dsdconjf_half_denom = fabs.clone().mul(fabs.clone().mul(fabs));
            let dsdconjf = f
                .clone()
                .mul(f.clone())
                .div(dsdconjf_half_denom.clone().add(dsdconjf_half_denom))
                .neg();

            dsdf.mul(df).add(dsdconjf.mul(dconjfdz))
        }
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, Grad) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_conj_grad(x, static_args);

            (f.signum(), df.zero())
        } else {
            // now we use df/dconjz and conj(df/dz)

            let (f, dfdconjz) = self.0.eval_conj_grad(x, static_args);
            let dconjfdconjz = self.0.grad(x, static_args).conj();
            let fabs = f.clone().abs();
            let dsdf = f.one().div(fabs.clone().add(fabs.clone()));
            let dsdconjf_half_denom = fabs.clone().mul(fabs.clone().mul(fabs));
            let dsdconjf = f
                .clone()
                .mul(f.clone())
                .div(dsdconjf_half_denom.clone().add(dsdconjf_half_denom))
                .neg();

            (
                f.signum(),
                dsdf.mul(dfdconjz).add(dsdconjf.mul(dconjfdconjz)),
            )
        }
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> Grad {
        if Input::is_always_real() && Output::is_always_real() {
            self.0.conj_grad(x, static_args).conj().zero()
        } else {
            let (f, dfdconjz) = self.0.eval_conj_grad(x, static_args);
            let dconjfdconjz = self.0.grad(x, static_args).conj();
            let fabs = f.clone().abs();
            let dsdf = f.one().div(fabs.clone().add(fabs.clone()));
            let dsdconjf_half_denom = fabs.clone().mul(fabs.clone().mul(fabs));
            let dsdconjf = f
                .clone()
                .mul(f.clone())
                .div(dsdconjf_half_denom.clone().add(dsdconjf_half_denom))
                .neg();

            dsdf.mul(dfdconjz).add(dsdconjf.mul(dconjfdconjz))
        }
    }
}

impl<StaticArgs, Input, Output, A> ForwardDiffable<StaticArgs> for ADSignum<A>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: PossiblyComplex,
    Output: Clone
        + PossiblyComplex
        + Signum<Output = Output>
        + Abs<Output = Output>
        + Conjugate<Output = Output>
        + InstOne
        + InstZero
        + Mul<Output, Output = Output>
        + Add<Output, Output = Output>
        + Neg<Output = Output>
        + Div<Output, Output = Output>,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).signum()
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

            (f.signum(), df.zero())
        } else {
            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
            let dconjf = self.0.forward_conj_grad(x, dx, static_args);
            let fabs = f.clone().abs();
            let dsdf = f.one().div(fabs.clone().add(fabs.clone()));
            let dsdconjf_half_denom = fabs.clone().mul(fabs.clone().mul(fabs));
            let dsdconjf = f
                .clone()
                .mul(f.clone())
                .div(dsdconjf_half_denom.clone().add(dsdconjf_half_denom))
                .neg();

            (f.signum(), dsdf.mul(df).add(dsdconjf.mul(dconjf)))
        }
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        if Input::is_always_real() && Output::is_always_real() {
            self.0.forward_grad(x, dx, static_args).zero()
        } else {
            let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
            let dconjf = self.0.forward_conj_grad(x, dx, static_args);
            let fabs = f.clone().abs();
            let dsdf = f.one().div(fabs.clone().add(fabs.clone()));
            let dsdconjf_half_denom = fabs.clone().mul(fabs.clone().mul(fabs));
            let dsdconjf = f
                .clone()
                .mul(f.clone())
                .div(dsdconjf_half_denom.clone().add(dsdconjf_half_denom))
                .neg();

            dsdf.mul(df).add(dsdconjf.mul(dconjf))
        }
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        if Input::is_always_real() && Output::is_always_real() {
            let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

            (f.signum(), df.zero())
        } else {
            let (f, dfdconjz) = self.0.eval_forward_conj_grad(x, dx, static_args);
            let dconjfdconjz = self.0.forward_grad(x, dx, static_args).conj();
            let fabs = f.clone().abs();
            let dsdf = f.one().div(fabs.clone().add(fabs.clone()));
            let dsdconjf_half_denom = fabs.clone().mul(fabs.clone().mul(fabs));
            let dsdconjf = f
                .clone()
                .mul(f.clone())
                .div(dsdconjf_half_denom.clone().add(dsdconjf_half_denom))
                .neg();

            (
                f.signum(),
                dsdf.mul(dfdconjz).add(dsdconjf.mul(dconjfdconjz)),
            )
        }
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        if Input::is_always_real() && Output::is_always_real() {
            self.0.forward_conj_grad(x, dx, static_args).zero()
        } else {
            let (f, dfdconjz) = self.0.eval_forward_conj_grad(x, dx, static_args);
            let dconjfdconjz = self.0.forward_grad(x, dx, static_args).conj();
            let fabs = f.clone().abs();
            let dsdf = f.one().div(fabs.clone().add(fabs.clone()));
            let dsdconjf_half_denom = fabs.clone().mul(fabs.clone().mul(fabs));
            let dsdconjf = f
                .clone()
                .mul(f.clone())
                .div(dsdconjf_half_denom.clone().add(dsdconjf_half_denom))
                .neg();

            dsdf.mul(dfdconjz).add(dsdconjf.mul(dconjfdconjz))
        }
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConjugate<A>(pub A);

impl<A: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADConjugate<A>
where
    A::Output: Conjugate<Output = A::Output>,
{
    type Input = A::Input;
    type Output = A::Output;
}

impl<StaticArgs, Input, AOutput, AGrad, A> AutoDiffable<StaticArgs> for ADConjugate<A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    // make sure A.conj() is defined and Output = Output
    AOutput: Conjugate<Output = AOutput>,
    // make sure dA.conj() is defined and Output = Grad
    AGrad: Conjugate<Output = AGrad>,
{
    fn eval(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval(x, static_args).conj()
    }

    fn eval_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, AGrad) {
        // Wirtinger derivative dconj(f)/dz = conj(df/dconjz)

        let (f, dfdzconj) = self.0.eval_conj_grad(x, static_args);

        (f.conj(), dfdzconj.conj())
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, static_args: &StaticArgs) -> AGrad {
        self.0.conj_grad(x, static_args).conj()
    }

    fn eval_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (<Self as Diffable<StaticArgs>>::Output, AGrad) {
        // Wirtinger derivative dconj(f)/dconj(z) = conj(df/dz)

        let (f, df) = self.0.eval_grad(x, static_args);

        (f.conj(), df.conj())
    }

    fn conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> AGrad {
        self.0.grad(x, static_args).conj()
    }
}

impl<StaticArgs, Input, AOutput, A> ForwardDiffable<StaticArgs> for ADConjugate<A>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    AOutput: Conjugate<Output = AOutput>,
{
    fn eval_forward(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        self.0.eval_forward(x, static_args).conj()
    }
    fn eval_forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        // Wirtinger derivative conj(df/dz * dz) = dconj(f)/dconj(z) * dconj(z)

        let (f, dfconj) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.conj(), dfconj.conj())
    }

    fn forward_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        // Wirtinger derivative conj(df/dz * dz) = dconj(f)/dconj(z) * dconj(z)

        self.0.forward_conj_grad(x, dx, static_args).conj()
    }

    fn eval_forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> (
        <Self as Diffable<StaticArgs>>::Output,
        <Self as Diffable<StaticArgs>>::Output,
    ) {
        // Wirtinger derivative conj(df/dz * dz) = dconj(f)/dconj(z) * dconj(z)

        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.conj(), df.conj())
    }

    fn forward_conj_grad(
        &self,
        x: &<Self as Diffable<StaticArgs>>::Input,
        dx: &<Self as Diffable<StaticArgs>>::Input,
        static_args: &StaticArgs,
    ) -> <Self as Diffable<StaticArgs>>::Output {
        // Wirtinger derivative conj(df/dz * dz) = dconj(f)/dconj(z) * dconj(z)

        self.0.forward_grad(x, dx, static_args).conj()
    }
}
