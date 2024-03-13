use crate::autodiffable::{AutoDiffable, ForwardDiffable};
use crate::diffable::Diffable;
use crate::forward::ForwardMul;
use crate::gradienttype::GradientType;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_einsum_beta::einsum;
use crate::ad_ndarray::scalar::*;
use crate::ad_ndarray::traits::NDDot;

use crate as autodiff;
use autodiff_derive::*;

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADDot<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADDot<A, B>
where
    A::Output: Dot<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as Dot<B::Output>>::Output;
}

impl<StaticArgs, Input, AOutput, BOutput, AGrad, BGrad, AGradB, ABGrad, Output, Grad, A, B> AutoDiffable<StaticArgs> for ADDot<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    AOutput: NDDot<BOutput, Output = Output>,
    AGrad: NDDot<BOutput, Output = AGradB>,
    AOutput: NDDot<BGrad, Output = ABGrad>,
    AGradB: Add<ABGrad, Output = Grad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.eval(x, static_args).dot(&self.1.eval(x, static_args))
    }

    fn eval_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        (f.dot(&g), df.dot(&g).add(f.dot(&dg)))

    }
}

