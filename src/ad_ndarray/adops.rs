use crate::autodiffable::{AutoDiffable, ForwardDiffable};
use crate::diffable::Diffable;
use crate::gradienttype::GradientType;
use std::ops::Add;
use ndarray::linalg::Dot;
use crate::ad_ndarray::traits::{TensorDot, TensorContraction};

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
    AOutput: Dot<BOutput, Output = Output>,
    AGrad: Dot<BOutput, Output = AGradB>,
    AOutput: Dot<BGrad, Output = ABGrad>,
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

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (f.dot(&g), df.dot(&g).add(f.dot(&dg)))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        df.dot(&g).add(f.dot(&dg))
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        df.dot(&g).add(f.dot(&dg))
    }
}

impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> ForwardDiffable<StaticArgs> for ADDot<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Dot<BOutput, Output = Output>,
    Output: Add<Output, Output = Output>,
{

    fn eval_forward(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let f = self.0.eval_forward(x, static_args);
        let g = self.1.eval_forward(x, static_args);

        f.dot(&g)
    }

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (f.dot(&g), df.dot(&g).add(f.dot(&dg)))
    }

    fn eval_forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                              static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        (f.dot(&g), df.dot(&g).add(f.dot(&dg)))
    }

    fn forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        df.dot(&g).add(f.dot(&dg))
    }

    fn forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        df.dot(&g).add(f.dot(&dg))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADTensorDot<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADTensorDot<A, B>
where
    A::Output: TensorDot<B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as TensorDot<B::Output>>::Output;
}

impl<StaticArgs, Input, AOutput, BOutput, AGrad, BGrad, AGradB, ABGrad, Output, Grad, A, B> AutoDiffable<StaticArgs> for ADTensorDot<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    AOutput: TensorDot<BOutput, Output = Output>,
    AGrad: TensorDot<BOutput, Output = AGradB>,
    AOutput: TensorDot<BGrad, Output = ABGrad>,
    AGradB: Add<ABGrad, Output = Grad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.eval(x, static_args).tensordot(&self.1.eval(x, static_args))
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

        (f.tensordot(&g), df.tensordot(&g).add(f.tensordot(&dg)))
    }

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (f.tensordot(&g), df.tensordot(&g).add(f.tensordot(&dg)))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        df.tensordot(&g).add(f.tensordot(&dg))
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        df.tensordot(&g).add(f.tensordot(&dg))
    }
}

impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> ForwardDiffable<StaticArgs> for ADTensorDot<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: TensorDot<BOutput, Output = Output>,
    Output: Add<Output, Output = Output>,
{

    fn eval_forward(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let f = self.0.eval_forward(x, static_args);
        let g = self.1.eval_forward(x, static_args);

        f.tensordot(&g)
    }

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (f.tensordot(&g), df.tensordot(&g).add(f.tensordot(&dg)))
    }

    fn eval_forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                              static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        (f.tensordot(&g), df.tensordot(&g).add(f.tensordot(&dg)))
    }

    fn forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        df.tensordot(&g).add(f.tensordot(&dg))
    }

    fn forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        df.tensordot(&g).add(f.tensordot(&dg))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADTensorContraction<A, B, const N: usize>(pub A, pub B, pub ([usize; N], [usize; N]));
// N is the number of dimensions contracted over
// const generics MUST be LAST for the derive macro FuncCompose to work

impl<const N: usize, A: Diffable<StaticArgs>, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADTensorContraction<A, B, N>
where
    A::Output: TensorContraction<N, B::Output>,
{
    type Input = A::Input;
    type Output = <A::Output as TensorContraction<N, B::Output>>::Output;
}

impl<const N: usize, StaticArgs, Input, AOutput, BOutput, AGrad, BGrad, AGradB, ABGrad, Output, Grad, A, B> AutoDiffable<StaticArgs> for ADTensorContraction<A, B, N>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    AOutput: TensorContraction<N, BOutput, Output = Output>,
    AGrad: TensorContraction<N, BOutput, Output = AGradB>,
    AOutput: TensorContraction<N, BGrad, Output = ABGrad>,
    AGradB: Add<ABGrad, Output = Grad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.eval(x, static_args).contract(&self.1.eval(x, static_args), (&self.2.0, &self.2.1))
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

        (f.contract(&g, (&self.2.0, &self.2.1)), df.contract(&g, (&self.2.0, &self.2.1)).add(f.contract(&dg, (&self.2.0, &self.2.1))))
    }

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (f.contract(&g, (&self.2.0, &self.2.1)), df.contract(&g, (&self.2.0, &self.2.1)).add(f.contract(&dg, (&self.2.0, &self.2.1))))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        df.contract(&g, (&self.2.0, &self.2.1)).add(f.contract(&dg, (&self.2.0, &self.2.1)))
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        df.contract(&g, (&self.2.0, &self.2.1)).add(f.contract(&dg, (&self.2.0, &self.2.1)))
    }
}

impl<const N: usize, StaticArgs, Input, AOutput, BOutput, Output, A, B> ForwardDiffable<StaticArgs> for ADTensorContraction<A, B, N>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: TensorContraction<N, BOutput, Output = Output>,
    Output: Add<Output, Output = Output>,
{

    fn eval_forward(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let f = self.0.eval_forward(x, static_args);
        let g = self.1.eval_forward(x, static_args);

        f.contract(&g, (&self.2.0, &self.2.1))
    }

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (f.contract(&g, (&self.2.0, &self.2.1)), df.contract(&g, (&self.2.0, &self.2.1)).add(f.contract(&dg, (&self.2.0, &self.2.1))))
    }

    fn eval_forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                              static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        (f.contract(&g, (&self.2.0, &self.2.1)), df.contract(&g, (&self.2.0, &self.2.1)).add(f.contract(&dg, (&self.2.0, &self.2.1))))
    }

    fn forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        df.contract(&g, (&self.2.0, &self.2.1)).add(f.contract(&dg, (&self.2.0, &self.2.1)))
    }

    fn forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        df.contract(&g, (&self.2.0, &self.2.1)).add(f.contract(&dg, (&self.2.0, &self.2.1)))
    }
}

// operations with constants
#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantDot<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantDot<A, B>
where
    A::Output: Dot<B>,
{
    type Input = A::Input;
    type Output = <A::Output as Dot<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs> for ADConstantDot<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    Input: GradientType<Output, GradientType = Grad>,
    // ensure A.dot(B) is defined and returns type Output
    AOutput: Dot<B, Output = Output>,
    AGrad: Dot<B, Output = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.eval(x, static_args).dot(&self.1)
    }

    fn eval_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.dot(&self.1), df.dot(&self.1))
    }

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        (f.dot(&self.1), df.dot(&self.1))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        let df = self.0.grad(x, static_args);

        df.dot(&self.1)
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        let df = self.0.conj_grad(x, static_args);

        df.dot(&self.1)
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantDot<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    AOutput: Dot<B, Output = Output>,
    Output: Add<Output, Output = Output>,
{
    fn eval_forward(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let f = self.0.eval_forward(x, static_args);

        f.dot(&self.1)
    }

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.dot(&self.1), df.dot(&self.1))
    }

    fn eval_forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                              static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.dot(&self.1), df.dot(&self.1))
    }

    fn forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let df = self.0.forward_grad(x, dx, static_args);

        df.dot(&self.1)
    }

    fn forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        let df = self.0.forward_conj_grad(x, dx, static_args);

        df.dot(&self.1)
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantLeftDot<A, B>(pub A, pub B);
// dot product by constant from the left

impl<A, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADConstantLeftDot<A, B>
where
    A: Dot<B::Output>,
{
    type Input = B::Input;
    type Output = <A as Dot<B::Output>>::Output;
}

impl<StaticArgs, Input, Output, Grad, BOutput, BGrad, A, B> AutoDiffable<StaticArgs> for ADConstantLeftDot<A, B>
where
    A: Dot<BOutput, Output = Output>,
    A: Dot<BGrad, Output = Grad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.dot(&self.1.eval(x, static_args))
    }

    fn eval_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (g, dg) = self.1.eval_grad(x, static_args);

        (self.0.dot(&g), self.0.dot(&dg))
    }

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (self.0.dot(&g), self.0.dot(&dg))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        self.0.dot(&self.1.grad(x, static_args))
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        self.0.dot(&self.1.conj_grad(x, static_args))
    }
}

impl<StaticArgs, Input, Output, BOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantLeftDot<A, B>
where
    A: Dot<BOutput, Output = Output>,
    Output: Add<Output, Output = Output>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
{
    fn eval_forward(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.dot(&self.1.eval_forward(x, static_args))
    }

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (self.0.dot(&g), self.0.dot(&dg))
    }

    fn eval_forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                              static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        (self.0.dot(&g), self.0.dot(&dg))
    }

    fn forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.dot(&self.1.forward_grad(x, dx, static_args))
    }

    fn forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.dot(&self.1.forward_conj_grad(x, dx, static_args))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantTensorDot<A, B>(pub A, pub B);

impl<A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantTensorDot<A, B>
where
    A::Output: TensorDot<B>,
{
    type Input = A::Input;
    type Output = <A::Output as TensorDot<B>>::Output;
}

impl<StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs> for ADConstantTensorDot<A, B>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    Input: GradientType<Output, GradientType = Grad>,
    // ensure A.tensordot(B) is defined and returns type Output
    AOutput: TensorDot<B, Output = Output>,
    AGrad: TensorDot<B, Output = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.eval(x, static_args).tensordot(&self.1)
    }

    fn eval_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.tensordot(&self.1), df.tensordot(&self.1))
    }

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        (f.tensordot(&self.1), df.tensordot(&self.1))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        self.0.grad(x, static_args).tensordot(&self.1)
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        self.0.conj_grad(x, static_args).tensordot(&self.1)
    }
}

impl<StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantTensorDot<A, B>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    AOutput: TensorDot<B, Output = Output>,
{
    fn eval_forward(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.eval_forward(x, static_args).tensordot(&self.1)
    }

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.tensordot(&self.1), df.tensordot(&self.1))
    }

    fn eval_forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                              static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.tensordot(&self.1), df.tensordot(&self.1))
    }

    fn forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.forward_grad(x, dx, static_args).tensordot(&self.1)
    }

    fn forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.forward_conj_grad(x, dx, static_args).tensordot(&self.1)
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantLeftTensorDot<A, B>(pub A, pub B);

impl<A, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADConstantLeftTensorDot<A, B>
where
    A: TensorDot<B::Output>,
{
    type Input = B::Input;
    type Output = <A as TensorDot<B::Output>>::Output;
}

impl<StaticArgs, Input, Output, Grad, BOutput, BGrad, A, B> AutoDiffable<StaticArgs> for ADConstantLeftTensorDot<A, B>
where
    A: TensorDot<BOutput, Output = Output>,
    A: TensorDot<BGrad, Output = Grad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.tensordot(&self.1.eval(x, static_args))
    }

    fn eval_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (g, dg) = self.1.eval_grad(x, static_args);

        (self.0.tensordot(&g), self.0.tensordot(&dg))
    }

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (self.0.tensordot(&g), self.0.tensordot(&dg))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        self.0.tensordot(&self.1.grad(x, static_args))
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        self.0.tensordot(&self.1.conj_grad(x, static_args))
    }
}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantTensorContraction<A, B, const N: usize>(pub A, pub B, pub ([usize; N], [usize; N]));

impl<const N: usize, A: Diffable<StaticArgs>, B, StaticArgs> Diffable<StaticArgs> for ADConstantTensorContraction<A, B, N>
where
    A::Output: TensorContraction<N, B>,
{
    type Input = A::Input;
    type Output = <A::Output as TensorContraction<N, B>>::Output;
}

impl<const N: usize, StaticArgs, Input, Output, Grad, AOutput, AGrad, A, B> AutoDiffable<StaticArgs> for ADConstantTensorContraction<A, B, N>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    Input: GradientType<AOutput, GradientType = AGrad>,
    Input: GradientType<Output, GradientType = Grad>,
    AOutput: TensorContraction<N, B, Output = Output>,
    AGrad: TensorContraction<N, B, Output = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.eval(x, static_args).contract(&self.1, (&self.2.0, &self.2.1))
    }

    fn eval_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.contract(&self.1, (&self.2.0, &self.2.1)), df.contract(&self.1, (&self.2.0, &self.2.1)))
    }

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (f, df) = self.0.eval_conj_grad(x, static_args);

        (f.contract(&self.1, (&self.2.0, &self.2.1)), df.contract(&self.1, (&self.2.0, &self.2.1)))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        self.0.grad(x, static_args).contract(&self.1, (&self.2.0, &self.2.1))
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        self.0.conj_grad(x, static_args).contract(&self.1, (&self.2.0, &self.2.1))
    }
}

impl<const N: usize, StaticArgs, Input, Output, AOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantTensorContraction<A, B, N>
where
    A: ForwardDiffable<StaticArgs, Input = Input, Output = AOutput>,
    AOutput: TensorContraction<N, B, Output = Output>,
{
    fn eval_forward(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.eval_forward(x, static_args).contract(&self.1, (&self.2.0, &self.2.1))
    }

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_grad(x, dx, static_args);

        (f.contract(&self.1, (&self.2.0, &self.2.1)), df.contract(&self.1, (&self.2.0, &self.2.1)))
    }

    fn eval_forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                              static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (f, df) = self.0.eval_forward_conj_grad(x, dx, static_args);

        (f.contract(&self.1, (&self.2.0, &self.2.1)), df.contract(&self.1, (&self.2.0, &self.2.1)))
    }

    fn forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.forward_grad(x, dx, static_args).contract(&self.1, (&self.2.0, &self.2.1))
    }

    fn forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.forward_conj_grad(x, dx, static_args).contract(&self.1, (&self.2.0, &self.2.1))
    }

}

#[derive(FuncCompose, Debug, Clone, Copy)]
pub struct ADConstantLeftTensorContraction<A, B, const N: usize>(pub A, pub B, pub ([usize; N], [usize; N]));

impl<const N: usize, A, B: Diffable<StaticArgs>, StaticArgs> Diffable<StaticArgs> for ADConstantLeftTensorContraction<A, B, N>
where
    A: TensorContraction<N, B::Output>,
{
    type Input = B::Input;
    type Output = <A as TensorContraction<N, B::Output>>::Output;
}

impl<const N: usize, StaticArgs, Input, Output, Grad, BOutput, BGrad, A, B> AutoDiffable<StaticArgs> for ADConstantLeftTensorContraction<A, B, N>
where
    A: TensorContraction<N, BOutput, Output = Output>,
    A: TensorContraction<N, BGrad, Output = Grad>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    Input: GradientType<BOutput, GradientType = BGrad>,
    Input: GradientType<Output, GradientType = Grad>,
{
    fn eval(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.contract(&self.1.eval(x, static_args), (&self.2.0, &self.2.1))
    }

    fn eval_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (g, dg) = self.1.eval_grad(x, static_args);

        (self.0.contract(&g, (&self.2.0, &self.2.1)), self.0.contract(&dg, (&self.2.0, &self.2.1)))
    }

    fn eval_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                      static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            Grad
        )
    {
        let (g, dg) = self.1.eval_conj_grad(x, static_args);

        (self.0.contract(&g, (&self.2.0, &self.2.1)), self.0.contract(&dg, (&self.2.0, &self.2.1)))
    }

    fn grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
            static_args: &StaticArgs) -> Grad
    {
        self.0.contract(&self.1.grad(x, static_args), (&self.2.0, &self.2.1))
    }

    fn conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                 static_args: &StaticArgs) -> Grad
    {
        self.0.contract(&self.1.conj_grad(x, static_args), (&self.2.0, &self.2.1))
    }
}

impl<const N: usize, StaticArgs, Input, Output, BOutput, A, B> ForwardDiffable<StaticArgs> for ADConstantLeftTensorContraction<A, B, N>
where
    A: TensorContraction<N, BOutput, Output = Output>,
    Output: Add<Output, Output = Output>,
    B: ForwardDiffable<StaticArgs, Input = Input, Output = BOutput>,
{
    fn eval_forward(&self, x: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.contract(&self.1.eval_forward(x, static_args), (&self.2.0, &self.2.1))
    }

    fn eval_forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (g, dg) = self.1.eval_forward_grad(x, dx, static_args);

        (self.0.contract(&g, (&self.2.0, &self.2.1)), self.0.contract(&dg, (&self.2.0, &self.2.1)))
    }

    fn eval_forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                              static_args: &StaticArgs) ->
        (
            <Self as Diffable<StaticArgs>>::Output,
            <Self as Diffable<StaticArgs>>::Output
        )
    {
        let (g, dg) = self.1.eval_forward_conj_grad(x, dx, static_args);

        (self.0.contract(&g, (&self.2.0, &self.2.1)), self.0.contract(&dg, (&self.2.0, &self.2.1)))
    }

    fn forward_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                    static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.contract(&self.1.forward_grad(x, dx, static_args), (&self.2.0, &self.2.1))
    }

    fn forward_conj_grad(&self, x: &<Self as Diffable<StaticArgs>>::Input, dx: &<Self as Diffable<StaticArgs>>::Input,
                         static_args: &StaticArgs) -> <Self as Diffable<StaticArgs>>::Output
    {
        self.0.contract(&self.1.forward_conj_grad(x, dx, static_args), (&self.2.0, &self.2.1))
    }
}

