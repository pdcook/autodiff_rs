use crate::adops::*;
use crate::autodiffable::{Diffable, AutoDiffable, ForwardDiffable};
use crate::func_traits;
use crate::traits::{InstOne, InstZero};
use num::traits::bounds::UpperBounded;
use num::traits::{Pow, Signed};
use std::marker::PhantomData;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use crate::gradienttype::GradientType;
use crate::forward::ForwardMul;

/// A wrapper type for an AutoDiffable type.
#[derive(Debug, Clone)]
pub struct AutoDiff<StaticArgs, T>(pub T, pub PhantomData<StaticArgs>);

/// Impl Copy for AutoDiff if T is Copy and all other types are Clone
impl<StaticArgs, T> Copy for AutoDiff<StaticArgs, T>
where
    StaticArgs: Clone,
    T: Copy + Diffable,
{
}

/// Impl of new for AutoDiff
impl<StaticArgs, T> AutoDiff<StaticArgs, T> {
    #[allow(dead_code)]
    pub fn new(t: T) -> Self {
        AutoDiff(t, PhantomData)
    }

    pub fn coerce<NewInputType, NewOutputType>(
        self,
    ) -> AutoDiff<StaticArgs, ADCoerce<T, NewInputType, NewOutputType>>
    {
        AutoDiff(ADCoerce(self.0, PhantomData), PhantomData)
    }

    pub fn append_static_args<NewStaticArgs>(
        self,
    ) -> AutoDiff<(StaticArgs, NewStaticArgs), ADAppendStaticArgs<T, NewStaticArgs>>
    {
        AutoDiff(ADAppendStaticArgs(self.0, PhantomData), PhantomData)
    }

    pub fn prepend_static_args<NewStaticArgs>(
        self,
    ) -> AutoDiff<(NewStaticArgs, StaticArgs), ADPrependStaticArgs<T, NewStaticArgs>>
    {
        AutoDiff(ADPrependStaticArgs(self.0, PhantomData), PhantomData)
    }
}

/// Impl of Diffable for AutoDiff
impl<StaticArgs, T> Diffable for AutoDiff<StaticArgs, T>
where
    T: Diffable,
{
}

/// Impl of AutoDiffable for AutoDiff
impl<StaticArgs, Input, Output, Grad, T> AutoDiffable<StaticArgs> for AutoDiff<StaticArgs, T>
where
    T: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Input: GradientType<Output, GradientType = Grad>,
{
    type Input = Input;
    type Output = Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args)
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Grad) {
        self.0.eval_grad(x, static_args)
    }
}

/// Impl of ForwardDiffable for AutoDiff
impl<StaticArgs, Input, Output, T> ForwardDiffable<StaticArgs> for AutoDiff<StaticArgs, T>
where
    T: ForwardDiffable<StaticArgs, Input = Input, Output = Output>,
{
    type Input = Input;
    type Output = Output;

    fn eval_forward(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval_forward(x, static_args)
    }

    fn eval_forward_grad(&self, x: &Self::Input, dx: &Self::Input, static_args: &StaticArgs) -> (Self::Output, Self::Output) {
        self.0.eval_forward_grad(x, dx, static_args)
    }
}

/// Impl of Deref for AutoDiff
impl<StaticArgs, T> Deref for AutoDiff<StaticArgs, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Impl of Add for AutoDiff
impl<StaticArgs, A, B> Add<AutoDiff<StaticArgs, B>>
    for AutoDiff<StaticArgs, A>
where
    A: Diffable,
    B: Diffable,
{
    type Output = AutoDiff<StaticArgs, ADAdd<A, B>>;

    fn add(self, _other: AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADAdd(self.0, _other.0), PhantomData)
    }
}

/// Impl of Sub for AutoDiff
impl<StaticArgs, A, B> Sub<AutoDiff<StaticArgs, B>>
    for AutoDiff<StaticArgs, A>
where
    A: Diffable,
    B: Diffable,
{
    type Output = AutoDiff<StaticArgs, ADSub<A, B>>;

    fn sub(self, _other: AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADSub(self.0, _other.0), PhantomData)
    }
}


/// Impl of Mul for AutoDiff
impl<StaticArgs, A, B> Mul<AutoDiff<StaticArgs, B>>
    for AutoDiff<StaticArgs, A>
where
    A: Diffable,
    B: Diffable,
{
    type Output = AutoDiff<StaticArgs, ADMul<A, B>>;

    fn mul(self, _other: AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADMul(self.0, _other.0), PhantomData)
    }
}

/// Impl of Div for AutoDiff
impl<StaticArgs, A, B> Div<AutoDiff<StaticArgs, B>>
    for AutoDiff<StaticArgs, A>
where
    A: Diffable,
    B: Diffable,
{
    type Output = AutoDiff<StaticArgs, ADDiv<A, B>>;

    fn div(self, _other: AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADDiv(self.0, _other.0), PhantomData)
    }
}

/// Impl of Neg for AutoDiff
impl<StaticArgs, A> Neg for AutoDiff<StaticArgs, A>
{
    type Output = AutoDiff<StaticArgs, ADNeg<A>>;

    fn neg(self) -> Self::Output {
        AutoDiff(ADNeg(self.0), PhantomData)
    }
}

/// Impl of Compose for AutoDiff
impl<StaticArgs, Outer, Inner>
    func_traits::Compose<AutoDiff<StaticArgs, Inner>> for AutoDiff<StaticArgs, Outer>
where
    Outer: Diffable,
    Inner: Diffable,
{
    type Output = AutoDiff<StaticArgs, ADCompose<Outer, Inner>>;

    fn compose(self, _other: AutoDiff<StaticArgs, Inner>) -> Self::Output {
        AutoDiff(ADCompose(self.0, _other.0), PhantomData)
    }
}

/// Impl constant Add for AutoDiff
impl<StaticArgs, A, B> Add<B> for AutoDiff<StaticArgs, A>
where
    // ensure B is Clone and B.zero is defined
    B: Clone + InstZero,
{
    type Output = AutoDiff<StaticArgs, ADConstantAdd<A, B>>;

    fn add(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantAdd(self.0, _other), PhantomData)
    }
}

/// Impl constant Sub for AutoDiff
impl<StaticArgs, A, B> Sub<B> for AutoDiff<StaticArgs, A>
where
    B: Clone + InstZero,
{
    type Output = AutoDiff<StaticArgs, ADConstantSub<A, B>>;

    fn sub(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantSub(self.0, _other), PhantomData)
    }
}

/// Impl constant Mul for AutoDiff
impl<StaticArgs, A, B> Mul<B> for AutoDiff<StaticArgs, A>
where
    // ensure B is Clone
    B: Clone + InstOne,
{
    type Output = AutoDiff<StaticArgs, ADConstantMul<A, B>>;

    fn mul(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantMul(self.0, _other), PhantomData)
    }
}

/// Impl constant Div for AutoDiff
impl<StaticArgs, A, B> Div<B> for AutoDiff<StaticArgs, A>
where
    // ensure B is Clone
    B: Clone + InstOne,
{
    type Output = AutoDiff<StaticArgs, ADConstantDiv<A, B>>;

    fn div(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantDiv(self.0, _other), PhantomData)
    }
}

/// Impl constant Pow for AutoDiff
impl<StaticArgs, A, B> Pow<B> for AutoDiff<StaticArgs, A>
where
    // ensure B is Clone and B.one is defined and B-1 is B
    B: Clone + InstOne + Sub<B, Output = B>,
{
    type Output = AutoDiff<StaticArgs, ADConstantPow<A, B>>;

    fn pow(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantPow(self.0, _other), PhantomData)
    }
}

/// Impl Abs
impl<StaticArgs, A> func_traits::Abs for AutoDiff<StaticArgs, A>
{
    type Output = AutoDiff<StaticArgs, ADAbs<A>>;
    fn abs(self) -> Self::Output {
        AutoDiff(ADAbs(self.0), PhantomData)
    }
}

/// Impl Signum
impl<StaticArgs, A> func_traits::Signum for AutoDiff<StaticArgs, A>
{
    type Output = AutoDiff<StaticArgs, ADSignum<A>>;
    fn signum(self) -> Self::Output {
        AutoDiff(ADSignum(self.0), PhantomData)
    }
}
