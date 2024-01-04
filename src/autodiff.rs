use crate::adops::*;
use crate::autodiffable::AutoDiffable;
use crate::func_traits;
use crate::traits::{InstOne, InstZero};
use num::traits::bounds::UpperBounded;
use num::traits::{Pow, Signed};
use std::marker::PhantomData;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};

/// A wrapper type for an AutoDiffable type.
#[derive(Debug, Clone)]
pub struct AutoDiff<StaticArgs, T>(pub T, pub PhantomData<StaticArgs>);

/// Impl Copy for AutoDiff if T is Copy and all other types are Clone
impl<StaticArgs, T> Copy for AutoDiff<StaticArgs, T>
where
    StaticArgs: Clone,
    T: Copy,
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

/// Impl of AutoDiffable for AutoDiff
impl<StaticArgs, T> AutoDiffable<StaticArgs> for AutoDiff<StaticArgs, T>
where
    T: AutoDiffable<StaticArgs>,
{
    type Input = <T as AutoDiffable<StaticArgs>>::Input;
    type Output = <T as AutoDiffable<StaticArgs>>::Output;
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.0.eval(x, static_args)
    }

    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output) {
        self.0.eval_grad(x, dx, static_args)
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
impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> Add<AutoDiff<StaticArgs, B>>
    for AutoDiff<StaticArgs, A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Add<BOutput, Output = Output>,
{
    type Output = AutoDiff<StaticArgs, ADAdd<A, B>>;

    fn add(self, _other: AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADAdd(self.0, _other.0), PhantomData)
    }
}

/// Impl of Sub for AutoDiff
impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> Sub<AutoDiff<StaticArgs, B>>
    for AutoDiff<StaticArgs, A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    B: AutoDiffable<StaticArgs, Input = Input, Output = BOutput>,
    AOutput: Sub<BOutput, Output = Output>,
{
    type Output = AutoDiff<StaticArgs, ADSub<A, B>>;

    fn sub(self, _other: AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADSub(self.0, _other.0), PhantomData)
    }
}

/// Impl of Mul for AutoDiff
impl<StaticArgs, Input, AOutput, BOutput, Output, A, B> Mul<AutoDiff<StaticArgs, B>>
    for AutoDiff<StaticArgs, A>
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
    type Output = AutoDiff<StaticArgs, ADMul<A, B>>;

    fn mul(self, _other: AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADMul(self.0, _other.0), PhantomData)
    }
}

/// Impl of Div for AutoDiff
impl<StaticArgs, Input, AOutput, BOutput, Output, BB, AB, ABOvBB, A, B> Div<AutoDiff<StaticArgs, B>>
    for AutoDiff<StaticArgs, A>
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
    type Output = AutoDiff<StaticArgs, ADDiv<A, B>>;

    fn div(self, _other: AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADDiv(self.0, _other.0), PhantomData)
    }
}

/// Impl of Neg for AutoDiff
impl<StaticArgs, Input, A> Neg for AutoDiff<StaticArgs, A>
where
    // ensure A has Neg
    <A as AutoDiffable<StaticArgs>>::Output: Neg,
    A: AutoDiffable<StaticArgs, Input = Input>,
{
    type Output = AutoDiff<StaticArgs, ADNeg<A>>;

    fn neg(self) -> Self::Output {
        AutoDiff(ADNeg(self.0), PhantomData)
    }
}

/// Impl of Compose for AutoDiff
impl<StaticArgs, InnerInput, InnerOutput, OuterInput, OuterOutput, Outer, Inner>
    func_traits::Compose<AutoDiff<StaticArgs, Inner>> for AutoDiff<StaticArgs, Outer>
where
    Outer: AutoDiffable<StaticArgs, Input = OuterInput, Output = OuterOutput>,
    Inner: AutoDiffable<StaticArgs, Input = InnerInput, Output = InnerOutput>,
    OuterInput: From<InnerOutput>,
{
    type Output = AutoDiff<StaticArgs, ADCompose<Outer, Inner>>;

    fn compose(self, _other: AutoDiff<StaticArgs, Inner>) -> Self::Output {
        AutoDiff(ADCompose(self.0, _other.0), PhantomData)
    }
}

/// Impl constant Add for AutoDiff
impl<StaticArgs, Input, AOutput, Output, A, B> Add<B> for AutoDiff<StaticArgs, A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A + B is defined and Output = A + B
    AOutput: Add<B, Output = Output>,
    // ensure B is Clone and B.zero is defined
    B: Clone + InstZero,
{
    type Output = AutoDiff<StaticArgs, ADConstantAdd<A, B>>;

    fn add(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantAdd(self.0, _other), PhantomData)
    }
}

/// Impl constant Sub for AutoDiff
impl<StaticArgs, Input, AOutput, Output, A, B> Sub<B> for AutoDiff<StaticArgs, A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A - B is defined and Output = A - B
    AOutput: Sub<B, Output = Output>,
    // ensure B is Clone and B.zero is defined
    B: Clone + InstZero,
{
    type Output = AutoDiff<StaticArgs, ADConstantSub<A, B>>;

    fn sub(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantSub(self.0, _other), PhantomData)
    }
}

/// Impl constant Mul for AutoDiff
impl<StaticArgs, Input, AOutput, Output, A, B> Mul<B> for AutoDiff<StaticArgs, A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A * B is defined and Output = A * B
    AOutput: Mul<B, Output = Output>,
    // ensure B is Clone
    B: Clone + InstOne,
{
    type Output = AutoDiff<StaticArgs, ADConstantMul<A, B>>;

    fn mul(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantMul(self.0, _other), PhantomData)
    }
}

/// Impl constant Div for AutoDiff
impl<StaticArgs, Input, AOutput, Output, A, B> Div<B> for AutoDiff<StaticArgs, A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = AOutput>,
    // ensure A / B is defined and Output = A / B
    AOutput: Div<B, Output = Output>,
    // ensure B is Clone
    B: Clone + InstOne,
{
    type Output = AutoDiff<StaticArgs, ADConstantDiv<A, B>>;

    fn div(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantDiv(self.0, _other), PhantomData)
    }
}

/// Impl constant Pow for AutoDiff
impl<StaticArgs, Input, AOutput, Output, A, B> Pow<B> for AutoDiff<StaticArgs, A>
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
    type Output = AutoDiff<StaticArgs, ADConstantPow<A, B>>;

    fn pow(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantPow(self.0, _other), PhantomData)
    }
}

/// Impl Abs
impl<StaticArgs, Input, Output, A> func_traits::Abs for AutoDiff<StaticArgs, A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Output: Signed + Mul<Output, Output = Output>,
{
    type Output = AutoDiff<StaticArgs, ADAbs<A>>;
    fn abs(self) -> Self::Output {
        AutoDiff(ADAbs(self.0), PhantomData)
    }
}

/// Impl Signum
impl<StaticArgs, Input, Output, A> func_traits::Signum for AutoDiff<StaticArgs, A>
where
    A: AutoDiffable<StaticArgs, Input = Input, Output = Output>,
    Output: InstZero + Signed + UpperBounded,
{
    type Output = AutoDiff<StaticArgs, ADSignum<A>>;
    fn signum(self) -> Self::Output {
        AutoDiff(ADSignum(self.0), PhantomData)
    }
}
