use crate::adops::*;
use crate::autodiffable::AutoDiffable;
use crate::func_traits;
use num::traits::bounds::UpperBounded;
use num::traits::{Pow, Signed};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A wrapper type for an AutoDiffable type.
#[derive(Debug, Clone)]
pub struct AutoDiff<StaticArgsType, InputType, OutputType, GradType, T>(
    pub T,
    pub PhantomData<(StaticArgsType, InputType, OutputType, GradType)>,
);

/// Impl Copy for AutoDiff if T is Copy and all other types are Clone
impl<StaticArgsType, InputType, OutputType, GradType, T> Copy
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, T>
where
    StaticArgsType: Clone,
    InputType: Clone,
    OutputType: Clone,
    GradType: Clone,
    T: Copy,
{
}

/// Impl of new for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, T>
    AutoDiff<StaticArgsType, InputType, OutputType, GradType, T>
where
{
    #[allow(dead_code)]
    pub fn new(t: T) -> Self {
        AutoDiff(t, PhantomData)
    }
}

/// Impl of AutoDiffable for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, T>
    AutoDiffable<StaticArgsType, InputType, OutputType, GradType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, T>
where
    T: AutoDiffable<StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args)
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        self.0.eval_grad(x, static_args)
    }
}

/// Impl of Add for AutoDiff
impl<StaticArgsType, InputType, AOutputType, BOutputType, AGradType, BGradType, A, B>
    Add<AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>>
    for AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, A>
where
    AOutputType: Add<BOutputType>,
    AGradType: Add<BGradType>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, ADAdd<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn add(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>,
    ) -> Self::Output {
        AutoDiff(ADAdd(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Sub for AutoDiff
impl<StaticArgsType, InputType, AOutputType, BOutputType, AGradType, BGradType, A, B>
    Sub<AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>>
    for AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, A>
where
    AOutputType: Sub<BOutputType>,
    AGradType: Sub<BGradType>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, ADSub<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn sub(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>,
    ) -> Self::Output {
        AutoDiff(ADSub(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Mul for AutoDiff
impl<'a, 'b, StaticArgsType, InputType, AOutputType, BOutputType, AGradType, BGradType, A, B>
    Mul<AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>>
    for AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, A>
where
    AOutputType: Mul<BOutputType> + Clone,
    BOutputType: Clone,
    AGradType: Mul<BOutputType>,
    BGradType: Mul<AOutputType>,
    <AGradType as Mul<BOutputType>>::Output: Add<<BGradType as Mul<AOutputType>>::Output>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, ADMul<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn mul(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>,
    ) -> Self::Output {
        AutoDiff(ADMul(self.0, _other.0, PhantomData), PhantomData)
    }
}




/*
/// Impl of Sub for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, A, B>
    Sub<AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADSub<A, B>>;

    fn sub(
        self,
        _other: AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADSub<A, B>> {
        AutoDiff(ADSub(self.0, _other.0), PhantomData)
    }
}

/// Impl of Mul for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, A, B>
    Mul<AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADMul<A, B>>;

    fn mul(
        self,
        _other: AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADMul<A, B>> {
        AutoDiff(ADMul(self.0, _other.0), PhantomData)
    }
}

/// Impl of Div for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, A, B>
    Div<AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADDiv<A, B>>;

    fn div(
        self,
        _other: AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADDiv<A, B>> {
        AutoDiff(ADDiv(self.0, _other.0), PhantomData)
    }
}

/// Impl of Neg for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, A> Neg
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADNeg<A>>;

    fn neg(self) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADNeg<A>> {
        AutoDiff(ADNeg(self.0), PhantomData)
    }
}

/// Impl of Compose for AutoDiff
impl<
        StaticArgsType,
        InputType,
        InnerOutputType,
        OuterOutputType,
        InnerGradType,
        OuterGradType,
        Outer,
        Inner,
    >
    func_traits::Compose<AutoDiff<StaticArgsType, InputType, InnerOutputType, InnerGradType, Inner>>
    for AutoDiff<StaticArgsType, InnerOutputType, OuterOutputType, OuterGradType, Outer>
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
    type Output = AutoDiff<
        StaticArgsType,
        InputType,
        OuterOutputType,
        OuterGradType,
        ADCompose<
            Outer,
            Inner,
            StaticArgsType,
            InputType,
            InnerOutputType,
            OuterOutputType,
            InnerGradType,
            OuterGradType,
        >,
    >;
    fn compose(
        self,
        _other: AutoDiff<StaticArgsType, InputType, InnerOutputType, InnerGradType, Inner>,
    ) -> AutoDiff<
        StaticArgsType,
        InputType,
        OuterOutputType,
        OuterGradType,
        ADCompose<
            Outer,
            Inner,
            StaticArgsType,
            InputType,
            InnerOutputType,
            OuterOutputType,
            InnerGradType,
            OuterGradType,
        >,
    > {
        AutoDiff(ADCompose(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of constant Add for AutoDiff (Add<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantAdd
impl<StaticArgsType, InputType, OutputType, GradType, A> Add<OutputType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output =
        AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantAdd<A, OutputType>>;

    fn add(
        self,
        _other: OutputType,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantAdd<A, OutputType>>
    {
        AutoDiff(ADConstantAdd(self.0, _other), PhantomData)
    }
}

/// Impl of constant Sub for AutoDiff (Sub<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantSub
impl<StaticArgsType, InputType, OutputType, GradType, A> Sub<OutputType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output =
        AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantSub<A, OutputType>>;

    fn sub(
        self,
        _other: OutputType,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantSub<A, OutputType>>
    {
        AutoDiff(ADConstantSub(self.0, _other), PhantomData)
    }
}

/// Impl of constant Mul for AutoDiff (Mul<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantMul
impl<StaticArgsType, InputType, OutputType, GradType, A> Mul<OutputType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output =
        AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantMul<A, OutputType>>;

    fn mul(
        self,
        _other: OutputType,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantMul<A, OutputType>>
    {
        AutoDiff(ADConstantMul(self.0, _other), PhantomData)
    }
}

/// Impl of constant Div for AutoDiff (Div<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantDiv
impl<StaticArgsType, InputType, OutputType, GradType, A> Div<OutputType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output =
        AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantDiv<A, OutputType>>;

    fn div(
        self,
        _other: OutputType,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantDiv<A, OutputType>>
    {
        AutoDiff(ADConstantDiv(self.0, _other), PhantomData)
    }
}

/// Impl of constant Pow for AutoDiff (Pow<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantPow
impl<StaticArgsType, InputType, OutputType, GradType, A> Pow<OutputType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type Output =
        AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantPow<A, OutputType>>;

    fn pow(
        self,
        _other: OutputType,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantPow<A, OutputType>>
    {
        AutoDiff(ADConstantPow(self.0, _other), PhantomData)
    }
}

/// Impl of Signed
impl<StaticArgsType, InputType, OutputType: Signed, GradType: UpperBounded, A> func_traits::Signed
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
    type AbsType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADAbs<A>>;
    type SignType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADSignum<A>>;

    fn abs(self) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADAbs<A>> {
        AutoDiff(ADAbs(self.0), PhantomData)
    }

    fn signum(self) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADSignum<A>> {
        AutoDiff(ADSignum(self.0), PhantomData)
    }
}*/
