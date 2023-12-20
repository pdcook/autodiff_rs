use crate::adops::*;
use crate::arithmetic::{Arithmetic, StrongAssociatedArithmetic, WeakAssociatedArithmetic, StrongAssociatedExtendedArithmetic, WeakAssociatedExtendedArithmetic};
use crate::autodiffable::AutoDiffable;
use crate::func_traits;
use num::traits::{Pow, Signed};
use num::traits::bounds::UpperBounded;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A wrapper type for an AutoDiffable type.
#[derive(Debug, Clone, Copy)]
pub struct AutoDiff<
    StaticArgsType,
    InputType: Arithmetic,
    OutputType: WeakAssociatedArithmetic<GradType>,
    GradType: StrongAssociatedArithmetic<OutputType>,
    T: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
>(
    pub T,
    pub PhantomData<(StaticArgsType, InputType, OutputType, GradType)>,
);

/// Impl of new for AutoDiff
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        T: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiff<StaticArgsType, InputType, OutputType, GradType, T>
{
    #[allow(dead_code)]
    pub fn new(t: T) -> Self {
        AutoDiff(t, PhantomData)
    }
}

/// Impl of AutoDiffable for AutoDiff
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        T: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > AutoDiffable<StaticArgsType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, T>
{
    type InType = InputType;
    type OutType = OutputType;
    type GradType = GradType;

    fn eval(&self, x: InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args)
    }

    fn grad(&self, x: InputType, static_args: &StaticArgsType) -> GradType {
        self.0.grad(x, static_args)
    }
}

/// Impl of Add for AutoDiff
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
        B: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Add<AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADAdd<A, B>>;

    fn add(
        self,
        _other: AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>,
    ) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADAdd<A, B>> {
        AutoDiff(ADAdd(self.0, _other.0), PhantomData)
    }
}

/// Impl of Sub for AutoDiff
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
        B: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Sub<AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
        B: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Mul<AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
        B: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Div<AutoDiff<StaticArgsType, InputType, OutputType, GradType, B>>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
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
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Neg for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADNeg<A>>;

    fn neg(self) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADNeg<A>> {
        AutoDiff(ADNeg(self.0), PhantomData)
    }
}

/// Impl of Compose for AutoDiff
impl<
    StaticArgsType,
    IIT: Arithmetic,
    IOT: WeakAssociatedArithmetic<IGT>,
    IGT: StrongAssociatedArithmetic<IOT> + WeakAssociatedArithmetic<OGT>,
    OOT: WeakAssociatedArithmetic<OGT>,
    OGT: StrongAssociatedArithmetic<OOT> + StrongAssociatedArithmetic<IGT>,
    Outer: AutoDiffable<
        StaticArgsType,
        InType = IOT,
        OutType = OOT,
        GradType = OGT,
    >,
    Inner: AutoDiffable<
        StaticArgsType,
        InType = IIT,
        OutType = IOT,
        GradType = IGT,
    >,
    >
    // Compose<A, Output> for f is f.compose(g: A) -> Output
    func_traits::Compose<
        AutoDiff<StaticArgsType, IIT, IOT, IGT, Inner>,
    >
    for AutoDiff<StaticArgsType, IOT, OOT, OGT, Outer>
{
    type Output = AutoDiff<StaticArgsType, IIT, OOT, OGT, ADCompose<Outer, Inner>>;
    fn compose(
        self,
        _other: AutoDiff<StaticArgsType, IIT, IOT, IGT, Inner>,
    ) -> AutoDiff<StaticArgsType, IIT, OOT, OGT, ADCompose<Outer, Inner>> {
        AutoDiff(ADCompose(self.0, _other.0), PhantomData)
    }
}

/// Impl of constant Add for AutoDiff (Add<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantAdd
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Add<OutputType> for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantAdd<A, OutputType>>;

    fn add(self, _other: OutputType) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantAdd<A, OutputType>> {
        AutoDiff(ADConstantAdd(self.0, _other), PhantomData)
    }
}

/// Impl of constant Sub for AutoDiff (Sub<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantSub
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Sub<OutputType> for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantSub<A, OutputType>>;

    fn sub(self, _other: OutputType) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantSub<A, OutputType>> {
        AutoDiff(ADConstantSub(self.0, _other), PhantomData)
    }
}

/// Impl of constant Mul for AutoDiff (Mul<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantMul
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Mul<OutputType> for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantMul<A, OutputType>>;

    fn mul(self, _other: OutputType) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantMul<A, OutputType>> {
        AutoDiff(ADConstantMul(self.0, _other), PhantomData)
    }
}

/// Impl of constant Div for AutoDiff (Div<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantDiv
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedArithmetic<GradType>,
        GradType: StrongAssociatedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Div<OutputType> for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantDiv<A, OutputType>>;

    fn div(self, _other: OutputType) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantDiv<A, OutputType>> {
        AutoDiff(ADConstantDiv(self.0, _other), PhantomData)
    }
}

/// Impl of constant Pow for AutoDiff (Pow<B> where B is a constant
/// of type OutputType)
/// this uses ADConstantPow
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedExtendedArithmetic<GradType>,
        GradType: StrongAssociatedExtendedArithmetic<OutputType>,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > Pow<OutputType> for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantPow<A, OutputType>>;

    fn pow(self, _other: OutputType) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADConstantPow<A, OutputType>> {
        AutoDiff(ADConstantPow(self.0, _other), PhantomData)
    }
}

/// Impl of Signed
impl<
        StaticArgsType,
        InputType: Arithmetic,
        OutputType: WeakAssociatedExtendedArithmetic<GradType> + Signed,
        GradType: StrongAssociatedExtendedArithmetic<OutputType> + UpperBounded,
        A: AutoDiffable<StaticArgsType, InType = InputType, OutType = OutputType, GradType = GradType>,
    > func_traits::Signed for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
{
    type AbsType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADAbs<A>>;
    type SignType = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADSignum<A>>;

    fn abs(self) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADAbs<A>> {
        AutoDiff(ADAbs(self.0), PhantomData)
    }

    fn signum(self) -> AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADSignum<A>> {
        AutoDiff(ADSignum(self.0), PhantomData)
    }
}
