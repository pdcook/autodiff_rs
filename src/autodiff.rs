use crate::adops::*;
use crate::autodiffable::{AutoDiffable, CustomForwardDiff};
use crate::func_traits;
use num::traits::bounds::UpperBounded;
use num::traits::{Pow, Signed};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub, Deref};
use crate::traits::{InstZero, InstOne, ComposedGradMul};

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

/// Impl of CustomForwardDiff for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, ForwardGradType, T>
    CustomForwardDiff<StaticArgsType, InputType, OutputType, GradType, ForwardGradType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, T>
where
    T: CustomForwardDiff<StaticArgsType, InputType, OutputType, GradType, ForwardGradType>,
{
    fn forward_eval_grad(
        &self,
        x: &InputType,
        dx: &ForwardGradType,
        static_args: &StaticArgsType,
    ) -> (OutputType, GradType) {
        self.0.forward_eval_grad(x, dx, static_args)
    }
}

/// Impl of Deref for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, T> Deref
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, T>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
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
    type Output = AutoDiff<StaticArgsType, InputType, <AOutputType as Add<BOutputType>>::Output, <AGradType as Add<BGradType>>::Output, ADAdd<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

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
    type Output = AutoDiff<StaticArgsType, InputType, <AOutputType as Sub<BOutputType>>::Output, <AGradType as Sub<BGradType>>::Output, ADSub<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn sub(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>,
    ) -> Self::Output {
        AutoDiff(ADSub(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Mul for AutoDiff
impl<StaticArgsType, InputType, AOutputType, BOutputType, AGradType, BGradType, A, B>
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
    type Output = AutoDiff<
        StaticArgsType,
        InputType,
        <AOutputType as Mul<BOutputType>>::Output,
        <<AGradType as Mul<BOutputType>>::Output as Add<<BGradType as Mul<AOutputType>>::Output>>::Output
        , ADMul<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn mul(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>,
    ) -> Self::Output {
        AutoDiff(ADMul(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Div for AutoDiff

impl<StaticArgsType, InputType, AOutputType, BOutputType, AGradType, BGradType, A, B>
    Div<AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>>
    for AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, A>
where
    AOutputType: Div<BOutputType> + Clone, // f/g
    BOutputType: Clone + Mul<BOutputType>, // g^2
    AGradType: Div<BOutputType>, // df/g
    BGradType: Mul<AOutputType>, // dg*f
    <BGradType as Mul<AOutputType>>::Output: Div<<BOutputType as Mul<BOutputType>>::Output>, // (dg*f)/g^2
      <AGradType as Div<BOutputType>>::Output: Sub< < <BGradType as Mul<AOutputType>>::Output as Div < <BOutputType as Mul<BOutputType>>::Output > >::Output >,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType>,
{
    type Output = AutoDiff<
        StaticArgsType,
        InputType,
        <AOutputType as Div<BOutputType>>::Output,
        // (df/g - f dg/g^2)
        // = ((df/g) - (dg * f) / g^2)
        <                                              //------------------+
          <AGradType as Div<BOutputType>>::Output      // df/g             |
          as Sub                                       //                  |
          <                                            //                  |
            <                                          // -----+           |
            <BGradType as Mul<AOutputType>>::Output    // dg*f |           |- df/g - dg*f/g^2
              as Div                                   //      |           |
              <                                        //      |- dg*f/g^2 |
                <BOutputType as Mul<BOutputType>>::Output// g^2  |           |
              >                                        //      |           |
            >::Output                                  // -----+           |
          >                                            //                  |
        >::Output                                      //------------------+
        , ADDiv<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn div(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, B>,
    ) -> Self::Output {
        AutoDiff(ADDiv(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Neg for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, A>
    Neg for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
where
    OutputType: Neg,
    GradType: Neg,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, <OutputType as Neg>::Output, <GradType as Neg>::Output
    , ADNeg<A, OutputType, GradType>>;

    fn neg(self) -> Self::Output {
        AutoDiff(ADNeg(self.0, PhantomData), PhantomData)
    }
}

/// Impl of Custom_Compose for CustomForwardDiff
impl<
    StaticArgsType,
    InnerInputType,
    InnerOutputType,
    InnerGradType,
    OuterInputType,
    OuterOutputType,
    OuterGradType,
    OutputGradType,
    Outer,
    Inner,
> func_traits::CustomCompose<
    AutoDiff<
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        Inner,
    >,
    OutputGradType,
> for AutoDiff<
    StaticArgsType,
    OuterInputType,
    OuterOutputType,
    OuterGradType,
    Outer,
>
where
    Outer: AutoDiffable<
        StaticArgsType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
    > + CustomForwardDiff<StaticArgsType, OuterInputType, OuterOutputType, OutputGradType, InnerGradType>,
    Inner: AutoDiffable<
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
    >,
    OuterInputType: From<InnerOutputType>,
    InnerOutputType: Clone,
    OuterOutputType: Clone,
{
    type Output = AutoDiff<
        StaticArgsType,
        InnerInputType,
        OuterOutputType,
        OutputGradType,
        ADCustomCompose<
        Outer,
        Inner,
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
        OutputGradType,
    >>;

    fn custom_compose(
        self,
        _other: AutoDiff<
            StaticArgsType,
            InnerInputType,
            InnerOutputType,
            InnerGradType,
            Inner,
        >,
    ) -> Self::Output {
        AutoDiff(ADCustomCompose(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Compose for AutoDiff
impl<
    StaticArgsType,
    InnerInputType,
    InnerOutputType,
    InnerGradType,
    OuterInputType,
    OuterOutputType,
    OuterGradType,
    Outer,
    Inner,
> func_traits::Compose<
    AutoDiff<
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        Inner,
    >,
> for AutoDiff<
    StaticArgsType,
    OuterInputType,
    OuterOutputType,
    OuterGradType,
    Outer,
>
where
    Outer: AutoDiffable<
        StaticArgsType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
    >,
    Inner: AutoDiffable<
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
    >,
    OuterInputType: From<InnerOutputType>,
    OuterGradType: ComposedGradMul<InnerInputType, OuterOutputType, InnerGradType>,
    InnerOutputType: Clone,
    OuterOutputType: Clone,
{
    type Output = AutoDiff<
        StaticArgsType,
        InnerInputType,
        OuterOutputType,
        <OuterGradType as ComposedGradMul<InnerInputType, OuterOutputType, InnerGradType>>::Output,
        ADCompose<
        Outer,
        Inner,
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
    >>;

    fn compose(
        self,
        _other: AutoDiff<
            StaticArgsType,
            InnerInputType,
            InnerOutputType,
            InnerGradType,
            Inner,
        >,
    ) -> Self::Output {
        AutoDiff(ADCompose(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl constant Pow for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, A, B> Pow<B>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
where
    OutputType: Clone + Pow<B>,
    GradType: Pow<B>,
    <OutputType as Pow<B>>::Output: Mul<B>,
    GradType: Mul<<<OutputType as Pow<B>>::Output as Mul<B>>::Output>,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType>,
    B: Clone + InstOne + Sub<B, Output = B>
{
    type Output = AutoDiff<StaticArgsType, InputType, <OutputType as Pow<B>>::Output, <GradType as Mul<<<OutputType as Pow<B>>::Output as Mul<B>>::Output>>::Output, ADConstantPow<A, B, OutputType, GradType>>;

    fn pow(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantPow(self.0, _other, PhantomData), PhantomData)
    }
}

/// Impl Abs
impl<StaticArgsType, InputType, OutputType, GradType, A> func_traits::Abs
for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
where
    OutputType: Signed,
    GradType: Signed + Mul<OutputType>,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, <GradType as Mul<OutputType>>::Output, ADAbs<A, GradType>>;
    fn abs(self) -> Self::Output {
        AutoDiff(ADAbs(self.0, PhantomData), PhantomData)
    }
}

/// Impl Signum
impl<StaticArgsType, InputType, OutputType, GradType, A> func_traits::Signum
for AutoDiff<StaticArgsType, InputType, OutputType, GradType, A>
where
    OutputType: InstZero + Signed,
    GradType: InstZero + UpperBounded,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, ADSignum<A>>;
    fn signum(self) -> Self::Output {
        AutoDiff(ADSignum(self.0), PhantomData)
    }
}



/*
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
