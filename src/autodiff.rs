use crate::adops::*;
use crate::autodiffable::AutoDiffable;
use crate::func_traits;
use num::traits::bounds::UpperBounded;
use num::traits::{Pow, Signed};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub, Deref};
use crate::traits::{InstZero, InstOne};

/// A wrapper type for an AutoDiffable type.
#[derive(Debug, Clone)]
pub struct AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, T>(
    pub T,
    pub PhantomData<(StaticArgsType, InputType, OutputType, GradType, GradInputType)>,
);

/// Impl Copy for AutoDiff if T is Copy and all other types are Clone
impl<StaticArgsType, InputType, OutputType, GradType, GradInputType, T> Copy
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, T>
where
    StaticArgsType: Clone,
    InputType: Clone,
    OutputType: Clone,
    GradType: Clone,
    GradInputType: Clone,
    T: Copy,
{
}

/// Impl of new for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, GradInputType, T>
    AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, T>
where
{
    #[allow(dead_code)]
    pub fn new(t: T) -> Self {
        AutoDiff(t, PhantomData)
    }
}

/// Impl of AutoDiffable for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, GradInputType, T>
    AutoDiffable<StaticArgsType, InputType, OutputType, GradType, GradInputType>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, T>
where
    T: AutoDiffable<StaticArgsType, InputType, OutputType, GradType, GradInputType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args)
    }

    fn eval_grad(&self, x: &InputType, dx: &GradInputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        self.0.eval_grad(x, dx, static_args)
    }
}

/// Impl of Deref for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, GradInputType, T> Deref
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, T>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Impl of Add for AutoDiff
impl<StaticArgsType, InputType, AOutputType, BOutputType, AGradType, GradInputType, BGradType, A, B>
    Add<AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, GradInputType, B>>
    for AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, GradInputType, A>
where
    AOutputType: Add<BOutputType>,
    AGradType: Add<BGradType>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType, GradInputType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType, GradInputType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, <AOutputType as Add<BOutputType>>::Output, <AGradType as Add<BGradType>>::Output, GradInputType, ADAdd<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn add(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, GradInputType, B>,
    ) -> Self::Output {
        AutoDiff(ADAdd(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Sub for AutoDiff
impl<StaticArgsType, InputType, AOutputType, BOutputType, AGradType, GradInputType, BGradType, A, B>
    Sub<AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, GradInputType, B>>
    for AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, GradInputType, A>
where
    AOutputType: Sub<BOutputType>,
    AGradType: Sub<BGradType>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType, GradInputType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType, GradInputType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, <AOutputType as Sub<BOutputType>>::Output, <AGradType as Sub<BGradType>>::Output, GradInputType, ADSub<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn sub(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, GradInputType, B>,
    ) -> Self::Output {
        AutoDiff(ADSub(self.0, _other.0, PhantomData), PhantomData)
    }
}


/// Impl of Mul for AutoDiff
impl<StaticArgsType, InputType, AOutputType, BOutputType, AGradType, GradInputType, BGradType, A, B>
    Mul<AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, GradInputType, B>>
    for AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, GradInputType, A>
where
    AOutputType: Mul<BOutputType> + Clone,
    BOutputType: Clone,
    AGradType: Mul<BOutputType>,
    BGradType: Mul<AOutputType>,
    <AGradType as Mul<BOutputType>>::Output: Add<<BGradType as Mul<AOutputType>>::Output>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType, GradInputType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType, GradInputType>,
{
    type Output = AutoDiff<
        StaticArgsType,
        InputType,
        <AOutputType as Mul<BOutputType>>::Output,
        <<AGradType as Mul<BOutputType>>::Output as Add<<BGradType as Mul<AOutputType>>::Output>>::Output,
        GradInputType
        , ADMul<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn mul(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, GradInputType, B>,
    ) -> Self::Output {
        AutoDiff(ADMul(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Div for AutoDiff
impl<StaticArgsType, InputType, AOutputType, BOutputType, AGradType, BGradType, GradInputType, A, B>
    Div<AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, GradInputType, B>>
    for AutoDiff<StaticArgsType, InputType, AOutputType, AGradType, GradInputType, A>
where
    AOutputType: Div<BOutputType> + Clone, // f/g
    BOutputType: Clone + Mul<BOutputType>, // g^2
    AGradType: Div<BOutputType>, // df/g
    BGradType: Mul<AOutputType>, // dg*f
    <BGradType as Mul<AOutputType>>::Output: Div<<BOutputType as Mul<BOutputType>>::Output>, // (dg*f)/g^2
      <AGradType as Div<BOutputType>>::Output: Sub< < <BGradType as Mul<AOutputType>>::Output as Div < <BOutputType as Mul<BOutputType>>::Output > >::Output >,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType, GradInputType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType, GradInputType>,
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
                <BOutputType as Mul<BOutputType>>::Output//g^2 |           |
              >                                        //      |           |
            >::Output                                  // -----+           |
          >                                            //                  |
        >::Output,                                      //------------------+
        GradInputType
        , ADDiv<A, B, AOutputType, AGradType, BOutputType, BGradType>>;

    fn div(
        self,
        _other: AutoDiff<StaticArgsType, InputType, BOutputType, BGradType, GradInputType, B>,
    ) -> Self::Output {
        AutoDiff(ADDiv(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl of Neg for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, GradInputType, A>
    Neg for AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, A>
where
    OutputType: Neg,
    GradType: Neg,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType, GradInputType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, <OutputType as Neg>::Output, <GradType as Neg>::Output, GradInputType
    , ADNeg<A, OutputType, GradType>>;

    fn neg(self) -> Self::Output {
        AutoDiff(ADNeg(self.0, PhantomData), PhantomData)
    }
}

/// Impl of Compose for AutoDiff
impl<
    StaticArgsType,
    InnerInputType,
    InnerOutputType,
    InnerGradType,
    InnerGradInputType,
    OuterInputType,
    OuterOutputType,
    OuterGradType,
    OuterGradInputType,
    Outer,
    Inner,
> func_traits::Compose<
    AutoDiff<
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        InnerGradInputType,
        Inner,
    >,
> for AutoDiff<
    StaticArgsType,
    OuterInputType,
    OuterOutputType,
    OuterGradType,
    OuterGradInputType,
    Outer,
>
where
    Outer: AutoDiffable<
        StaticArgsType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
        OuterGradInputType,
    >,
    Inner: AutoDiffable<
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        InnerGradInputType,
    >,
    OuterInputType: From<InnerOutputType>,
    OuterGradInputType: From<InnerGradType>,
{
    type Output = AutoDiff<
        StaticArgsType,
        InnerInputType,
        OuterOutputType,
        OuterGradType,
        InnerGradInputType,
        ADCompose<
        Outer,
        Inner,
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        InnerGradInputType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
        OuterGradInputType,
    >>;

    fn compose(
        self,
        _other: AutoDiff<
            StaticArgsType,
            InnerInputType,
            InnerOutputType,
            InnerGradType,
            InnerGradInputType,
            Inner,
        >,
    ) -> Self::Output {
        AutoDiff(ADCompose(self.0, _other.0, PhantomData), PhantomData)
    }
}

/// Impl constant Pow for AutoDiff
impl<StaticArgsType, InputType, OutputType, GradType, GradInputType, A, B> Pow<B>
    for AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, A>
where
    OutputType: Clone + Pow<B>,
    GradType: Pow<B>,
    <OutputType as Pow<B>>::Output: Mul<B>,
    GradType: Mul<<<OutputType as Pow<B>>::Output as Mul<B>>::Output>,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType, GradInputType>,
    B: Clone + InstOne + Sub<B, Output = B>
{
    type Output = AutoDiff<StaticArgsType, InputType, <OutputType as Pow<B>>::Output, <GradType as Mul<<<OutputType as Pow<B>>::Output as Mul<B>>::Output>>::Output, GradInputType, ADConstantPow<A, B, OutputType, GradType>>;

    fn pow(self, _other: B) -> Self::Output {
        AutoDiff(ADConstantPow(self.0, _other, PhantomData), PhantomData)
    }
}

/// Impl Abs
impl<StaticArgsType, InputType, OutputType, GradType, GradInputType, A> func_traits::Abs
for AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, A>
where
    OutputType: Signed,
    GradType: Signed + Mul<OutputType>,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType, GradInputType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, <GradType as Mul<OutputType>>::Output, GradInputType, ADAbs<A, GradType>>;
    fn abs(self) -> Self::Output {
        AutoDiff(ADAbs(self.0, PhantomData), PhantomData)
    }
}

/// Impl Signum
impl<StaticArgsType, InputType, OutputType, GradType, GradInputType, A> func_traits::Signum
for AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, A>
where
    OutputType: InstZero + Signed,
    GradType: InstZero + UpperBounded,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType, GradInputType>,
{
    type Output = AutoDiff<StaticArgsType, InputType, OutputType, GradType, GradInputType, ADSignum<A>>;
    fn signum(self) -> Self::Output {
        AutoDiff(ADSignum(self.0), PhantomData)
    }
}
