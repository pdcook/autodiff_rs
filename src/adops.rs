use crate::autodiffable::{AutoDiffable, CustomForwardDiff};
use num::traits::bounds::UpperBounded;
use num::traits::{Signed, Pow};
use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::traits::{InstZero, InstOne, ComposedGradMul};

#[derive(Debug, Clone, Copy)]
pub struct ADAdd<A, B, AOutputType, AGradType, BOutputType, BGradType>(pub A, pub B, pub PhantomData<( AOutputType, AGradType, BOutputType, BGradType,)>);

impl<StaticArgsType,
    InputType,
    AOutputType,
    BOutputType,
    AGradType,
    BGradType,
    A,
    B>
    AutoDiffable<StaticArgsType, InputType, <AOutputType as Add<BOutputType>>::Output, <AGradType as Add<BGradType>>::Output> for ADAdd<A, B, AOutputType, AGradType, BOutputType, BGradType>
where
    AOutputType: Add<BOutputType>,
    AGradType: Add<BGradType>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> <AOutputType as Add<BOutputType>>::Output {
        // use .add instead of + to allow for newtypes which implement Deref
        self.0.eval(x, static_args).add(self.1.eval(x, static_args))
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (<AOutputType as Add<BOutputType>>::Output, <AGradType as Add<BGradType>>::Output) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        (f.add(g), df.add(dg))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSub<A, B, AOutputType, AGradType, BOutputType, BGradType>(pub A, pub B, pub PhantomData<( AOutputType, AGradType, BOutputType, BGradType,)>);

impl<StaticArgsType,
    InputType,
    AOutputType,
    BOutputType,
    AGradType,
    BGradType,
    A,
    B>
    AutoDiffable<StaticArgsType, InputType, <AOutputType as Sub<BOutputType>>::Output, <AGradType as Sub<BGradType>>::Output> for ADSub<A, B, AOutputType, AGradType, BOutputType, BGradType>
where
    AOutputType: Sub<BOutputType>,
    AGradType: Sub<BGradType>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> <AOutputType as Sub<BOutputType>>::Output {
        // use .sub instead of - to allow for newtypes which implement Deref
        self.0.eval(x, static_args).sub(self.1.eval(x, static_args))
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (<AOutputType as Sub<BOutputType>>::Output, <AGradType as Sub<BGradType>>::Output) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        (f.sub(g), df.sub(dg))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADMul<A, B, AOutputType, AGradType, BOutputType, BGradType>(pub A, pub B, pub PhantomData<( AOutputType, AGradType, BOutputType, BGradType,)>);

impl<StaticArgsType,
    InputType,
    AOutputType,
    BOutputType,
    AGradType,
    BGradType,
    A,
    B>
    AutoDiffable<
        StaticArgsType,
        InputType,
        <AOutputType as Mul<BOutputType>>::Output,
        <<AGradType as Mul<BOutputType>>::Output as Add<<BGradType as Mul<AOutputType>>::Output>>::Output
        >
        for ADMul<A, B, AOutputType, AGradType, BOutputType, BGradType>
where
    AOutputType: Mul<BOutputType> + Clone,
    BOutputType: Clone,
    AGradType: Mul<BOutputType>,
    BGradType: Mul<AOutputType>,
    <AGradType as Mul<BOutputType>>::Output: Add<<BGradType as Mul<AOutputType>>::Output>,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType>,
    B: AutoDiffable<StaticArgsType, InputType, BOutputType, BGradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> <AOutputType as Mul<BOutputType>>::Output {
        // use .mul instead of * to allow for newtypes which implement Deref
        self.0.eval(x, static_args).mul(self.1.eval(x, static_args))
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (<AOutputType as Mul<BOutputType>>::Output, <<AGradType as Mul<BOutputType>>::Output as Add<<BGradType as Mul<AOutputType>>::Output>>::Output) {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        // f * g : AOutputType: Mul<BOutputType>
        //
        // df * g : AGradType: Mul<BOutputType>
        // dg * f : BGradType: Mul<AOutputType>
        // df * g + dg * f : <AGradType as Mul<BOutputType>>::Output: Add<<BGradType as Mul<AOutputType>>::Output>
        //

        (f.clone().mul(g.clone()), df.mul(g).add(dg.mul(f)))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADDiv<A, B, AOutputType, AGradType, BOutputType, BGradType>(pub A, pub B, pub PhantomData<( AOutputType, AGradType, BOutputType, BGradType,)>);

impl<StaticArgsType,
    InputType,
    AOutputType,
    BOutputType,
    AGradType,
    BGradType,
    A,
    B>
    AutoDiffable<
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
    >
        for ADDiv<A, B, AOutputType, AGradType, BOutputType, BGradType>
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
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> <AOutputType as Div<BOutputType>>::Output {
        // use .div instead of / to allow for newtypes which implement Deref
        self.0.eval(x, static_args).div(self.1.eval(x, static_args))
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (<AOutputType as Div<BOutputType>>::Output,
      <<AGradType as Div<BOutputType>>::Output as Sub< < <BGradType as Mul<AOutputType>>::Output as Div < <BOutputType as Mul<BOutputType>>::Output > >::Output >>::Output,
    )

    {
        let (f, df) = self.0.eval_grad(x, static_args);
        let (g, dg) = self.1.eval_grad(x, static_args);

        // d(f/g) = (df*g - f*dg)/g^2 = df/g - f*dg/g^2
        // = (df/g - (dg*f)/(g*g))

        (f.clone().div(g.clone()),
            df.div(g.clone()).sub(dg.mul(f).div(g.clone().mul(g))))

    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADNeg<A, AOutputType, AGradType>(pub A, pub PhantomData<(AOutputType, AGradType)>);

impl<StaticArgsType, InputType, AOutputType, AGradType, A>
    AutoDiffable<StaticArgsType, InputType, <AOutputType as Neg>::Output, <AGradType as Neg>::Output>
    for ADNeg<A, AOutputType, AGradType>
where
    AOutputType: Neg,
    AGradType: Neg,
    A: AutoDiffable<StaticArgsType, InputType, AOutputType, AGradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> <AOutputType as Neg>::Output {
        // use .neg instead of - to allow for newtypes which implement Deref
        self.0.eval(x, static_args).neg()
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (<AOutputType as Neg>::Output, <AGradType as Neg>::Output) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.neg(), df.neg())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADCompose<
    Outer,
    Inner,
    StaticArgsType,
    InnerInputType,
    InnerOutputType,
    InnerGradType,
    OuterInputType,
    OuterOutputType,
    OuterGradType,
>(
    pub Outer,
    pub Inner,
    pub PhantomData<(
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
    )>,
);

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
    >
    AutoDiffable<
        StaticArgsType,
        InnerInputType,
        OuterOutputType,
        <OuterGradType as ComposedGradMul<InnerInputType, OuterOutputType, InnerGradType>>::Output,
    >
    for ADCompose<
        Outer,
        Inner,
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
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
    fn eval(
        &self,
        x: &InnerInputType,
        static_args: &StaticArgsType,
    ) -> OuterOutputType {
        self.0.eval(&self.1.eval(x, static_args).into(), static_args)
    }

    fn eval_grad(
        &self,
        x: &InnerInputType,
        static_args: &StaticArgsType,
    ) -> (OuterOutputType, <OuterGradType as ComposedGradMul<InnerInputType, OuterOutputType, InnerGradType>>::Output) {

        let (g, dg) = self.1.eval_grad(x, static_args);
        let (f_of_g, df_of_g) = self.0.eval_grad(&g.into(), static_args);
        // chain rule
        // (f(g))' = f'(g) * g'
        // mul here has to be compose_mul, not the usual mul
        // since this type of multiplication may be different from the usual,
        // specifically for things like arrays in which case
        // compose_mul = mul followed by sum over all trailing dimensions
        (f_of_g.clone(), df_of_g.compose_mul(x, &f_of_g, &dg))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADCustomCompose<
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
>(
    pub Outer,
    pub Inner,
    pub PhantomData<(
        StaticArgsType,
        InnerInputType,
        InnerOutputType,
        InnerGradType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
        OutputGradType,
    )>,
);

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
    >
    AutoDiffable<
        StaticArgsType,
        InnerInputType,
        OuterOutputType,
        OutputGradType,
    >
    for ADCustomCompose<
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
    >
where
    Outer: AutoDiffable<
        StaticArgsType,
        OuterInputType,
        OuterOutputType,
        OuterGradType,
    > + CustomForwardDiff<StaticArgsType, OuterInputType, OuterOutputType, OutputGradType, InnerGradType>
    ,
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
    fn eval(
        &self,
        x: &InnerInputType,
        static_args: &StaticArgsType,
    ) -> OuterOutputType {
        self.0.eval(&self.1.eval(x, static_args).into(), static_args)
    }

    fn eval_grad(
        &self,
        x: &InnerInputType,
        static_args: &StaticArgsType,
    ) -> (OuterOutputType, OutputGradType)
    {
        let (g, dg) = self.1.eval_grad(x, static_args);
        self.0.forward_eval_grad(&g.clone().into(), Some(&dg), static_args)
    }
}


#[derive(Debug, Clone, Copy)]
pub struct ADConstantPow<A, B, OutputType, GradType>(pub A, pub B, pub PhantomData<(OutputType, GradType)>);

impl<StaticArgsType, InputType, OutputType, GradType, A, B>
    AutoDiffable<StaticArgsType, InputType, <OutputType as Pow<B>>::Output,
    <GradType as Mul<<<OutputType as Pow<B>>::Output as Mul<B>>::Output>>::Output
    > for ADConstantPow<A, B, OutputType, GradType>
where
    OutputType: Clone + Pow<B>,
    GradType: Pow<B>,
    <OutputType as Pow<B>>::Output: Mul<B>,
    GradType: Mul<<<OutputType as Pow<B>>::Output as Mul<B>>::Output>,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType>,
    B: Clone + InstOne + Sub<B, Output = B>
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> <OutputType as Pow<B>>::Output {
        self.0.eval(x, static_args).pow(self.1.clone())
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (<OutputType as Pow<B>>::Output
    , <GradType as Mul<<<OutputType as Pow<B>>::Output as Mul<B>>::Output>>::Output) {
        let (f, df) = self.0.eval_grad(x, static_args);

        // d(f^p) = p * f^(p-1) * df
        // = df * ((f^(p-1)) * p)

        (f.clone().pow(self.1.clone()), (df.mul(f.pow(self.1.clone().sub(self.1.one())).mul(self.1.clone()))))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADAbs<A, GradType>(pub A, pub PhantomData<GradType>);

impl<StaticArgsType, InputType, OutputType, GradType, A>
    AutoDiffable<StaticArgsType, InputType, OutputType, <GradType as Mul<OutputType>>::Output> for ADAbs<A, GradType>
where
    OutputType: Signed,
    GradType: Signed + Mul<OutputType>,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).abs()
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, <GradType as Mul<OutputType>>::Output) {
        let (f, df) = self.0.eval_grad(x, static_args);

        (f.abs(), df.mul(f.signum()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ADSignum<A>(pub A);

impl<StaticArgsType, InputType, OutputType, GradType, A>
    AutoDiffable<StaticArgsType, InputType, OutputType, GradType> for ADSignum<A>
where
    OutputType: InstZero + Signed,
    GradType: InstZero + UpperBounded,
    A: AutoDiffable<StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args).signum()
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        // chain rule on signum, (sign(f(x)))' = 2 delta(f(x))
        // we approximate delta(x) as
        // delta(x) = GradType::MAX if x == 0, 0 otherwise

        let (f, df) = self.0.eval_grad(x, static_args);

        if InstZero::is_zero(&f) {
            (f.signum(), GradType::max_value())
        } else {
            (f.signum(), df.zero())
        }
    }
}
