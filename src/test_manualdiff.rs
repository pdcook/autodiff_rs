use crate::autodiff::AutoDiff;
use crate::autodiffable::{AutoDiffable, ForwardDiffable};
use crate::func_traits::Compose;
use crate::funcs::*;
use std::ops::Add;
use crate::gradienttype::GradientType;

use crate as autodiff;
use forwarddiffable_derive::SimpleForwardDiffable;

#[test]
fn test_manual() {
    // newtype for (f64, f64)
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct F(f64, f64);

    impl GradientType<F> for F {
        type GradientType = (F, F);
    }

    #[derive(Debug, Clone, Copy, SimpleForwardDiffable)]
    struct Swap;

    impl Swap {
        fn new() -> Self {
            Swap
        }
    }

    impl AutoDiffable<()> for Swap {
        type Input = F;
        type Output = F;
        fn eval(&self, x: &F, _: &()) -> F {
            F(x.1, x.0)
        }
        fn eval_grad(&self, x: &F, _: &()) -> (F, (F,F)) {
            (self.eval(x, &()), (F(0.0, 1.0), F(1.0, 0.0)))
        }
    }

    // manually define addition for Swap + Swap
    #[derive(Debug, Clone, Copy, SimpleForwardDiffable)]
    struct AddSwap(Swap, Swap);

    impl AutoDiffable<()> for AddSwap {
        type Input = F;
        type Output = F;
        fn eval(&self, x: &F, _: &()) -> F {
            let f0 = self.0.eval(x, &());
            let f1 = self.1.eval(x, &());
            F(f0.0 + f1.0, f0.1 + f1.1)
        }
        fn eval_grad(&self, x: &F, _: &()) -> (F, (F,F)) {
            let (f0, df0) = self.0.eval_grad(x, &());
            let (f1, df1) = self.1.eval_grad(x, &());

            (F(f0.0 + f1.0, f0.1 + f1.1),
                (F(df0.0.0 + df1.0.0, df0.0.1 + df1.0.1),
                 F(df0.1.0 + df1.1.0, df0.1.1 + df1.1.1)))

        }
    }

    impl Add for Swap {
        type Output = AutoDiff<(), AddSwap>;
        fn add(self, rhs: Swap) -> Self::Output {
            AutoDiff::new(AddSwap(self, rhs))
        }
    }

    // manually define composition for Monomial(Swap)
    // first we have to manually implement
    // AutoDiffable<()> for Monomial
    impl AutoDiffable<()> for Monomial<(), F, f64> {
        type Input = F;
        type Output = F;
        fn eval(&self, x: &F, _: &()) -> F {
            F(x.0.powf(self.0), x.1.powf(self.0))
        }
        fn eval_grad(&self, x: &F, _: &()) -> (F, (F, F)) {
            let (f0, f1) = (x.0.powf(self.0), x.1.powf(self.0));
            (
                F(f0, f1),
                (F(self.0 * f0 / x.0, 0.0),
                 F(0.0, self.0 * f1 / x.1),)
            )
        }
    }

    #[derive(Debug, Clone, Copy, SimpleForwardDiffable)]
    struct ComposeMonomialSwap(Monomial<(), F, f64>, Swap);

    impl AutoDiffable<()> for ComposeMonomialSwap {
        type Input = F;
        type Output = F;
        fn eval(&self, x: &F, _: &()) -> F {
            self.0.eval(&self.1.eval(x, &()), &())
        }
        fn eval_grad(&self, x: &F, _: &()) -> (F, (F, F)) {
            let (f, (fx, fy)) = self.1.eval_grad(x, &());
            let (fg, (fgx, fgy)) = self.0.eval_grad(&f, &());
            let (fxx, fxy) = (fx.0, fx.1);
            let (fyx, fyy) = (fy.0, fy.1);
            let (fgxx, fgxy) = (fgx.0, fgx.1);
            let (fgyx, fgyy) = (fgy.0, fgy.1);
            // chain rule, elementwise
            let grad = (
                F(fgxx * fxx + fgxy * fyx, fgxx * fxy + fgxy * fyy),
                F(fgyx * fxx + fgyy * fyx, fgyx * fxy + fgyy * fyy),
            );
            (fg, grad)
        }
    }

    impl Compose<Swap> for Monomial<(), F, f64> {
        type Output = AutoDiff<(), ComposeMonomialSwap>;
        fn compose(self, rhs: Swap) -> Self::Output {
            AutoDiff::new(ComposeMonomialSwap(self, rhs))
        }
    }

    // manual operations have to be done via Deref (i.e. a+b doesn't work, but a.add(*b) does as
    // well as *a + *b)

    // f(x,y) = (y,x)
    let f = AutoDiff::new(Swap::new());

    // this doesn't compile
    // let f2 = f + f;
    // this does
    // let f2 = f.add(*f);
    // this does
    // f2(x,y) = (y,x) + (y,x) = (2y, 2x)
    let f2 = *f + *f;

    let m = Monomial::new(2.0);

    // f3(x,y) = x^2 o (y,x) = (y^2, x^2)
    let f3 = m.compose(*f);

    let x = F(1.0, 2.0);

    assert_eq!(f.eval(&x, &()), F(2.0, 1.0));
    assert_eq!(f2.eval(&x, &()), F(4.0, 2.0));
    assert_eq!(f3.eval(&x, &()), F(4.0, 1.0));

    // this shouldn't compile
    // TODO: use the compiletest_rs crate to ensure that this doesn't compile
    // let f4 = *f / *f;
}
