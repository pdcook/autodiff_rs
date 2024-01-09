use crate::autodiff::AutoDiff;
use crate::autodiffable::*;
use crate::func_traits::Compose;
use crate::funcs::*;
use std::ops::Add;
use crate::gradienttype::GradientType;
use crate::forward::ForwardMul;

use crate as autodiff;
use forwarddiffable_derive::SimpleForwardDiffable;

#[test]
fn test_manual() {
    // newtype for (f64, f64)
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct F(f64, f64);

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct G(F, F);

    impl GradientType<F> for F {
        type GradientType = G;
    }

    /// implementation of ForwardMul for
    /// df/dx * dx -> df
    /// x: F - SelfInput
    /// f: F - SelfOutput
    /// df/dx: (F, F) - Self
    /// dx: F - OtherGrad
    /// df: F - ResultGrad
    impl ForwardMul<F, F> for G {
        type ResultGrad = F;
        fn forward_mul(&self, dx: &F) -> F {
            // define
            // df/dx * dx -> df
            // where df/dx is (F, F), and dx is F
            // this is just the dot product between self and dx

            let G(dfdx0, dfdx1) = self;
            let F(dfdx00, dfdx01) = dfdx0;
            let F(dfdx10, dfdx11) = dfdx1;
            let F(dx0, dx1) = dx;

            F(dfdx00 * dx0 + dfdx10 * dx1, dfdx01 * dx0 + dfdx11 * dx1)
        }
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
        fn eval_grad(&self, x: &F, _: &()) -> (F, G) {
            (self.eval(x, &()), G(F(0.0, 1.0), F(1.0, 0.0)))
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
        fn eval_grad(&self, x: &F, _: &()) -> (F, G) {
            let (f0, df0) = self.0.eval_grad(x, &());
            let (f1, df1) = self.1.eval_grad(x, &());

            (F(f0.0 + f1.0, f0.1 + f1.1),
                G(F(df0.0.0 + df1.0.0, df0.0.1 + df1.0.1),
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
        fn eval_grad(&self, x: &F, _: &()) -> (F, G) {
            let (f0, f1) = (x.0.powf(self.0), x.1.powf(self.0));
            (
                F(f0, f1),
                G(F(self.0 * f0 / x.0, 0.0),
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
        fn eval_grad(&self, x: &F, _: &()) -> (F, G) {
            let (f, G(fx, fy)) = self.1.eval_grad(x, &());
            let (fg, G(fgx, fgy)) = self.0.eval_grad(&f, &());
            let (fxx, fxy) = (fx.0, fx.1);
            let (fyx, fyy) = (fy.0, fy.1);
            let (fgxx, fgxy) = (fgx.0, fgx.1);
            let (fgyx, fgyy) = (fgy.0, fgy.1);
            // chain rule, elementwise
            let grad = G(
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

    let m = AutoDiff::<(), Monomial<(), F, f64>>::new(Monomial::new(2.0));

    // f3(x,y) = x^2 o (y,x) = (y^2, x^2)
    let f3 = AutoDiff::new((*m).compose(*f));

    let x = F(1.0, 2.0);

    assert_eq!(f.eval(&x, &()), F(2.0, 1.0));
    assert_eq!(f.grad(&x, &()), G(F(0.0, 1.0), F(1.0, 0.0)));
    assert_eq!(f2.eval(&x, &()), F(4.0, 2.0));
    assert_eq!(f2.grad(&x, &()), G(F(0.0, 2.0), F(2.0, 0.0)));
    assert_eq!(f3.eval(&x, &()), F(4.0, 1.0));
    assert_eq!(f3.grad(&x, &()), G(F(0.0, 4.0), F(2.0, 0.0)));

    let dx = F(1.0, 1.0);

    let df1 = f.forward_grad(&x, &dx, &());
    let df2 = f2.forward_grad(&x, &dx, &());
    let df3 = f3.forward_grad(&x, &dx, &());

    assert_eq!(df1, F(1.0, 1.0));
    assert_eq!(df2, F(2.0, 2.0));
    assert_eq!(df3, F(2.0, 4.0));

    // this shouldn't compile
    // TODO: use the compiletest_rs crate to ensure that this doesn't compile
    // let f4 = *f / *f;
}
