use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use crate::func_traits::Compose;
use crate::funcs::*;
use std::ops::Add;

#[test]
fn test_manual() {
    // define a function which takes in a tuple and outputs a f64
    // manually define addition and composition for this function

    // newtype for (f64, f64)
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct F(f64, f64);

    // define a function which takes in an F and outputs a f64

    #[derive(Debug, Clone, Copy)]
    struct Swap;

    impl Swap {
        fn new() -> Self {
            Swap
        }
    }

    impl AutoDiffable<(), F, F, (F, F)> for Swap {
        fn eval(&self, x: &F, _: &()) -> F {
            F(x.1, x.0)
        }
        fn eval_grad(&self, x: &F, _: &()) -> (F, (F, F)) {
            (self.eval(x, &()), (F(0.0, 1.0), F(1.0, 0.0)))
        }
    }

    // manually define addition for Swap + Swap
    #[derive(Debug, Clone, Copy)]
    struct AddSwap(Swap, Swap);

    impl AutoDiffable<(), F, F, (F, F)> for AddSwap {
        fn eval(&self, x: &F, _: &()) -> F {
            let f0 = self.0.eval(x, &());
            let f1 = self.1.eval(x, &());
            F(f0.0 + f1.0, f0.1 + f1.1)
        }
        fn eval_grad(&self, x: &F, _: &()) -> (F, (F, F)) {
            let (f0, (f0x, f0y)) = self.0.eval_grad(x, &());
            let (f1, (f1x, f1y)) = self.1.eval_grad(x, &());
            (
                F(f0.0 + f1.0, f0.1 + f1.1),
                (
                    F(f0x.0 + f1x.0, f0x.1 + f1x.1),
                    F(f0y.0 + f1y.0, f0y.1 + f1y.1),
                ),
            )
        }
    }

    impl Add for Swap {
        type Output = AutoDiff<(), F, F, (F, F), AddSwap>;
        fn add(self, rhs: Swap) -> Self::Output {
            AutoDiff::new(AddSwap(self, rhs))
        }
    }

    // manuall define composition for Monomial(Swap)
    // first we have to manuall implement AutoDiffable<(), F, F, (F, F,)> for Monomial
    impl AutoDiffable<(), F, F, (F, F)> for Monomial<F, f64> {
        fn eval(&self, x: &F, _: &()) -> F {
            F(x.0.powf(self.0), x.1.powf(self.0))
        }
        fn eval_grad(&self, x: &F, _: &()) -> (F, (F, F)) {
            let (f0, f1) = (x.0.powf(self.0), x.1.powf(self.0));
            (
                F(f0, f1),
                (F(self.0 * f0 / x.0, 0.0), F(0.0, self.0 * f1 / x.1)),
            )
        }
    }

    struct ComposeMonomialSwap(Monomial<F, f64>, Swap);

    impl AutoDiffable<(), F, F, (F, F)> for ComposeMonomialSwap {
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

    impl Compose<Swap> for Monomial<F, f64> {
        type Output = AutoDiff<(), F, F, (F, F), ComposeMonomialSwap>;
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
