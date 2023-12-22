use crate::diffable::Diffable;
use crate::autodiff::AutoDiff;
use crate::func_traits::{Compose};
use crate::funcs::*;
use num::traits::Pow;
use std::ops::Add;

#[test]
fn test_manual() {
    // define a function which takes in a tuple and outputs a f64
    // manually define addition and composition for this function

    // newtype for (f64, f64)
    #[derive(Debug, Clone, Copy)]
    struct F(f64, f64);

    // define a function which takes in an F and outputs a f64

    #[derive(Debug, Clone, Copy)]
    struct Func;

    impl Func {
        fn new() -> Self {
            Func
        }
    }

    impl Diffable<(), F, f64, F> for Func {
        fn eval(&self, x: &F, _: &()) -> f64 {
            x.0.pow(2.0) + x.1.pow(3.0)
        }
        fn eval_grad(&self, x: &F, _: &()) -> (f64, F) {
            (self.eval(x, &()), F(2.0 * x.0, 3.0 * x.1.pow(2.0)))
        }
    }


    ////
    eigh(x) -> (ms, vs)
    deigh(x) -> ((ms, vs), (dms, dvs))

    loss(x) = loss(eigh(x).0)
    dloss(x) = deigh(x).0 * dloss(eigh(x).0)

}
