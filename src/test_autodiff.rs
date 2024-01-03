use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use crate::func_traits::*;
use crate::funcs::*;
use num::traits::Pow;
use std::ops::Deref;

#[test]
fn test_all_ops() {
    // test all supported unary operations on p(x) = 1 + 2x + 3x^2
    // test all supported binary operations on p(x) and q(x) = x^5

    let x = 2.0_f64;
    let dx = 1.0_f64;

    let p = AutoDiff::new(Polynomial::new(vec![1.0, 2.0, 3.0]));
    let q = AutoDiff::new(Monomial::<f64, f64>::new(5.0));

    let (p_x, dp_x): (f64, f64) = p.eval_grad(&x, &dx, &());
    let (q_x, dq_x): (f64, f64) = q.eval_grad(&x, &dx, &());

    // unary ops
    // neg
    assert_eq!((-p_x, -dp_x), (-p.clone()).eval_grad(&x, &dx, &()));
    // abs
    assert_eq!(
        (p_x.abs(), dp_x * p_x.signum()),
        p.clone().abs().eval_grad(&x, &dx, &())
    );
    // signum
    assert_eq!(
        (p_x.signum(), 0.0),
        p.clone().signum().eval_grad(&x, &dx, &())
    );

    // binary ops with constants
    let c = 2.0_f64;
    assert_eq!((p_x + c, dp_x), (p.clone() + c).eval_grad(&x, &dx, &()));
    assert_eq!((p_x - c, dp_x), (p.clone() - c).eval_grad(&x, &dx, &()));
    assert_eq!((p_x * c, dp_x * c), (p.clone() * c).eval_grad(&x, &dx, &()));
    assert_eq!((p_x / c, dp_x / c), (p.clone() / c).eval_grad(&x, &dx, &()));
    assert_eq!(
        (p_x.pow(c), c * p_x * dp_x),
        (p.clone().pow(c)).eval_grad(&x, &dx, &())
    );

    // binary ops with other functions
    assert_eq!(
        (p_x + q_x, dp_x + dq_x),
        (p.clone() + q.clone()).eval_grad(&x, &dx, &())
    );
    assert_eq!(
        (p_x - q_x, dp_x - dq_x),
        (p.clone() - q.clone()).eval_grad(&x, &dx, &())
    );
    assert_eq!(
        (p_x * q_x, p_x * dq_x + q_x * dp_x),
        (p.clone() * q.clone()).eval_grad(&x, &dx, &())
    );
    assert_eq!(
        (p_x / q_x, (dp_x * q_x - p_x * dq_x) / q_x.pow(2.0)),
        (p.clone() / q.clone()).eval_grad(&x, &dx, &())
    );

    // result for composition
    let (q_of_p_x, dq_of_p_x) = q.eval_grad(&p_x, &dp_x, &());
    let (p_of_q_x, dp_of_q_x) = p.eval_grad(&q_x, &dq_x, &());

    assert_eq!(
        (q_of_p_x, dq_of_p_x),
        q.clone().compose(p.clone()).eval_grad(&x, &dx, &())
    );
    assert_eq!(
        (p_of_q_x, dp_of_q_x),
        p.clone().compose(q.clone()).eval_grad(&x, &dx, &())
    );

    // test custom wrapper type with Deref
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Wrapper<T>(T);

    impl<T> Deref for Wrapper<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    let w = Wrapper(2.0_f64);
    let dw = Wrapper(1.0_f64);

    let (p_w, dp_w) = p.eval_grad(&w, &dw, &());

    assert_eq!((p_w, dp_w), (p_x, dp_x));
}
