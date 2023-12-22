use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use crate::func_traits::*;
use crate::funcs::*;
use num::traits::Pow;

#[test]
fn test_all_ops() {
    // test all supported unary operations on p(x) = 1 + 2x + 3x^2
    // test all supported binary operations on p(x) and q(x) = x^5

    let x = 2.0_f64;

    let p = AutoDiff::new(Polynomial::new(vec![1.0, 2.0, 3.0]));
    let q = AutoDiff::new(Monomial::new(5.0));

    let (p_x, dp_x) = p.eval_grad(&x, &());
    let (q_x, dq_x) = q.eval_grad(&x, &());

    // unary ops
    // neg
    assert_eq!((-p_x, -dp_x), (-p.clone()).eval_grad(&x, &()));
    // abs
    assert_eq!(
        (p_x.abs(), dp_x * p_x.signum()),
        p.clone().abs().eval_grad(&x, &())
    );
    // signum
    assert_eq!((p_x.signum(), 0.0), p.clone().signum().eval_grad(&x, &()));

    // binary ops with constants
    assert_eq!((p_x + 1.0, dp_x), (p.clone() + 1.0).eval_grad(&x, &()));
    assert_eq!((p_x - 1.0, dp_x), (p.clone() - 1.0).eval_grad(&x, &()));
    assert_eq!(
        (p_x * 2.0, dp_x * 2.0),
        (p.clone() * 2.0).eval_grad(&x, &())
    );
    assert_eq!(
        (p_x / 2.0, dp_x / 2.0),
        (p.clone() / 2.0).eval_grad(&x, &())
    );
    assert_eq!(
        (p_x.pow(2.0), 2.0 * p_x * dp_x),
        (p.clone().pow(2.0)).eval_grad(&x, &())
    );

    // binary ops with other functions
    assert_eq!(
        (p_x + q_x, dp_x + dq_x),
        (p.clone() + q.clone()).eval_grad(&x, &())
    );
    assert_eq!(
        (p_x - q_x, dp_x - dq_x),
        (p.clone() - q.clone()).eval_grad(&x, &())
    );
    assert_eq!(
        (p_x * q_x, p_x * dq_x + q_x * dp_x),
        (p.clone() * q.clone()).eval_grad(&x, &())
    );
    assert_eq!(
        (p_x / q_x, (dp_x * q_x - p_x * dq_x) / q_x.pow(2.0)),
        (p.clone() / q.clone()).eval_grad(&x, &())
    );

    // result for composition
    let (q_of_p_x, dq_of_p_x) = q.eval_grad(&p_x, &());
    let (p_of_q_x, dp_of_q_x) = p.eval_grad(&q_x, &());

    assert_eq!(
        (q_of_p_x, dq_of_p_x * dp_x),
        q.clone().compose(p.clone()).eval_grad(&x, &())
    );
    assert_eq!(
        (p_of_q_x, dp_of_q_x * dq_x),
        p.clone().compose(q.clone()).eval_grad(&x, &())
    );
}
