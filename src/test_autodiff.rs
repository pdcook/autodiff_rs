use crate::autodiff::AutoDiff;
use crate::autodiffable::*;
use crate::func_traits::*;
use crate::funcs::*;
use num::traits::Pow;
use std::ops::Deref;
use crate::forward::ForwardMul;

#[test]
fn test_all_ops() {
    // test all supported unary operations on p(x) = 1 + 2x + 3x^2
    // test all supported binary operations on p(x) and q(x) = x^5

    let x = 2.0_f64;
    let dx = 1.0_f64;

    let p = AutoDiff::new(Polynomial::new(vec![1.0, 2.0, 3.0]));
    let q = AutoDiff::new(Monomial::<(), f64, f64>::new(5.0));

    let (p_x, dp_x): (f64, f64) = p.eval_forward_grad(&x, &dx, &());
    let dp_dx: f64 = p.grad(&x, &());
    let (q_x, dq_x): (f64, f64) = q.eval_forward_grad(&x, &dx, &());
    let dq_dx: f64 = q.grad(&x, &());

    // unary ops
    // neg
    assert_eq!((-p_x, -dp_x), (-p.clone()).eval_forward_grad(&x, &dx, &()));
    assert_eq!(-dp_dx, (-p.clone()).grad(&x, &()));
    // abs
    assert_eq!(
        (p_x.abs(), dp_x * p_x.signum()),
        p.clone().abs().eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(dp_dx * p_x.signum(), p.clone().abs().grad(&x, &()));
    // signum
    assert_eq!(
        (p_x.signum(), 0.0),
        p.clone().signum().eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(0.0, p.clone().signum().grad(&x, &()));

    // binary ops with constants
    let c = 3.0_f64;
    assert_eq!((p_x + c, dp_x), (p.clone() + c).eval_forward_grad(&x, &dx, &()));
    assert_eq!(dp_dx, (p.clone() + c).grad(&x, &()));
    assert_eq!((p_x - c, dp_x), (p.clone() - c).eval_forward_grad(&x, &dx, &()));
    assert_eq!(dp_dx, (p.clone() - c).grad(&x, &()));
    assert_eq!((p_x * c, dp_x * c), (p.clone() * c).eval_forward_grad(&x, &dx, &()));
    assert_eq!(dp_dx * c, (p.clone() * c).grad(&x, &()));
    assert_eq!((p_x / c, dp_x / c), (p.clone() / c).eval_forward_grad(&x, &dx, &()));
    assert_eq!(dp_dx / c, (p.clone() / c).grad(&x, &()));
    assert_eq!(
        (p_x.pow(c), c * p_x.pow(c-1.0) * dp_x),
        (p.clone().pow(c)).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(c * p_x.pow(c-1.0) * dp_dx, (p.clone().pow(c)).grad(&x, &()));

    // binary ops with other functions
    assert_eq!(
        (p_x + q_x, dp_x + dq_x),
        (p.clone() + q.clone()).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(
        dp_dx + dq_dx,
        (p.clone() + q.clone()).grad(&x, &())
    );
    assert_eq!(
        (p_x - q_x, dp_x - dq_x),
        (p.clone() - q.clone()).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(
        dp_dx - dq_dx,
        (p.clone() - q.clone()).grad(&x, &())
    );
    assert_eq!(
        (p_x * q_x, p_x * dq_x + q_x * dp_x),
        (p.clone() * q.clone()).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(
        p_x * dq_dx + q_x * dp_dx,
        (p.clone() * q.clone()).grad(&x, &())
    );
    assert_eq!(
        (p_x / q_x, (dp_x * q_x - p_x * dq_x) / q_x.pow(2.0)),
        (p.clone() / q.clone()).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(
        (dp_x * q_x - p_x * dq_x) / q_x.pow(2.0),
        (p.clone() / q.clone()).grad(&x, &())
    );

    // result for composition
    let (q_of_p_x, dq_of_p_x) = q.eval_forward_grad(&p_x, &dp_x, &());
    let dq_of_p_dx = q.grad(&p_x, &());
    let dp_dx = p.grad(&x, &());
    let (p_of_q_x, dp_of_q_x) = p.eval_forward_grad(&q_x, &dq_x, &());
    let dp_of_q_dx = p.grad(&q_x, &());
    let dq_dx = q.grad(&x, &());

    assert_eq!(
        (q_of_p_x, dq_of_p_x),
        q.clone().compose(p.clone()).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(
        dq_of_p_dx * dp_dx,
        q.clone().compose(p.clone()).grad(&x, &())
    );
    assert_eq!(
        (p_of_q_x, dp_of_q_x),
        p.clone().compose(q.clone()).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(
        dp_of_q_dx * dq_dx,
        p.clone().compose(q.clone()).grad(&x, &())
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

    let (p_w, dp_w) = p.eval_forward_grad(&w, &dw, &());

    assert_eq!((p_w, dp_w), (p_x, dp_x));

    // test special operations
    let p_i32: AutoDiff<(), Polynomial<(), i32, i32>> = AutoDiff::new(Polynomial::<(), i32, i32>::new(vec![1i32, 2i32, 3i32]));
    let p_i32_coerced = p.clone().coerce::<i32, f64>();

    let x_i32 = 2i32;
    let dx_i32 = 1i32;

    let (p_i32_x, dp_i32_x): (i32, i32) = p_i32.eval_forward_grad(&x_i32, &dx_i32, &());
    let (p_i32_coerced_x, dp_i32_coerced_x): (f64, f64) = p_i32_coerced.eval_forward_grad(&x_i32, &dx_i32, &());

    assert_eq!((p_i32_x as f64, dp_i32_x as f64), (p_i32_coerced_x, dp_i32_coerced_x));

    let s1 = 0_usize;
    let s2 = 0.0_f32;

    let p_static_1: AutoDiff<usize, Polynomial<usize, f64, f64>> = AutoDiff::new(Polynomial::<usize, f64, f64>::new(vec![1.0, 2.0, 3.0]));
    let p_static_2: AutoDiff<f32, Polynomial<f32, f64, f64>> = AutoDiff::new(Polynomial::<f32, f64, f64>::new(vec![1.0, 2.0, 3.0]));

    let p1_compat = p_static_1.clone().append_static_args::<f32>();
    let p2_compat = p_static_2.clone().prepend_static_args::<usize>();
    let p1_p2 = p1_compat.clone() + p2_compat.clone();

    let (p1_x, dp1_x): (f64, f64) = p_static_1.eval_forward_grad(&x, &dx, &s1);
    let (p2_x, dp2_x): (f64, f64) = p_static_2.eval_forward_grad(&x, &dx, &s2);
    let (p1_p2_x, dp1_p2_x): (f64, f64) = p1_p2.eval_forward_grad(&x, &dx, &(s1, s2));

    assert_eq!((p1_x + p2_x, dp1_x + dp2_x), (p1_p2_x, dp1_p2_x));
}
