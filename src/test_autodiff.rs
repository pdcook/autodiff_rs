use crate::autodiff::AutoDiff;
use crate::autodiffable::*;
use crate::compose::*;
use crate::func_traits::*;
use crate::funcs::*;
use num::complex::Complex;
use num::traits::Pow;
use std::ops::Deref;

#[test]
fn test_all_ops() {
    // test all supported unary operations on p(x) = 1 + 2x + 3x^2
    // test all supported binary operations on p(x) and q(x) = x^5

    let x = 2.0_f64;
    let dx = 0.5_f64;

    let p = AutoDiff::new(Polynomial::new(vec![1.0, 2.0, 3.0]));
    let q = AutoDiff::new(Monomial::<(), f64, f64>::new(5.0));

    let (p_x, dp_x): (f64, f64) = p.eval_forward_grad(&x, &dx, &());
    let dp_dx: f64 = p.grad(&x, &());
    let dp_dconjx: f64 = p.conj_grad(&x, &());
    let (q_x, dq_x): (f64, f64) = q.eval_forward_grad(&x, &dx, &());
    let dq_dx: f64 = q.grad(&x, &());
    let dq_dconjx: f64 = q.conj_grad(&x, &());

    // assert equality
    assert_eq!(p_x, 1.0 + 2.0 * x + 3.0 * x.pow(2.0));
    assert_eq!(dp_x, (2.0 + 6.0 * x) * dx);
    assert_eq!(dp_dx, 2.0 + 6.0 * x);
    assert_eq!(dp_dconjx, 0.0);

    assert_eq!(q_x, x.pow(5));
    assert_eq!(dq_x, dx * 5.0 * x.pow(4));
    assert_eq!(dq_dx, 5.0 * x.pow(4));
    assert_eq!(dq_dconjx, 0.0);

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
    // abssqr
    assert_eq!(
        (p_x.abs_sqr(), 2.0 * p_x * dp_x),
        p.clone().abs_sqr().eval_forward_grad(&x, &dx, &())
    );
    // signum
    assert_eq!(
        (p_x.signum(), 0.0),
        p.clone().signum().eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(0.0, p.clone().signum().grad(&x, &()));
    // conj
    assert_eq!((p_x.conj(), 0.0), p.conj().eval_forward_grad(&x, &dx, &()));

    // binary ops with constants
    let c = 3.0_f64;
    assert_eq!(
        (p_x + c, dp_x),
        (p.clone() + c).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(dp_dx, (p.clone() + c).grad(&x, &()));
    assert_eq!(
        (p_x - c, dp_x),
        (p.clone() - c).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(dp_dx, (p.clone() - c).grad(&x, &()));
    assert_eq!(
        (p_x * c, dp_x * c),
        (p.clone() * c).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(dp_dx * c, (p.clone() * c).grad(&x, &()));
    assert_eq!(
        (p_x / c, dp_x / c),
        (p.clone() / c).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(dp_dx / c, (p.clone() / c).grad(&x, &()));
    assert_eq!(
        (p_x.pow(c), c * p_x.pow(c - 1.0) * dp_x),
        (p.clone().pow(c)).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(
        c * p_x.pow(c - 1.0) * dp_dx,
        (p.clone().pow(c)).grad(&x, &())
    );

    // binary ops with other functions
    assert_eq!(
        (p_x + q_x, dp_x + dq_x),
        (p.clone() + q.clone()).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(dp_dx + dq_dx, (p.clone() + q.clone()).grad(&x, &()));
    assert_eq!(
        (p_x - q_x, dp_x - dq_x),
        (p.clone() - q.clone()).eval_forward_grad(&x, &dx, &())
    );
    assert_eq!(dp_dx - dq_dx, (p.clone() - q.clone()).grad(&x, &()));
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
        (dp_dx * q_x - p_x * dq_dx) / q_x.pow(2.0),
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

    let w = Wrapper(x.clone());
    let dw = Wrapper(dx.clone());

    let (p_w, dp_w) = p.eval_forward_grad(&w, &dw, &());

    assert_eq!((p_w, dp_w), (p_x, dp_x));

    // test special operations
    let p_i32: AutoDiff<(), Polynomial<(), i32, i32>> =
        AutoDiff::new(Polynomial::<(), i32, i32>::new(vec![1i32, 2i32, 3i32]));
    let p_i32_coerced = p.clone().coerce::<i32, f64>();

    let x_i32 = 3i32;
    let dx_i32 = 2i32;

    let (p_i32_x, dp_i32_x): (i32, i32) = p_i32.eval_forward_grad(&x_i32, &dx_i32, &());
    let (p_i32_coerced_x, dp_i32_coerced_x): (f64, f64) =
        p_i32_coerced.eval_forward_grad(&x_i32, &dx_i32, &());

    assert_eq!(
        (p_i32_x as f64, dp_i32_x as f64),
        (p_i32_coerced_x, dp_i32_coerced_x)
    );

    let s1 = 3_usize;
    let s2 = 2.0_f32;

    let p_static_1: AutoDiff<usize, Polynomial<usize, f64, f64>> =
        AutoDiff::new(Polynomial::<usize, f64, f64>::new(vec![1.0, 2.0, 3.0]));
    let p_static_2: AutoDiff<f32, Polynomial<f32, f64, f64>> =
        AutoDiff::new(Polynomial::<f32, f64, f64>::new(vec![1.0, 2.0, 3.0]));

    let p1_compat = p_static_1.clone().append_static_args::<f32>();
    let p2_compat = p_static_2.clone().prepend_static_args::<usize>();
    let p1_p2 = p1_compat.clone() + p2_compat.clone();

    let (p1_x, dp1_x): (f64, f64) = p_static_1.eval_forward_grad(&x, &dx, &s1);
    let (p2_x, dp2_x): (f64, f64) = p_static_2.eval_forward_grad(&x, &dx, &s2);
    let (p1_p2_x, dp1_p2_x): (f64, f64) = p1_p2.eval_forward_grad(&x, &dx, &(s1, s2));

    assert_eq!((p1_x + p2_x, dp1_x + dp2_x), (p1_p2_x, dp1_p2_x));

    // test with Complex numbers
    let z = Complex::<f64>::new(2.0_f64, 3.0);
    let dz = Complex::<f64>::new(0.5_f64, 0.75);

    let i_complex = AutoDiff::new(Identity::new());
    let m_complex = i_complex.abs_sqr();

    let (i_z, di_z): (Complex<f64>, Complex<f64>) = i_complex.eval_forward_grad(&z, &dz, &());
    let di_dz: Complex<f64> = i_complex.grad(&z, &());
    let di_dconjz: Complex<f64> = i_complex.conj_grad(&z, &());
    let di_conjz: Complex<f64> = i_complex.forward_conj_grad(&z, &dz, &());

    let (m_z, dm_z): (Complex<f64>, Complex<f64>) = m_complex.eval_forward_grad(&z, &dz, &());
    let dm_dz: Complex<f64> = m_complex.grad(&z, &());
    let dm_dconjz: Complex<f64> = m_complex.conj_grad(&z, &());
    let dm_conjz: Complex<f64> = m_complex.forward_conj_grad(&z, &dz, &());

    // Wirtinger derivatives
    assert_eq!(i_z, z);
    assert_eq!(di_z, dz);
    assert_eq!(di_dz, Complex::new(1.0, 0.0));
    assert_eq!(di_dconjz, Complex::new(0.0, 0.0));
    assert_eq!(di_conjz, Complex::new(0.0, 0.0));

    assert_eq!(m_z, z.norm_sqr().into());
    assert_eq!(dm_z, z.conj() * dz);
    assert_eq!(dm_dz, z.conj());
    assert_eq!(dm_dconjz, z);
    assert_eq!(dm_conjz, z * dz.conj());

    // abs
    let a_complex = i_complex.abs();
    let (a_z, da_z): (Complex<f64>, Complex<f64>) = a_complex.eval_forward_grad(&z, &dz, &());
    let da_dz: Complex<f64> = a_complex.grad(&z, &());
    let da_dconjz: Complex<f64> = a_complex.conj_grad(&z, &());
    let da_conjz: Complex<f64> = a_complex.forward_conj_grad(&z, &dz, &());

    // Wirtinger derivative of abs(z) = 0.5 * conj(z)/(|z|)
    assert_eq!(a_z, z.norm().into());
    assert_eq!(da_z, 0.5 * z.conj() * dz / z.norm());
    assert_eq!(da_dz, 0.5 * z.conj() / z.norm());
    assert_eq!(da_dconjz, 0.5 * z / z.norm());
    assert_eq!(da_conjz, 0.5 * z * dz.conj() / z.norm());

    // trivial composition
    let m_of_i = m_complex.compose(i_complex);

    let (m_of_i_z, dm_of_i_z): (Complex<f64>, Complex<f64>) =
        m_of_i.eval_forward_grad(&z, &dz, &());
    let dm_of_i_dz: Complex<f64> = m_of_i.grad(&z, &());
    let dm_of_i_dconjz: Complex<f64> = m_of_i.conj_grad(&z, &());
    let dm_of_i_conjz: Complex<f64> = m_of_i.forward_conj_grad(&z, &dz, &());

    assert_eq!(m_of_i_z, z.norm_sqr().into());
    assert_eq!(dm_of_i_z, z.conj() * dz);
    assert_eq!(dm_of_i_dz, z.conj());
    assert_eq!(dm_of_i_dconjz, z);
    assert_eq!(dm_of_i_conjz, z * dz.conj());

    // test mix of operations on complex numbers
    // z + conj(z) (= 2 * Re(z))

    let z_plus_conjz = i_complex + i_complex.conj();

    let (z_plus_conjz_z, dz_plus_conjz_z): (Complex<f64>, Complex<f64>) =
        z_plus_conjz.eval_forward_grad(&z, &dz, &());
    let dz_plus_conjz_dz: Complex<f64> = z_plus_conjz.grad(&z, &());
    let dz_plus_conjz_dconjz: Complex<f64> = z_plus_conjz.conj_grad(&z, &());
    let dz_plus_conjz_conjz: Complex<f64> = z_plus_conjz.forward_conj_grad(&z, &dz, &());

    assert_eq!(z_plus_conjz_z, (2.0 * z.re).into());
    assert_eq!(dz_plus_conjz_z, dz);
    assert_eq!(dz_plus_conjz_dz, Complex::new(1.0, 0.0));
    assert_eq!(dz_plus_conjz_dconjz, Complex::new(1.0, 0.0));
    assert_eq!(dz_plus_conjz_conjz, dz.conj());

    // z * (z + conj(z)) (= z * 2 * Re(z))

    let z_times_z_plus_conjz = i_complex * z_plus_conjz;

    let (z_times_z_plus_conjz_z, dz_times_z_plus_conjz_z): (Complex<f64>, Complex<f64>) =
        z_times_z_plus_conjz.eval_forward_grad(&z, &dz, &());
    let dz_times_z_plus_conjz_dz: Complex<f64> = z_times_z_plus_conjz.grad(&z, &());
    let dz_times_z_plus_conjz_dconjz: Complex<f64> = z_times_z_plus_conjz.conj_grad(&z, &());
    let dz_times_z_plus_conjz_conjz: Complex<f64> =
        z_times_z_plus_conjz.forward_conj_grad(&z, &dz, &());

    assert_eq!(z_times_z_plus_conjz_z, 2.0 * z.re * z);
    assert_eq!(dz_times_z_plus_conjz_z, (2.0 * z + z.conj()) * dz);
    assert_eq!(dz_times_z_plus_conjz_dz, 2.0 * z + z.conj());
    assert_eq!(dz_times_z_plus_conjz_dconjz, z);
    assert_eq!(dz_times_z_plus_conjz_conjz, z * dz.conj());

    // | z *(z + conj(z)) |

    let abs_z_times_z_plus_conjz = z_times_z_plus_conjz.abs();
    let (abs_z_times_z_plus_conjz_z, dabs_z_times_z_plus_conjz_z): (Complex<f64>, Complex<f64>) =
        abs_z_times_z_plus_conjz.eval_forward_grad(&z, &dz, &());
    let dabs_z_times_z_plus_conjz_dz: Complex<f64> = abs_z_times_z_plus_conjz.grad(&z, &());
    let dabs_z_times_z_plus_conjz_dconjz: Complex<f64> =
        abs_z_times_z_plus_conjz.conj_grad(&z, &());
    let dabs_z_times_z_plus_conjz_conjz: Complex<f64> =
        abs_z_times_z_plus_conjz.forward_conj_grad(&z, &dz, &());

    assert_eq!(abs_z_times_z_plus_conjz_z, (2.0 * z.re * z).norm().into());

    macro_rules! assert_complex_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if (!($x.re - $y.re < $d.re
                || $y.re - $x.re < $d.re)
                || !(
                $x.im - $y.im < $d.im
                || $y.im - $x.im < $d.im))
            {
                // show a panic message with the values
                panic!(
                    "assertion failed: `(left == right)`\n  left: `{:?}`,\n right: `{:?}`\n delta: `{:?}`",
                    $x, $y, $d
                );
            }
        };
    }

    let expected_dabs_z_times_z_plus_conjz_dz: Complex<f64> = Complex::<f64>::new(0.5, 0.0)
        * z.conj()
        * (z.conj().pow(2.0) + 4.0 * z.conj() * z + 3.0 * z.pow(2.0))
        / Complex::<f64>::sqrt(z.conj() * z * (z.conj() + z).pow(2.0));

    assert_complex_delta!(
        dabs_z_times_z_plus_conjz_z,
        expected_dabs_z_times_z_plus_conjz_dz * dz,
        Complex::<f64>::new(1e-15, 1e-15)
    );

    assert_complex_delta!(
        dabs_z_times_z_plus_conjz_dz,
        expected_dabs_z_times_z_plus_conjz_dz,
        Complex::<f64>::new(1e-15, 1e-15)
    );

    // the conjugate gradient is the same
    assert_complex_delta!(
        dabs_z_times_z_plus_conjz_dconjz,
        expected_dabs_z_times_z_plus_conjz_dz,
        Complex::<f64>::new(1e-15, 1e-15)
    );
    assert_complex_delta!(
        dabs_z_times_z_plus_conjz_conjz,
        expected_dabs_z_times_z_plus_conjz_dz * dz.conj(),
        Complex::<f64>::new(1e-15, 1e-15)
    );
}
