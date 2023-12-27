use crate::ad_ndarray::scalar::*;
use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use crate::autotuple::*;
use crate::func_traits::Compose;
use crate::funcs::Identity;
use crate::traits::InstOne;
use ndarray::prelude::*;
use crate::traits::*;

#[test]
fn test_ad_ndarray() {
    #[derive(Clone, Copy)]
    struct Sum1 {}

    impl AutoDiffable<(), Array1<f64>, Scalar<f64>, Array1<f64>> for Sum1 {
        fn eval(&self, x: &Array1<f64>, _: &()) -> Scalar<f64> {
            Scalar::new(x.sum())
        }

        fn eval_grad(&self, x: &Array1<f64>, _: &()) -> (Scalar<f64>, Array1<f64>) {
            (self.eval(x, &()), Array1::ones(x.len()))
        }
    }

    #[derive(Clone, Copy)]
    struct Sum2 {}

    impl AutoDiffable<(), Array2<f64>, Scalar<f64>, Array2<f64>> for Sum2 {
        fn eval(&self, x: &Array2<f64>, _: &()) -> Scalar<f64> {
            Scalar::new(x.sum())
        }

        fn eval_grad(&self, x: &Array2<f64>, _: &()) -> (Scalar<f64>, Array2<f64>) {
            (self.eval(x, &()), Array2::ones(x.dim()))
        }
    }

    #[derive(Clone, Copy)]
    struct Prod2 {}
    // product of all elements in a 2D array
    // prod([[a, b], [c, d]]) = abcd
    // dprod/dx = [[bcd, acd], [abd, abc]]

    impl AutoDiffable<(), Array2<f64>, Scalar<f64>, Array2<f64>> for Prod2 {
        fn eval(&self, x: &Array2<f64>, _: &()) -> Scalar<f64> {
            Scalar::new(x.iter().product())
        }

        fn eval_grad(&self, x: &Array2<f64>, _: &()) -> (Scalar<f64>, Array2<f64>) {
            let mut grad = Array2::<f64>::ones(x.raw_dim());
            for i in 0..x.dim().0 {
                for j in 0..x.dim().1 {
                    let mut x = x.clone();
                    x[[i, j]] = 1.0;
                    grad[[i, j]] = self.eval(&x, &()).value();
                }
            }
            (self.eval(x, &()), grad)
        }
    }

    #[derive(Clone, Copy)]
    struct UpcastN {
        n: usize,
    }

    impl AutoDiffable<(), Array1<f64>, Array2<f64>, Array3<f64>> for UpcastN {
        fn eval(&self, x: &Array1<f64>, _: &()) -> Array2<f64> {
            // make a 2D array with n rows all of which are x
            Array2::from_shape_fn((self.n, x.len()), |(_, i)| x[i])
        }

        fn eval_grad(&self, x: &Array1<f64>, _: &()) -> (Array2<f64>, Array3<f64>) {
            // gradient is a 3D array grad[i, j, k] which represents
            // df[j, k] / dx[i]
            let mut grad = Array3::zeros((x.len(), self.n, x.len()));
            for i in 0..x.len() {
                for j in 0..self.n {
                    grad[[i, j, i]] = 1.0;
                }
            }
            (self.eval(x, &()), grad)
        }
    }

    let f = AutoDiff::new(Sum1 {});
    let i = AutoDiff::new(Identity::new());

    let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    let (f_x, df_dx) = f.eval_grad(&x, &());
    let (i_x, di_dx) = i.eval_grad(&x, &());

    assert_eq!(f_x.value(), 6.0);
    assert_eq!(df_dx, Array1::ones(3));
    assert_eq!(i_x, x);
    assert_eq!(di_dx, Array1::ones(3));

    let g = f * i;
    let (g_x, dg_dx) = g.eval_grad(&x, &());
    // g = f * i = (a + b + c) * (a, b, c) = (a^2 + ab + ac, ab + b^2 + bc, ac + bc + c^2)
    // dg/dx = (2a + b + c, a + 2b + c, a + b + 2c)
    // so dg/dx = (2 + 2 + 3, 1 + 4 + 3, 1 + 2 + 6) = (7, 8, 9)

    assert_eq!(g_x, Array1::from_vec(vec![6.0, 12.0, 18.0]));
    assert_eq!(dg_dx, Array1::from_vec(vec![7.0, 8.0, 9.0]));

    let y = Array1::from_vec(vec![2.0, 3.0]);

    let u = AutoDiff::new(UpcastN { n: 3 });
    let (u_y, du_dy) = u.eval_grad(&y, &());
    // u = (a, b) -> [[a, b], [a, b], [a, b]]
    // du/dy = [ [[1, 0], [1, 0], [1, 0]], [[0, 1], [0, 1], [0, 1]] ]
    assert_eq!(
        u_y,
        Array2::from_shape_vec((3, 2), vec![2.0, 3.0, 2.0, 3.0, 2.0, 3.0]).unwrap()
    );
    assert_eq!(
        du_dy,
        Array3::from_shape_vec(
            (2, 3, 2),
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        )
        .unwrap()
    );

    let s2 = AutoDiff::new(Sum2 {});

    let v = s2.compose(u);

    let (v_y, dv_dy) = v.eval_grad(&y, &());
    // v(a,b) -> sum(u(a,b)) = sum([[a, b], [a, b], [a, b]]) = 3a + 3b
    // dv/dy = [3, 3]
    assert_eq!(v_y.value(), 15.0);
    assert_eq!(dv_dy, Array1::from_vec(vec![3.0, 3.0]));

    let p2 = AutoDiff::new(Prod2 {});
    let w = p2.compose(u);

    let (w_y, dw_dy) = w.eval_grad(&y, &());
    // w(a,b) -> prod(u(a,b)) = prod([[a, b], [a, b], [a, b]]) = a^3 * b^3
    // dw/dy = [3a^2b^3, 3a^3b^2]
    // dw/dy = [324, 216]
    assert_eq!(w_y.value(), 216.0);
    assert_eq!(dw_dy, Array1::from_vec(vec![324.0, 216.0]));

    // test with AutoTuple

    #[derive(Clone, Copy)]
    struct SumAutoTuples {}

    type SInput = AutoTuple<(Array2<f64>, Array1<f64>)>;
    type SOutput = AutoTuple<(Scalar<f64>,)>;
    type SGrad = AutoTuple<(Array2<f64>, Array1<f64>)>;

    impl AutoDiffable<(), SInput, SOutput, SGrad> for SumAutoTuples {
        fn eval(&self, x: &SInput, _: &()) -> SOutput {
            AutoTuple::new((Scalar::new((**x).0.sum() + (**x).1.sum()),))
        }

        fn eval_grad(&self, x: &SInput, _: &()) -> (SOutput, SGrad) {
            let grad = x.one();
            (self.eval(x, &()), grad)
        }
    }

    let sum_auto_tuples: AutoDiff<(), SInput, SOutput, SGrad, SumAutoTuples> =
        AutoDiff::new(SumAutoTuples {});
    let a2 = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let a1 = Array1::<f64>::from_vec(vec![5.0, 6.0]);
    let x: SInput = AutoTuple::new((a2.clone(), a1.clone()));

    let (y, dy_dx): (SOutput, SGrad) = sum_auto_tuples.eval_grad(&x, &());
    assert_eq!((*y).0.value(), 21.0);
    assert_eq!(*dy_dx, (a2.one(), a1.one()));

    let f2 = sum_auto_tuples * sum_auto_tuples;
    let (f2_x, df2_dx): (SOutput, SGrad) = f2.eval_grad(&x, &());
    assert_eq!((*f2_x).0.value(), 441.0);
    assert_eq!(*df2_dx, (a2.one() * 2.0 * 21.0, a1.one() * 2.0 * 21.0));

    #[derive(Clone, Copy)]
    struct UpcastAutoTuple {}

    type UInput = AutoTuple<(Array1<f64>,)>;
    type UOutput = AutoTuple<(Array2<f64>, Array1<f64>)>;
    type UGrad = AutoTuple<(Array3<f64>, Array2<f64>)>;

    impl AutoDiffable<(), UInput, UOutput, UGrad> for UpcastAutoTuple {
        fn eval(&self, x: &UInput, _: &()) -> UOutput {
            // make square matrix where the rows are x
            let x = (**x).0.clone();
            let n = x.len();
            let mut y = Array2::zeros((n, n));
            for i in 0..n {
                y.row_mut(i).assign(&x);
            }
            AutoTuple::new((y, x))
        }

        fn eval_grad(&self, x: &UInput, _: &()) -> (UOutput, UGrad) {
            let xc = (**x).0.clone();
            let n = xc.len();
            let mut d1_dx = Array3::zeros((n, n, n));
            // d1_dx[i, j, k] = dupcast[j,k]/dx[i]
            // and since upcast[j,k] = x[j] for all k
            // dupcast[j,k]/dx[i] = 1 if j == i else 0
            for i in 0..n {
                for j in 0..n {
                    d1_dx[[i, j, i]] = 1.0;
                }
            }
            // d2_dx[i, j] = d(x[i])/dx[j] = 1 if i == j else 0

            (self.eval(x, &()), AutoTuple::new((d1_dx, Array2::eye(n))))
        }
    }

    let upcast_auto_tuple: AutoDiff<(), UInput, UOutput, UGrad, UpcastAutoTuple> =
        AutoDiff::new(UpcastAutoTuple {});

    let a1 = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);

    let x = AutoTuple::new((a1.clone(),));

    let (y, dy_dx): (SInput, UGrad) = upcast_auto_tuple.eval_grad(&x, &());

    assert_eq!(
        (*y).0,
        Array2::from_shape_vec((3, 3), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
            .unwrap()
            .t()
    );
    assert_eq!((*y).1, a1);

    assert_eq!(
        (*dy_dx).0,
        Array3::from_shape_vec(
            (3, 3, 3),
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1., 0.0, 0.0, 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
                0., 1., 0., 0., 1., 0., 0., 1.,
            ]
        )
        .unwrap()
    );
    assert_eq!((*dy_dx).1, Array2::eye(3));

    let (s_of_u, dsu) = sum_auto_tuples.eval_grad(&y, &());
    assert_eq!((*s_of_u).0.value(), 24.0);

    // now composing the two autotuple functions
    let su = sum_auto_tuples.compose(upcast_auto_tuple);

    let (su_x, dsu_dx) = su.eval_grad(&x, &());

    assert_eq!(su_x, s_of_u);
    assert_eq!((*dsu_dx).0, ((*dsu).0.clone() * (*dy_dx).0.clone()).sum_axis(Axis(2)).sum_axis(Axis(1)));
    assert_eq!((*dsu_dx).1, ((*dsu).1.clone() * (*dy_dx).1.clone()).sum_axis(Axis(1)));
}
