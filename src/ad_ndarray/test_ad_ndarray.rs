use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use ndarray::prelude::*;
use crate::funcs::Identity;
use crate::func_traits::Compose;
use crate::ad_ndarray::scalar::*;

#[test]
fn test_ad_ndarray() {

    #[derive(Clone, Copy)]
    struct Sum1 { }

    impl AutoDiffable<(), Array1<f64>, Scalar<f64>, Array1<f64>> for Sum1 {
        fn eval(&self, x: &Array1<f64>, _: &()) -> Scalar<f64> {
            Scalar::new(x.sum())
        }

        fn eval_grad(&self, x: &Array1<f64>, _: &()) -> (Scalar<f64>, Array1<f64>) {
            (self.eval(x, &()), Array1::ones(x.len()))
        }
    }

    #[derive(Clone, Copy)]
    struct Sum2 { }

    impl AutoDiffable<(), Array2<f64>, Scalar<f64>, Array2<f64>> for Sum2 {
        fn eval(&self, x: &Array2<f64>, _: &()) -> Scalar<f64> {
            Scalar::new(x.sum())
        }

        fn eval_grad(&self, x: &Array2<f64>, _: &()) -> (Scalar<f64>, Array2<f64>) {
            (self.eval(x, &()), Array2::ones(x.dim()))
        }
    }

    #[derive(Clone, Copy)]
    struct Prod2 { }
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
    struct UpcastN { n: usize }

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

    let f = AutoDiff::new(Sum1 { });
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
    assert_eq!(u_y, Array2::from_shape_vec((3, 2), vec![2.0, 3.0, 2.0, 3.0, 2.0, 3.0]).unwrap());
    assert_eq!(du_dy, Array3::from_shape_vec((2, 3, 2), vec![
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0]).unwrap());

    let s2 = AutoDiff::new(Sum2 { });

    let v = s2.compose(u);

    let (v_y, dv_dy) = v.eval_grad(&y, &());
    // v(a,b) -> sum(u(a,b)) = sum([[a, b], [a, b], [a, b]]) = 3a + 3b
    // dv/dy = [3, 3]
    assert_eq!(v_y.value(), 15.0);
    //assert_eq!(dv_dy.sum_axis(Axis(2)).sum_axis(Axis(1)), Array1::from_vec(vec![3.0, 3.0]));
    assert_eq!(dv_dy, Array1::from_vec(vec![3.0, 3.0]));

    let p2 = AutoDiff::new(Prod2 { });
    let w = p2.compose(u);

    let (w_y, dw_dy) = w.eval_grad(&y, &());
    // w(a,b) -> prod(u(a,b)) = prod([[a, b], [a, b], [a, b]]) = a^3 * b^3
    // dw/dy = [3a^2b^3, 3a^3b^2]
    // dw/dy = [324, 216]
    assert_eq!(w_y.value(), 216.0);
    //assert_eq!(dw_dy.sum_axis(Axis(2)).sum_axis(Axis(1)), Array1::from_vec(vec![324.0, 216.0]));
    assert_eq!(dw_dy, Array1::from_vec(vec![324.0, 216.0]));

}

