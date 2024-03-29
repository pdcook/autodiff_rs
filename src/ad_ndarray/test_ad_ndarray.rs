use crate::ad_ndarray::scalar::*;
use crate::autodiff::AutoDiff;
use crate::autodiffable::*;
use crate::autotuple::*;
use crate::compose::*;
use crate::diffable::Diffable;
use crate::funcs::Identity;
use crate::traits::InstOne;
use ndarray::prelude::*;
use num::complex::Complex;

use crate as autodiff;
use autodiff_derive::*;

#[derive(Clone, Copy, SimpleForwardDiffable, FuncCompose)]
struct Sum1 {}

impl Diffable<()> for Sum1 {
    type Input = Array1<f64>;
    type Output = Scalar<f64>;
}

impl AutoDiffable<()> for Sum1 {
    fn eval(&self, x: &Array1<f64>, _: &()) -> Scalar<f64> {
        Scalar::new(x.sum())
    }

    fn eval_grad(&self, x: &Array1<f64>, _: &()) -> (Scalar<f64>, Array1<f64>) {
        (self.eval(x, &()), x.one())
    }

    fn eval_conj_grad(&self, x: &Array1<f64>, _: &()) -> (Scalar<f64>, Array1<f64>) {
        (self.eval(x, &()), x.zero())
    }
}

#[derive(Clone, Copy, SimpleForwardDiffable, FuncCompose)]
struct Sum2 {}

impl Diffable<()> for Sum2 {
    type Input = Array2<f64>;
    type Output = Scalar<f64>;
}

impl AutoDiffable<()> for Sum2 {
    fn eval(&self, x: &Array2<f64>, _: &()) -> Scalar<f64> {
        Scalar::new(x.sum())
    }

    fn eval_grad(&self, x: &Array2<f64>, _: &()) -> (Scalar<f64>, Array2<f64>) {
        (self.eval(x, &()), x.one())
    }

    fn eval_conj_grad(&self, x: &Array2<f64>, _: &()) -> (Scalar<f64>, Array2<f64>) {
        (self.eval(x, &()), x.zero())
    }
}

#[derive(Clone, Copy, SimpleForwardDiffable, FuncCompose)]
struct Prod2 {}
// product of all elements in a 2D array
// prod([[a, b], [c, d]]) = abcd
// dprod/dx = [[bcd, acd], [abd, abc]]
// dprod = SUM(dx * prod) = da * bcd + db * acd + dc * abd + dd * abc

impl Diffable<()> for Prod2 {
    type Input = Array2<f64>;
    type Output = Scalar<f64>;
}

impl AutoDiffable<()> for Prod2 {
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

    fn eval_conj_grad(&self, x: &Array2<f64>, _: &()) -> (Scalar<f64>, Array2<f64>) {
        (self.eval(x, &()), x.zero())
    }
}

#[derive(Clone, Copy, SimpleForwardDiffable, FuncCompose)]
struct UpcastN {
    n: usize,
}

impl Diffable<()> for UpcastN {
    type Input = Array1<f64>;
    type Output = Array2<f64>;
}

impl AutoDiffable<()> for UpcastN {
    fn eval(&self, x: &Array1<f64>, _: &()) -> Array2<f64> {
        // make a 2D array with n rows all of which are x
        Array2::from_shape_fn((self.n, x.len()), |(_, i)| x[i])
    }

    fn eval_grad(&self, x: &Array1<f64>, _: &()) -> (Array2<f64>, Array3<f64>) {
        let mut grad = Array3::<f64>::zeros((x.len(), self.n, x.len()));
        for i in 0..x.len() {
            for j in 0..self.n {
                grad[[i, j, i]] = 1.0;
            }
        }

        (self.eval(x, &()), grad)
    }

    fn eval_conj_grad(&self, x: &Array1<f64>, _: &()) -> (Array2<f64>, Array3<f64>)
    {
        let grad = Array3::<f64>::zeros((x.len(), self.n, x.len()));

        (self.eval(x, &()), grad)
    }
}

#[derive(Clone, Copy, SimpleForwardDiffable, FuncCompose)]
struct VertCastN {
    n: usize,
}

impl Diffable<()> for VertCastN {
    type Input = Array1<f64>;
    type Output = Array2<f64>;
}

impl AutoDiffable<()> for VertCastN {
    fn eval(&self, x: &Array1<f64>, _: &()) -> Array2<f64> {
        // make a 2D array with n cols all of which are x
        Array2::from_shape_fn((x.len(), self.n), |(i, _)| x[i])
    }

    fn eval_grad(&self, x: &Array1<f64>, _: &()) -> (Array2<f64>, Array3<f64>) {
        let mut grad = Array3::<f64>::zeros((x.len(), x.len(), self.n));
        for i in 0..x.len() {
            for j in 0..self.n {
                grad[[i, i, j]] = 1.0;
            }
        }

        (self.eval(x, &()), grad)
    }

    fn eval_conj_grad(&self, x: &Array1<f64>, _: &()) -> (Array2<f64>, Array3<f64>)
    {
        let grad = Array3::<f64>::zeros((x.len(), x.len(), self.n));

        (self.eval(x, &()), grad)
    }
}

// test with AutoTuple

#[derive(Clone, Copy)]
struct SumAutoTuples {}

type SInput = AutoTuple<(Array2<f64>, Array1<f64>)>;
type SOutput = AutoTuple<(Scalar<f64>,)>;

impl Diffable<()> for SumAutoTuples {
    type Input = SInput;
    type Output = SOutput;
}

impl AutoDiffable<()> for SumAutoTuples {
    fn eval(&self, x: &SInput, _: &()) -> SOutput {
        AutoTuple::new((Scalar::new((**x).0.sum() + (**x).1.sum()),))
    }

    fn eval_grad(&self, x: &SInput, _: &()) -> (SOutput, SInput) {
        (
            self.eval(x, &()),
            AutoTuple::new(((**x).0.one(), (**x).1.one())),
        )
    }

    fn eval_conj_grad(&self, x: &SInput, _: &()) -> (SOutput, SInput) {
        (
            self.eval(x, &()),
            AutoTuple::new(((**x).0.zero(), (**x).1.zero())),
        )
    }
}

impl ForwardDiffable<()> for SumAutoTuples {
    fn eval_forward_grad(&self, x: &SInput, dx: &SInput, _: &()) -> (SOutput, SOutput) {
        let mut gradval = 0.0_f64;
        gradval += (**dx).0.sum();
        gradval += (**dx).1.sum();
        (self.eval(x, &()), AutoTuple::new((Scalar::new(gradval),)))
    }
    fn eval_forward_conj_grad(&self, x: &SInput, dx: &SInput, _: &()) -> (SOutput, SOutput) {
        (self.eval(x, &()), AutoTuple::new((Scalar::new(0.0),)))
    }
}

#[derive(Clone, Copy, FuncCompose, SimpleForwardDiffable)]
struct ComposeSumUpcastAutoTuple(SumAutoTuples, UpcastAutoTuple);

impl Diffable<()> for ComposeSumUpcastAutoTuple {
    type Input = AutoTuple<(Array1<f64>,)>;
    type Output = AutoTuple<(Scalar<f64>,)>;
}

impl AutoDiffable<()> for ComposeSumUpcastAutoTuple {
    // gradient of Sum(Upcast((a,b,c))) is:
    //
    // Sum(UpcastAutoTuple((a,b,c))) = Sum(AutoTuple(n*(a,b,c), (a,b,c)))
    // where n = len(a,b,c)
    // so
    // d/da Sum(...) = n+1
    fn eval_grad(&self, x: &Self::Input, _: &()) -> (Self::Output, AutoTuple<(Array1<f64>,)>) {
        let (y, _dy) = self.0.eval_grad(&self.1.eval(x, &()), &());
        (
            y,
            (Array1::ones((**x).0.len()) * ((**x).0.len() as f64 + 1.0)).into(),
        )
    }
    fn eval_conj_grad(&self, x: &Self::Input, _: &()) -> (Self::Output, AutoTuple<(Array1<f64>,)>) {
        let (y, _dy) = self.0.eval_conj_grad(&self.1.eval(x, &()), &());
        (y, (Array1::zeros((**x).0.len())).into())
    }
}

impl FuncCompose<(), UpcastAutoTuple> for SumAutoTuples {
    type Output = AutoDiff<(), ComposeSumUpcastAutoTuple>;
    fn func_compose(self, rhs: UpcastAutoTuple) -> Self::Output {
        AutoDiff::new(ComposeSumUpcastAutoTuple(self, rhs))
    }
}

#[derive(Clone, Copy, FuncCompose)]
struct UpcastAutoTuple {}

type UInput = AutoTuple<(Array1<f64>,)>;
type UOutput = AutoTuple<(Array2<f64>, Array1<f64>)>;
type UGrad = AutoTuple<(Array3<f64>, Array2<f64>)>;

impl Diffable<()> for UpcastAutoTuple {
    type Input = UInput;
    type Output = UOutput;
}

impl AutoDiffable<()> for UpcastAutoTuple {
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
        let mut d2_dx = Array2::zeros((n, n));
        // d1_dx[i, j, k] = dupcast[j,k]/dx[i]
        // and since upcast[j,k] = x[j] for all k
        // dupcast[j,k]/dx[i] = 1 if j == i else 0
        //d2_dx[i, j] = d(x[i])/dx[j] = 1 if i == j else 0
        for i in 0..n {
            for j in 0..n {
                d1_dx[[i, j, i]] = 1.0;
                d2_dx[[i, j]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        (self.eval(x, &()), AutoTuple::new((d1_dx, d2_dx)))
    }

    fn eval_conj_grad(&self, x: &UInput, _: &()) -> (UOutput, UGrad) {
        let n = (**x).0.len();
        (self.eval(x, &()), AutoTuple::new((Array3::zeros((n, n, n)), Array2::zeros((n, n)))))
    }
}

impl ForwardDiffable<()> for UpcastAutoTuple {
    fn eval_forward_grad(&self, x: &UInput, dx: &UInput, _: &()) -> (UOutput, UOutput) {
        let xc = (**x).0.clone();
        let n = xc.len();
        let mut d1 = Array2::zeros((n, n));
        let mut d2 = Array1::zeros((n,));
        // d1[j, k] = sum_i dupcast[j,k]/dx[i] * dx[i]
        // and since upcast[j,k] = x[j] for all k
        // dupcast[j,k]/dx[i] = 1 if j == i else 0
        // so d1[j, k] = sum_i dx[i] if j == i else 0
        // -> d1[j, k] = dx[j]
        // d2[i] = sum_j d(x[i])/dx[j] * dx[j] = dx[j] if i == j else 0
        // -> d2[i] = dx[i]
        for i in 0..n {
            for j in 0..n {
                d1[[i, j]] = (**dx).0[i];
            }
            d2[[i]] = (**dx).0[i];
        }

        (self.eval(x, &()), AutoTuple::new((d1, d2)))
    }
    fn eval_forward_conj_grad(&self, x: &UInput, dx: &UInput, _: &()) -> (UOutput, UOutput) {
        let res = self.eval(x, &());
        (res, res.zero())
    }
}

// function that must use custom composition (complex numbers)
// and so only works in forward mode
#[derive(Clone, Copy)]
struct OnlyForward {}

impl Diffable<()> for OnlyForward {
    type Input = Complex<f64>;
    type Output = Complex<f64>;
}

impl ForwardDiffable<()> for OnlyForward {
    fn eval_forward(&self, x: &Complex<f64>, _: &()) -> Complex<f64> {
        x * x.conj()
    }
    fn eval_forward_grad(
        &self,
        x: &Complex<f64>,
        dx: &Complex<f64>,
        _: &(),
    ) -> (Complex<f64>, Complex<f64>) {
        (x * x.conj(), x.conj() * dx)
    }
    fn eval_forward_conj_grad(
        &self,
        x: &Complex<f64>,
        dx: &Complex<f64>,
        _: &(),
    ) -> (Complex<f64>, Complex<f64>) {
        (x * x.conj(), x * dx.conj())
    }
}

#[test]
fn test_ad_ndarray() {
    let f = AutoDiff::new(Sum1 {});
    let i = AutoDiff::new(Identity::new());

    let x = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
    let dx = Array1::<f64>::from_vec(vec![1.0, 1.0, 1.0]);

    let (f_x, df_dx) = f.eval_forward_grad(&x, &dx, &());
    let (i_x, di_dx) = i.eval_forward_grad(&x, &dx, &());

    assert_eq!(f_x.value(), 6.0);
    assert_eq!(df_dx.value(), 3.0);
    assert_eq!(i_x, x);
    assert_eq!(di_dx, dx);

    let g = f * i;
    let (g_x, dg_dx) = g.eval_forward_grad(&x, &dx, &());
    // g = f * i = (a + b + c) * (a, b, c) = (a^2 + ab + ac, ab + b^2 + bc, ac + bc + c^2)
    // dg/dx = [[2a + b + c,            a,            a],
    //          [b         ,   a + 2b + c,            b],
    //          [c         ,            c,   a + b + 2c]]
    // so dg = dg/dx dot dx
    // = [(2a + b + c)da + adb + adc, (a + 2b + c)db + bda + bdc, (a + b + 2c)dc + cda + cdb]
    // so dg = (2 + 2 + 3 + 1 + 1, 1 + 4 + 3 + 2 + 2, 1 + 2 + 6 + 3 + 3) = (9, 12, 15)

    assert_eq!(g_x, Array1::from_vec(vec![6.0, 12.0, 18.0]));
    assert_eq!(dg_dx, Array1::from_vec(vec![9.0, 12.0, 15.0]));

    let y = Array1::from_vec(vec![2.0, 3.0]);
    let dy = Array1::from_vec(vec![1.0, 1.0]);

    let u = AutoDiff::new(UpcastN { n: 3 });
    let (u_y, du_dy) = u.eval_grad(&y, &());
    let du = u.forward_grad(&y, &dy, &());
    // u = (a, b) -> [[a, b], [a, b], [a, b]]
    // du/dy = [ [[1, 0], [1, 0], [1, 0]], [[0, 1], [0, 1], [0, 1]] ]
    // du = [[da, db], [da, db], [da, db]]
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
    assert_eq!(
        du,
        Array2::from_shape_vec((3, 2), vec![dy[0], dy[1], dy[0], dy[1], dy[0], dy[1],]).unwrap()
    );
    assert_eq!(
        du,
        (du_dy.reversed_axes() * &dy)
            .sum_axis(Axis(2))
            .reversed_axes()
    );

    let vu = AutoDiff::new(VertCastN { n: 3 });
    let (vu_y, dvu_dy) = vu.eval_grad(&y, &());
    let dvu = vu.forward_grad(&y, &dy, &());

    assert_eq!(
        vu_y,
        Array2::from_shape_vec((2, 3), vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0]).unwrap()
    );
    assert_eq!(
        dvu_dy,
        Array3::from_shape_vec(
            (2, 2, 3),
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        )
        .unwrap()
    );
    assert_eq!(
        dvu,
        Array2::from_shape_vec((2, 3), vec![dy[0], dy[0], dy[0], dy[1], dy[1], dy[1],]).unwrap()
    );

    let s2 = AutoDiff::new(Sum2 {});

    let v = s2.compose(u);
    // v(a,b) -> sum(u(a,b)) = sum([[a, b], [a, b], [a, b]]) = 3a + 3b
    // ds_dx = [3, 3]
    // du_dx = [[[1, 0], [1, 0], [1, 0]], [[0, 1], [0, 1], [0, 1]]]
    // dv_dx = (ds_dx.rev dot du_dx.rev).rev -> (2,) dot (2, 3, 2) -> (3, 2)
    let (_v_y, _dv_dy) = v.eval_grad(&y, &());

    let (v_y, dv_dy) = v.eval_forward_grad(&y, &dy, &());
    // v(a,b) -> sum(u(a,b)) = sum([[a, b], [a, b], [a, b]]) = 3a + 3b
    // dv = 3da + 3db
    assert_eq!(v_y.value(), 15.0);
    assert_eq!(dv_dy.value(), 6.0);

    let p2 = AutoDiff::new(Prod2 {});
    let w = p2.compose(u);

    let (w_y, dw_dy) = w.eval_forward_grad(&y, &dy, &());
    // w(a,b) -> prod(u(a,b)) = prod([[a, b], [a, b], [a, b]]) = a^3 * b^3
    // dw = 3a^2 * b^3 * da + 3a^3 * b^2 * db
    // = 3 * 2^2 * 3^3 * 1 + 3 * 2^3 * 3^2 * 1 = 540
    assert_eq!(w_y.value(), 216.0);
    assert_eq!(dw_dy.value(), 540.0);

    let sum_auto_tuples: AutoDiff<(), SumAutoTuples> = AutoDiff::new(SumAutoTuples {});
    let a2 = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let a1 = Array1::<f64>::from_vec(vec![5.0, 6.0]);
    let x: SInput = AutoTuple::new((a2.clone(), a1.clone()));
    let dx: SInput = AutoTuple::new((a2.clone().one(), a1.clone().one()));

    let (y, dy_dx): (SOutput, SOutput) = sum_auto_tuples.eval_forward_grad(&x, &dx, &());
    assert_eq!((*y).0.value(), 21.0);
    assert_eq!((*dy_dx).0.value(), 6.0);

    let f2 = sum_auto_tuples * sum_auto_tuples;
    let (f2_x, df2_dx): (SOutput, SOutput) = f2.eval_forward_grad(&x, &dx, &());
    assert_eq!((*f2_x).0.value(), 441.0);
    assert_eq!((*df2_dx).0.value(), 2.0 * 21.0 * 6.0);

    let upcast_auto_tuple: AutoDiff<(), UpcastAutoTuple> = AutoDiff::new(UpcastAutoTuple {});

    let a1 = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);

    let x = AutoTuple::new((a1.clone(),));
    let dx = AutoTuple::new((a1.clone().one(),));

    let (y, dy_dx): (SInput, UOutput) = upcast_auto_tuple.eval_forward_grad(&x, &dx, &());

    assert_eq!(
        (*y).0,
        Array2::from_shape_vec((3, 3), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
            .unwrap()
            .t()
    );
    assert_eq!((*y).1, a1);

    assert_eq!(
        (*dy_dx).0,
        (dx.0 .0.clone()
            * Array3::from_shape_vec(
                (3, 3, 3),
                vec![
                    1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1., 0.0, 0.0, 0., 1., 0., 0., 1., 0., 0., 1., 0.,
                    0., 0., 1., 0., 0., 1., 0., 0., 1.,
                ]
            )
            .unwrap())
        .sum_axis(Axis(2))
    );
    assert_eq!((*dy_dx).1, (*dx).0);

    let (s_of_u, _dsu) = sum_auto_tuples.eval_forward_grad(&y, &dy_dx, &());
    assert_eq!((*s_of_u).0.value(), 24.0);

    // now composing the two autotuple functions
    let su = sum_auto_tuples.compose(upcast_auto_tuple);
    // su(a, b, c) = sum(upcast(a, b, c))
    //          = sum(((a, b, c), (a, b, c), (a, b, c)), (a, b, c))
    //          = 4a + 4b + 4c
    // dsu = 4da + 4db + 4dc

    let (su_x, dsu_dx) = su.eval_forward_grad(&x, &dx, &());

    assert_eq!(su_x, s_of_u);
    assert_eq!((*dsu_dx).0.value(), 12.0);
}
