mod adops;
mod arithmetic;
mod autodiff;
mod autodiffable;
mod func_traits;
mod funcs;
mod traits;

#[cfg(feature = "ndarray")]
mod ad_ndarray;
#[cfg(feature = "ndarray")]
use ad_ndarray::impls::*;
#[cfg(feature = "ndarray")]
use ndarray::{Array, Array1, Array2, Array3, Dimension};

use crate::arithmetic::*;
use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use crate::func_traits::Compose;
use crate::funcs::*;
use crate::traits::{InstOne, InstZero};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use num::traits::Float;

// ================================================= //

#[derive(Debug, Clone, Copy)]
struct Upcast();
// takes in an Array1<f64> and returns an Array2<f64> where each row is the input

impl Upcast {
    fn new() -> Self {
        Self()
    }
}

impl<'a> AutoDiffable<'a, usize, Array1<f64>, Array2<f64>, Array3<f64>> for Upcast {
    fn eval(&self, x: &Array1<f64>, rows: &usize) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((*rows, x.len()));
        for i in 0..*rows {
            out.row_mut(i).assign(x);
        }
        out
    }

    /// grad[i, j, k] = d out[j, k] / d x[i]
    fn grad(&self, x: &Array1<f64>, rows: &usize) -> Array3<f64> {
        let mut g = Array3::<f64>::zeros((x.len(), *rows, x.len()));
        for j in 0..*rows {
            for i in 0..x.len() {
                g[[i, j, i]] = 1.0;
            }
        }
        g
    }
}

#[derive(Debug, Clone)]
struct Exp<A, D>(PhantomData<(A, D)>)
where
    A: InstZero + InstOne + Clone,
    D: Dimension;

impl<A, D> Exp<A, D>
where
    A: InstZero + InstOne + Clone,
    D: Dimension,
{
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl<A: InstZero + InstOne + Clone, D: Dimension + Clone> Copy for Exp<A, D> {}

impl<'a, A, D> AutoDiffable<'a, (), Array<A, D>, Array<A, D>, Array<A, D>> for Exp<A, D>
where
    A: InstZero + InstOne + Clone + 'a + Float,
    D: Dimension + 'a,
    for<'b> Array<A, D>:
        StrongAssociatedArithmetic<'b, Array<A, D>> + WeakAssociatedArithmetic<'b, Array<A, D>>,
    for<'b> &'b Array<A, D>: CastingArithmetic<'b, Array<A, D>, Array<A, D>>,
{
    fn eval(&self, x: &Array<A, D>, _: &()) -> Array<A, D> {
        x.mapv(|x| x.exp())
    }

    fn grad(&self, x: &Array<A, D>, _: &()) -> Array<A, D> {
        x.mapv(|x| x.exp())
    }
}

fn main() {
    let x = Array1::<f64>::from(vec![1.0, 2.0, 3.0]);
    let rows = 2;

    let f = AutoDiff::new(Upcast::new());
    println!("f(x) = {}", f.eval(&x, &rows));
    println!("f'(x) = {}", f.grad(&x, &rows));

    let g = f * f * f + f;
    println!("g(x) = {}", g.eval(&x, &rows));
    println!("g'(x) = {}", g.grad(&x, &rows));

}
