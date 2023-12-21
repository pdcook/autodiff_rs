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
use crate::ad_ndarray::impls::*;
#[cfg(feature = "ndarray")]
use ndarray::{Array, Array1, Array2, Dimension};

#[cfg(feature = "complex")]
mod complex;
#[cfg(feature = "complex")]
use crate::complex::funcs::*;

use crate::arithmetic::*;
use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use crate::func_traits::Compose;
use crate::funcs::*;
use crate::traits::{InstOne, InstZero};
use num::complex::Complex;
use num::traits::Float;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ================================================= //

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
    let x = Array1::<i64>::from(vec![1, 2, 3]);

    let f = AutoDiff::new(Identity::new());
    println!("f(x) = {}", f.eval(&x, &()));
    println!("f'(x) = {}", f.grad(&x, &()));

    let g = (f * Array1::from(vec![2]))/(f*f);
    println!("g(x) = {}", g.eval(&x, &()));
    println!("g'(x) = {}", g.grad(&x, &()));

    //let c = AutoDiff::new(Constant::new(x.clone()));

    //let h = g + c;
    //println!("h(x) = {}", (&h).eval(&x, &()));
    //println!("h'(x) = {}", h.grad(&x, &()));

    //let g = AutoDiff::new(Monomial::new(Complex::new(1.0, 0.0)));
    //println!("g(x) = {}", g.eval(&x, &()));
    //println!("g'(x) = {}", g.grad(&x, &()));

    //let h = f.clone().compose(g);
    //println!("h(x) = {}", h.eval(&x, &()));
    //println!("h'(x) = {}", h.grad(&x, &()));

    //let i = f.clone() * f;
    //println!("i(x) = {}", i.eval(&x, &()));
    //println!("i'(x) = {}", i.grad(&x, &()));
}
