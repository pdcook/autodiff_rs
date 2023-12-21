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

#[cfg(feature = "complex")]
mod complex;
#[cfg(feature = "complex")]
use crate::complex::funcs::*;

use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use crate::traits::{InstZero, InstOne};
use crate::func_traits::Compose;
use crate::funcs::*;
use num::complex::Complex;
use ndarray::Array1;
use std::ops::{Add, Mul, Sub, Div, Neg};
use crate::arithmetic::*;

// ================================================= //

fn main() {
    let x = Array1::from(vec![0.0, 2.0, 3.0_f64]);

    let f = AutoDiff::new(Identity::<Array1::<f64>>::new());
    println!("f(x) = {}", f.eval(&x, &()));
    println!("f'(x) = {}", f.grad(&x, &()));

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
