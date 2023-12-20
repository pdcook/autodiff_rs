mod adops;
mod arithmetic;
mod autodiff;
mod autodiffable;
mod func_traits;
mod funcs;

#[cfg(feature = "complex")]
mod complex;
#[cfg(feature = "complex")]
use crate::complex::funcs::*;

use crate::autodiff::AutoDiff;
use crate::autodiffable::AutoDiffable;
use crate::func_traits::Compose;
use crate::funcs::*;
use num::complex::Complex;

// ================================================= //

fn main() {

    let x = Complex::new(1.0, 1.0);

    let f = AutoDiff::new(Polynomial::new(vec![Complex::new(-1.0, 0.0), Complex::new(1.0, 1.0), Complex::new(2.0, -1.0)]));
    println!("f(x) = {}", f.eval(&x, &()));
    println!("f'(x) = {}", f.grad(&x, &()));

    let g = AutoDiff::new(Monomial::new(Complex::new(1.0, 0.0)));
    println!("g(x) = {}", g.eval(&x, &()));
    println!("g'(x) = {}", g.grad(&x, &()));

    let h = f.clone().compose(g);
    println!("h(x) = {}", h.eval(&x, &()));
    println!("h'(x) = {}", h.grad(&x, &()));

    let i = f.clone() * f;
    println!("i(x) = {}", i.eval(&x, &()));
    println!("i'(x) = {}", i.grad(&x, &()));
}
