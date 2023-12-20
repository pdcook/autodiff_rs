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
//use crate::func_traits::Compose;
use crate::funcs::*;
use num::complex::Complex;

// ================================================= //


fn main() {

    let z = Complex::new(1.0_f64, 2.0_f64);

    let f = AutoDiff::new(Identity::new());
    println!("f(z) = {}", f.eval(z, &()));
    println!("f'(z) = {}", f.grad(z, &()));

    let g = f * f;
    println!("g(z) = {}", g.eval(z, &()));
    println!("g'(z) = {}", g.grad(z, &()));

    /*
    let x = 2.0_f64;

    let f = AutoDiff::new(Polynomial::new(vec![-1.0, 2.0, 5.0, 6.6, -10.0]));
    println!("f(x) = {}", f.eval(x, &()));
    println!("f'(x) = {}", f.grad(x, &()));

    let g = AutoDiff::new(Monomial::new(1.0000001));
    println!("g(x) = {}", g.eval(x, &()));
    println!("g'(x) = {}", g.grad(x, &()));

    let h = f.compose(g);



    println!("h(x) = {}", h.eval(x, &()));
    println!("h'(x) = {}", h.grad(x, &()));
    */
}
