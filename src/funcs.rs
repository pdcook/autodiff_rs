#![allow(dead_code)]

use crate::autodiffable::AutoDiffable;
use crate::traits::{InstOne, InstZero};
use num::traits::Pow;
use std::marker::PhantomData;
use std::ops::{Mul, Sub};

#[cfg(test)]
use crate::autodiff::AutoDiff;

#[derive(Debug, Clone)]
pub struct Identity<I>(pub PhantomData<I>);

impl<I: Clone> Copy for Identity<I> {}

impl<I> Identity<I> {
    pub fn new() -> Self {
        Identity(PhantomData)
    }
}

impl<I> Default for Identity<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Clone + InstOne> AutoDiffable<(), I, I, I, I> for Identity<I> {
    fn eval(&self, x: &I, _: &()) -> I {
        x.clone()
    }

    fn eval_grad(&self, x: &I, dx: &I, s: &()) -> (I, I) {
        (self.eval(x, s), dx.clone())
    }
}

#[test]
fn test_identity() {
    let x = 2.0;
    let dx = 3.3;
    let id = AutoDiff::new(Identity::new());
    assert_eq!(id.eval(&x, &()), x);
    assert_eq!(id.eval_grad(&x, &dx, &()), (x, dx));
}

#[derive(Debug, Clone)]
pub struct Constant<I, O>(pub O, pub PhantomData<I>);

impl<I: Clone, O: Copy> Copy for Constant<I, O> {}

impl<I, O> Constant<I, O> {
    pub fn new(x: O) -> Self {
        Constant(x, PhantomData)
    }
}

impl<I, O: InstOne + InstZero + Clone> AutoDiffable<(), I, O, O, I> for Constant<I, O> {
    fn eval(&self, _: &I, _: &()) -> O {
        self.0.clone()
    }

    fn eval_grad(&self, x: &I, _: &I, s: &()) -> (O, O) {
        (self.eval(x, s), self.0.zero())
    }
}

#[test]
fn test_constant() {
    let x = 2.0;
    let dx = 3.3;
    let c = AutoDiff::new(Constant::new(3.0));
    assert_eq!(c.eval(&x, &()), 3.0);
    assert_eq!(c.eval_grad(&x, &dx, &()), (3.0, 0.0));
}

#[derive(Debug, Clone)]
pub struct Polynomial<I, O>(pub Vec<O>, pub PhantomData<I>);

impl<I, O> Polynomial<I, O> {
    pub fn new(coeffs: Vec<O>) -> Self {
        Polynomial(coeffs, PhantomData)
    }
}

impl<I, O: InstZero + InstOne> AutoDiffable<(), I, O, O, I> for Polynomial<I, O>
where
    for<'b> O: Mul<&'b O, Output = O>,
    for<'b> &'b I: Mul<&'b O, Output = O>,
    for<'b> &'b O: Mul<&'b I, Output = O> + Mul<&'b O, Output = O>,
{
    fn eval(&self, x: &I, _: &()) -> O {
        let mut res = self.0[0].zero();
        let mut x_pow = self.0[0].one();
        for c in &self.0 {
            res = res + c * &x_pow;
            x_pow = &x_pow * x;
        }
        res
    }

    fn eval_grad(&self, x: &I, dx: &I, _: &()) -> (O, O) {
        let mut res = self.0[0].zero();
        let mut grad = self.0[0].zero();
        let mut x_pow = self.0[0].one();
        let mut pow = self.0[0].zero();

        for i in 0..self.0.len() {
            res = res + &self.0[i] * &x_pow;
            if i < self.0.len() - 1 {
                pow = pow + self.0[0].one();
                grad = grad + &self.0[i + 1] * &pow * (&x_pow * dx);
            }
            x_pow = &x_pow * x;
        }

        (res, grad)
    }
}

#[test]
fn test_polynomial() {
    // p(x) = 3 + 2x + x^2
    // p'(x) = 2dx + 2xdx
    // p(2) = 3 + 4 + 4 = 11
    // p'(2) = 2dx + 4dx = 6dx

    let x = 2.0;
    let dx = 1.0;
    let dx2 = 2.0;
    let p = AutoDiff::new(Polynomial::new(vec![3.0, 2.0, 1.0]));
    assert_eq!(p.eval(&x, &()), 11.0);
    assert_eq!(p.eval_grad(&x, &dx, &()), (11.0, 6.0));
    assert_eq!(p.eval_grad(&x, &dx2, &()), (11.0, 12.0));
}

#[derive(Debug, Clone)]
pub struct Monomial<I, P>(pub P, pub PhantomData<I>);

impl<I: Clone, P: Copy> Copy for Monomial<I, P> {}

impl<I, P> Monomial<I, P> {
    pub fn new(p: P) -> Self {
        Monomial(p, PhantomData)
    }
}

impl<I: Clone + InstOne + Pow<P, Output = I> + Mul<I, Output = I>, P: InstOne>
    AutoDiffable<(), I, I, I, I> for Monomial<I, P>
where
    for<'b> I: Mul<&'b I, Output = I> + Mul<&'b P, Output = I> + Pow<&'b P, Output = I>,
    for<'b> &'b I: Mul<&'b I, Output = I>,
    for<'b> &'b P: Sub<&'b P, Output = P> + Sub<P, Output = P>,
{
    fn eval(&self, x: &I, _: &()) -> I {
        x.clone().pow(&self.0)
    }

    fn eval_grad(&self, x: &I, dx: &I, _: &()) -> (I, I) {
        let x_pow = x.clone().pow(&self.0 - self.0.one());
        (&x_pow * x, x_pow * &self.0 * dx)
    }
}

#[test]
fn test_monomial() {
    // p(x) = x^3
    // p'(x) = 3x^2dx
    // p(2) = 8
    // p'(2) = 12dx

    let x = 2.0;
    let dx = 1.0;
    let dx2 = 2.0;
    let p = AutoDiff::new(Monomial::new(3.0));
    assert_eq!(p.eval(&x, &()), 8.0);
    assert_eq!(p.eval_grad(&x, &dx, &()), (8.0, 12.0));
    assert_eq!(p.eval_grad(&x, &dx2, &()), (8.0, 24.0));
}
