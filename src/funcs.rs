#![allow(dead_code)]

use crate::autodiffable::*;
use crate::gradienttype::GradientType;
use crate::traits::{GradientIdentity, InstOne, InstZero};
use num::traits::Pow;
use std::marker::PhantomData;
use std::ops::{Mul, Sub};

use crate as autodiff;
use autodiff_derive::*;

#[cfg(test)]
use crate::autodiff::AutoDiff;

#[derive(Debug, Clone, FuncCompose)]
pub struct Identity<S, I>(pub PhantomData<(S, I)>);

impl<S: Clone, I: Clone> Copy for Identity<S, I> {}

impl<S, I> Identity<S, I> {
    pub fn new() -> Self {
        Identity(PhantomData)
    }
}

impl<S, I> Default for Identity<S, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, I> Diffable<S> for Identity<S, I> {
    type Input = I;
    type Output = I;
}

impl<S, I: Clone + InstOne + GradientType<I, GradientType = G> + GradientIdentity, G>
    AutoDiffable<S> for Identity<S, I>
{
    //type Input = I;
    //type Output = I;
    fn eval(&self, x: &I, _: &S) -> I {
        x.clone()
    }

    fn eval_grad(&self, x: &I, s: &S) -> (I, G) {
        (self.eval(x, s), x.grad_identity())
    }
}

impl<S, I: Clone + InstOne + InstZero + GradientType<I> + GradientIdentity> ForwardDiffable<S>
    for Identity<S, I>
{
    //type Input = I;
    //type Output = I;
    fn eval_forward_grad(&self, x: &I, dx: &I, s: &S) -> (I, I) {
        (self.eval(x, s), dx.clone())
    }
}

#[test]
fn test_identity() {
    let x = 2.0;
    let dx = 3.3;
    let id = AutoDiff::new(Identity::new());
    assert_eq!(id.eval(&x, &()), x);
    assert_eq!(id.eval_grad(&x, &()), (x, 1.0));
    assert_eq!(id.eval_forward_grad(&x, &dx, &()), (x, dx));
}

#[derive(Debug, Clone, FuncCompose)]
pub struct Polynomial<S, I, O>(pub Vec<O>, pub PhantomData<(S, I)>);

impl<S, I, O> Polynomial<S, I, O> {
    pub fn new(coeffs: Vec<O>) -> Self {
        Polynomial(coeffs, PhantomData)
    }
}

impl<S, I, O> Diffable<S> for Polynomial<S, I, O> {
    type Input = I;
    type Output = O;
}

impl<S, I: GradientType<O, GradientType = O>, O: InstZero + InstOne> AutoDiffable<S>
    for Polynomial<S, I, O>
where
    for<'b> O: Mul<&'b O, Output = O>,
    for<'b> &'b I: Mul<&'b O, Output = O>,
    for<'b> &'b O: Mul<&'b I, Output = O> + Mul<&'b O, Output = O>,
{
    //type Input = I;
    //type Output = O;

    fn eval(&self, x: &I, _: &S) -> O {
        let mut res = self.0[0].zero();
        let mut x_pow = self.0[0].one();
        for c in &self.0 {
            res = res + c * &x_pow;
            x_pow = &x_pow * x;
        }
        res
    }

    fn eval_grad(&self, x: &I, _: &S) -> (O, O) {
        let mut res = self.0[0].zero();
        let mut grad = self.0[0].zero();
        let mut x_pow = self.0[0].one();
        let mut pow = self.0[0].zero();

        for i in 0..self.0.len() {
            res = res + &self.0[i] * &x_pow;
            if i < self.0.len() - 1 {
                pow = pow + self.0[0].one();
                grad = grad + &self.0[i + 1] * &pow * &x_pow;
            }
            x_pow = &x_pow * x;
        }

        (res, grad)
    }
}

impl<
        S,
        I: Clone + GradientType<O, GradientType = O>,
        O: InstZero + InstOne + Mul<I, Output = O>,
    > ForwardDiffable<S> for Polynomial<S, I, O>
where
    for<'b> O: Mul<&'b O, Output = O>,
    for<'b> &'b I: Mul<&'b O, Output = O>,
    for<'b> &'b O: Mul<&'b I, Output = O> + Mul<&'b O, Output = O>,
{
    //type Input = I;
    //type Output = O;
    fn eval_forward_grad(&self, x: &I, dx: &I, _: &S) -> (O, O) {
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
    assert_eq!(p.eval_forward_grad(&x, &dx, &()), (11.0, 6.0));
    assert_eq!(p.eval_forward_grad(&x, &dx2, &()), (11.0, 12.0));
}

#[derive(Debug, Clone, FuncCompose)]
pub struct Monomial<S, I, P>(pub P, pub PhantomData<(S, I)>);

impl<S: Clone, I: Clone, P: Copy> Copy for Monomial<S, I, P> {}

impl<S, I, P> Monomial<S, I, P> {
    pub fn new(p: P) -> Self {
        Monomial(p, PhantomData)
    }
}

impl<S, I, P> Diffable<S> for Monomial<S, I, P> {
    type Input = I;
    type Output = I;
}

impl<
        S,
        I: Clone
            + InstOne
            + Pow<P, Output = I>
            + Mul<I, Output = I>
            + GradientType<I, GradientType = I>,
        P: InstOne,
    > AutoDiffable<S> for Monomial<S, I, P>
where
    for<'b> I: Mul<&'b I, Output = I> + Mul<&'b P, Output = I> + Pow<&'b P, Output = I>,
    for<'b> &'b I: Mul<&'b I, Output = I>,
    for<'b> &'b P: Sub<&'b P, Output = P> + Sub<P, Output = P>,
{
    //type Input = I;
    //type Output = I;
    fn eval(&self, x: &I, _: &S) -> I {
        x.clone().pow(&self.0)
    }

    fn eval_grad(&self, x: &I, _: &S) -> (I, I) {
        let x_pow = x.clone().pow(&self.0 - self.0.one());
        (&x_pow * x, x_pow * &self.0)
    }
}

impl<
        S,
        I: Clone
            + InstOne
            + Pow<P, Output = I>
            + Mul<I, Output = I>
            + GradientType<I, GradientType = I>,
        P: InstOne,
    > ForwardDiffable<S> for Monomial<S, I, P>
where
    for<'b> I: Mul<&'b I, Output = I> + Mul<&'b P, Output = I> + Pow<&'b P, Output = I>,
    for<'b> &'b I: Mul<&'b I, Output = I>,
    for<'b> &'b P: Sub<&'b P, Output = P> + Sub<P, Output = P>,
{
    //type Input = I;
    //type Output = I;
    fn eval_forward_grad(&self, x: &I, dx: &I, _: &S) -> (I, I) {
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

    let x = 2.0_f64;
    let dx = 1.0_f64;
    let dx2 = 2.0_f64;
    let p = AutoDiff::new(Monomial::<(), f64, f64>::new(3.0));
    assert_eq!(p.eval(&x, &()), 8.0);
    assert_eq!(p.eval_grad(&x, &()), (8.0, 12.0));
    assert_eq!(p.eval_forward_grad(&x, &dx, &()), (8.0, 12.0));
    assert_eq!(p.eval_forward_grad(&x, &dx2, &()), (8.0, 24.0));
}
