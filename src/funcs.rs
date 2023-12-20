#![allow(dead_code)]

use std::marker::PhantomData;
use crate::arithmetic::{Arithmetic, StrongAssociatedArithmetic, WeakAssociatedArithmetic, ExtendedArithmetic, StrongAssociatedExtendedArithmetic, WeakAssociatedExtendedArithmetic};
use crate::autodiffable::AutoDiffable;

#[derive(Debug, Clone, Copy)]
pub struct Identity<I>(pub PhantomData<I>);

impl<I> Identity<I> {
    pub fn new() -> Self {
        Identity(PhantomData)
    }
}

impl<I: StrongAssociatedArithmetic<I> + WeakAssociatedArithmetic<I>> AutoDiffable<()>
    for Identity<I>
{
    type InType = I;
    type OutType = I;
    type GradType = I;
    fn eval(&self, x: I, _: &()) -> I {
        x
    }

    fn grad(&self, _: I, _: &()) -> I {
        I::one()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Constant<I, O>(pub O, pub PhantomData<I>);

impl<I, O> Constant<I, O> {
    pub fn new(x: O) -> Self {
        Constant(x, PhantomData)
    }
}

impl<I: Arithmetic, O: StrongAssociatedArithmetic<O> + WeakAssociatedArithmetic<O>>
    AutoDiffable<()> for Constant<I, O>
{
    type InType = I;
    type OutType = O;
    type GradType = O;
    fn eval(&self, _: I, _: &()) -> O {
        self.0
    }

    fn grad(&self, _: I, _: &()) -> O {
        O::zero()
    }
}

#[derive(Debug, Clone)]
pub struct Polynomial<I, O>(pub Vec<O>, pub PhantomData<I>);

impl<I: Arithmetic, O: StrongAssociatedArithmetic<I> + StrongAssociatedArithmetic<O> + WeakAssociatedArithmetic<O>>
    Polynomial<I, O>
{
    pub fn new(coeffs: Vec<O>) -> Self {
        Polynomial(coeffs, PhantomData)
    }
}

impl<I: Arithmetic, O: StrongAssociatedArithmetic<I> + StrongAssociatedArithmetic<O> + WeakAssociatedArithmetic<O>>
    AutoDiffable<()> for Polynomial<I, O>
{
    type InType = I;
    type OutType = O;
    type GradType = O;
    fn eval(&self, x: I, _: &()) -> O {
        let mut res = O::zero();
        let mut x_pow = O::one();
        for c in &self.0 {
            res = res + *c * x_pow;
            x_pow = x_pow * x;
        }
        res
    }

    fn grad(&self, x: I, _: &()) -> O {
        let mut res = O::zero();
        let mut x_pow = O::one();
        let mut pow = O::one();
        for c in &self.0[1..] {
            res = res + *c * pow * x_pow;
            pow = pow + O::one();
            x_pow = x_pow * x;
        }
        res
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Monomial<I, P>(pub P, pub PhantomData<I>);

impl<I: ExtendedArithmetic + StrongAssociatedExtendedArithmetic<P>, P: WeakAssociatedExtendedArithmetic<I>> Monomial<I, P> {
    pub fn new(p: P) -> Self {
        Monomial(p, PhantomData)
    }
}

impl<I: ExtendedArithmetic + StrongAssociatedExtendedArithmetic<P>, P: WeakAssociatedExtendedArithmetic<I>> AutoDiffable<()> for Monomial<I, P> {
    type InType = I;
    type OutType = I;
    type GradType = I;
    fn eval(&self, x: I, _: &()) -> I {
        x.pow(self.0)
    }

    fn grad(&self, x: I, _: &()) -> I {
        self.0 * x.pow(self.0 - I::one())
    }
}
