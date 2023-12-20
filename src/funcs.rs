#![allow(dead_code)]

use crate::arithmetic::*;
use crate::autodiffable::AutoDiffable;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Identity<I: StrongAssociatedArithmetic<I> + WeakAssociatedArithmetic<I>>(
    pub PhantomData<I>,
)
where
    for<'a> &'a I: CastingArithmetic<I, I>;

impl<I: StrongAssociatedArithmetic<I> + WeakAssociatedArithmetic<I>> Identity<I>
where
    for<'a> &'a I: CastingArithmetic<I, I>,
{
    pub fn new() -> Self {
        Identity(PhantomData)
    }
}

impl<I: StrongAssociatedArithmetic<I> + WeakAssociatedArithmetic<I>> AutoDiffable<(), I, I, I>
    for Identity<I>
where
    for<'a> &'a I: CastingArithmetic<I, I>,
{
    fn eval(&self, x: &I, _: &()) -> I {
        x.clone()
    }

    fn grad(&self, _: &I, _: &()) -> I {
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
    AutoDiffable<(), I, O, O> for Constant<I, O>
where
    for<'a> &'a I: CastingArithmetic<I, I>,
    for<'a> &'a O: CastingArithmetic<O, O>,
{
    fn eval(&self, _: &I, _: &()) -> O {
        self.0.clone()
    }

    fn grad(&self, _: &I, _: &()) -> O {
        O::zero()
    }
}

#[derive(Debug, Clone)]
pub struct Polynomial<I, O>(pub Vec<O>, pub PhantomData<I>);

impl<I, O> Polynomial<I, O> {
    pub fn new(coeffs: Vec<O>) -> Self {
        Polynomial(coeffs, PhantomData)
    }
}

impl<
        I: Arithmetic,
        O: StrongAssociatedArithmetic<I> + StrongAssociatedArithmetic<O> + WeakAssociatedArithmetic<O>,
    > AutoDiffable<(), I, O, O> for Polynomial<I, O>
where
    for<'a> &'a I: CastingArithmetic<I, I> + CastingArithmetic<O, O>,
    for<'a> &'a O: CastingArithmetic<O, O> + CastingArithmetic<I, O>,
{
    fn eval(&self, x: &I, _: &()) -> O {
        let mut res = O::zero();
        let mut x_pow = O::one();
        for c in &self.0 {
            res = res + c * &x_pow;
            x_pow = &x_pow * x;
        }
        res
    }

    fn grad(&self, x: &I, _: &()) -> O {
        let mut res = O::zero();
        let mut x_pow = O::one();
        let mut pow = O::one();
        for c in &self.0[1..] {
            res = res + c * &pow * &x_pow;
            pow = &pow + O::one();
            x_pow = &x_pow * x;
        }
        res
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Monomial<I, P>(pub P, pub PhantomData<I>);

impl<I, P> Monomial<I, P> {
    pub fn new(p: P) -> Self {
        Monomial(p, PhantomData)
    }
}

impl<
        I: ExtendedArithmetic + StrongAssociatedExtendedArithmetic<P>,
        P: WeakAssociatedExtendedArithmetic<I>,
    > AutoDiffable<(), I, I, I> for Monomial<I, P>
where
    for<'a> &'a I: CastingArithmetic<I, I> + CastingArithmetic<P, I>,
    for<'a> &'a P: CastingArithmetic<P, P> + CastingArithmetic<I, I>,
{
    fn eval(&self, x: &I, _: &()) -> I {
        x.clone().pow(&self.0)
    }

    fn grad(&self, x: &I, _: &()) -> I {
        &self.0 * x.clone().pow(&self.0 - I::one())
    }
}
