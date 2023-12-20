#![allow(dead_code)]

use crate::arithmetic::*;
use crate::autodiffable::AutoDiffable;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Identity<I>
(
    pub PhantomData<I>,
);

impl<I> Identity<I>
{
    pub fn new() -> Self {
        Identity(PhantomData)
    }
}

impl<'a, I> AutoDiffable<'a, (), I, I, I>
    for Identity<I>
where for<'b> I: StrongAssociatedArithmetic<'b, I> + WeakAssociatedArithmetic<'b, I>,
    for<'b> &'b I: CastingArithmetic<'b, I, I>,
{
    fn eval(&self, x: &I, _: &()) -> I {
        x.clone()
    }

    fn grad(&self, x: &I, _: &()) -> I {
        x.one()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Constant<I, O>(pub O, pub PhantomData<I>);

impl<I, O> Constant<I, O> {
    pub fn new(x: O) -> Self {
        Constant(x, PhantomData)
    }
}

impl<'a, I, O>
    AutoDiffable<'a, (), I, O, O> for Constant<I, O>
where
    for<'b> I: Arithmetic<'b>,
    for<'b> O: StrongAssociatedArithmetic<'b, O> + WeakAssociatedArithmetic<'b, O>,
    for<'b> &'b I: CastingArithmetic<'b, I, I>,
    for<'b> &'b O: CastingArithmetic<'b, O, O>,
{
    fn eval(&self, _: &I, _: &()) -> O {
        self.0.clone()
    }

    fn grad(&self, _: &I, _: &()) -> O {
        self.0.zero()
    }
}

#[derive(Debug, Clone)]
pub struct Polynomial<I, O>(pub Vec<O>, pub PhantomData<I>);

impl<I, O> Polynomial<I, O> {
    pub fn new(coeffs: Vec<O>) -> Self {
        Polynomial(coeffs, PhantomData)
    }
}

impl<'a,
        I,
        O,
    > AutoDiffable<'a, (), I, O, O> for Polynomial<I, O>
where
    for<'b> I: Arithmetic<'b>,
    for<'b> O: StrongAssociatedArithmetic<'b, I> + WeakAssociatedArithmetic<'b, O> + StrongAssociatedArithmetic<'b, O>,
    for<'b> &'b I: CastingArithmetic<'b, I, I> + CastingArithmetic<'b, O, O>,
    for<'b> &'b O: CastingArithmetic<'b, O, O> + CastingArithmetic<'b, I, O>,
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

    fn grad(&self, x: &I, _: &()) -> O {
        let mut res = self.0[0].zero();
        let mut x_pow = self.0[0].one();
        let mut pow = self.0[0].one();
        for c in &self.0[1..] {
            res = res + c * &pow * &x_pow;
            pow = &pow + self.0[0].one();
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

impl<'a,
        I,
        P,
    > AutoDiffable<'a, (), I, I, I> for Monomial<I, P>
where
    for<'b> I: ExtendedArithmetic<'b> + StrongAssociatedExtendedArithmetic<'b, P>,
    for<'b> P: WeakAssociatedExtendedArithmetic<'b, I>,
    for<'b> &'b I: CastingArithmetic<'b, I, I> + CastingArithmetic<'b, P, I>,
    for<'b> &'b P: CastingArithmetic<'b, P, P> + CastingArithmetic<'b, I, I>,
{
    fn eval(&self, x: &I, _: &()) -> I {
        x.clone().pow(&self.0)
    }

    fn grad(&self, x: &I, _: &()) -> I {
        &self.0 * x.clone().pow(&self.0 - self.0.one())
    }
}
