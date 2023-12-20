#![allow(dead_code)]

use crate::autodiffable::AutoDiffable;
use crate::complex::arithmetic::*;
use crate::funcs::*;
use num::traits::Float;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct ComplexIdentity<F: Float, I>(pub PhantomData<(F, I)>);

impl<F: Float, I> ComplexIdentity<F, I> {
    pub fn new() -> Self {
        ComplexIdentity(PhantomData)
    }
}

impl<
        F: Float,
        I: ComplexStrongAssociatedArithmetic<F, F, I> + ComplexWeakAssociatedArithmetic<F, F, I>,
    > AutoDiffable<()> for ComplexIdentity<F, I>
{
    type InType = I;
    type OutType = I;
    type GradType = I;
    fn eval(&self, x: I, _: &()) -> I {
        x
    }

    fn grad(&self, _: I, _: &()) -> I {
        I::one() + I::one()
    }
}
