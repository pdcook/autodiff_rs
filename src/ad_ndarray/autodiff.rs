use crate::ad_ndarray::adops::*;
use crate::autodiff::AutoDiff;
use crate::diffable::Diffable;
use crate::ad_ndarray::traits::{TensorDot, TensorContraction};
use crate::ad_ndarray::func_traits;
use ndarray::linalg::Dot;
use crate::traits::{InstZero, InstOne};
use std::marker::PhantomData;
use ndarray::{ArrayBase, Dimension, DataOwned, RawDataClone};

/// Impl of Dot for AutoDiff
impl<StaticArgs, A, B> func_traits::Dot<AutoDiff<StaticArgs, B>> for AutoDiff<StaticArgs, A>
where
    A: Diffable<StaticArgs> + Clone,
    B: Diffable<StaticArgs> + Clone,
{
    type Output = AutoDiff<StaticArgs, ADDot<A, B>>;

    fn dot(&self, _other: &AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADDot(self.0.clone(), _other.0.clone()), PhantomData)
    }
}

/// Impl of constant Dot for AutoDiff
impl<StaticArgs, A, B> func_traits::Dot<B> for AutoDiff<StaticArgs, A>
where
    // ensure B is Clone and is a constant, distict from Diffable
    // to avoid conflict with the other impl
    A: Clone,
    B: Clone + InstZero + InstOne,
{
    type Output = AutoDiff<StaticArgs, ADConstantDot<A, B>>;

    fn dot(&self, _other: &B) -> Self::Output {
        AutoDiff(ADConstantDot(self.0.clone(), _other.clone()), PhantomData)
    }
}

/// Impl of constant Dot from the left
impl<StaticArgs, B, D, S, A> Dot<AutoDiff<StaticArgs, B>> for ArrayBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: DataOwned<Elem = A> + RawDataClone,
    B: Clone,
{
    type Output = AutoDiff<StaticArgs, ADConstantLeftDot<ArrayBase<S, D>, B>>;

    fn dot(&self, _other: &AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADConstantLeftDot((*self).clone(), _other.0.clone()), PhantomData)
    }
}

/// Impl of TensorDot for AutoDiff
impl<StaticArgs, A, B> func_traits::TensorDot<AutoDiff<StaticArgs, B>> for AutoDiff<StaticArgs, A>
where
    A: Diffable<StaticArgs> + Clone,
    B: Diffable<StaticArgs> + Clone,
{
    type Output = AutoDiff<StaticArgs, ADTensorDot<A, B>>;

    fn tensordot(&self, _other: &AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADTensorDot(self.0.clone(), _other.0.clone()), PhantomData)
    }
}

/// Impl of constant TensorDot for AutoDiff
impl<StaticArgs, A, B> func_traits::TensorDot<B> for AutoDiff<StaticArgs, A>
where
    // ensure B is Clone and is a constant, distict from Diffable
    // to avoid conflict with the other impl
    A: Clone,
    B: Clone + InstZero + InstOne,
{
    type Output = AutoDiff<StaticArgs, ADConstantTensorDot<A, B>>;

    fn tensordot(&self, _other: &B) -> Self::Output {
        AutoDiff(ADConstantTensorDot(self.0.clone(), _other.clone()), PhantomData)
    }
}

/// Impl of constant TensorDot from the left
impl<StaticArgs, B, D, S, A> TensorDot<AutoDiff<StaticArgs, B>> for ArrayBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: DataOwned<Elem = A> + RawDataClone,
    B: Clone,
{
    type Output = AutoDiff<StaticArgs, ADConstantLeftTensorDot<ArrayBase<S, D>, B>>;

    fn tensordot(&self, _other: &AutoDiff<StaticArgs, B>) -> Self::Output {
        AutoDiff(ADConstantLeftTensorDot((*self).clone(), _other.0.clone()), PhantomData)
    }
}

/// Impl of TensorContraction for AutoDiff, N is the number of axes to contract
impl<const N: usize, StaticArgs, A, B> func_traits::TensorContraction<N, AutoDiff<StaticArgs, B>> for AutoDiff<StaticArgs, A>
where
    A: Diffable<StaticArgs> + Clone,
    B: Diffable<StaticArgs> + Clone,
{
    type Output = AutoDiff<StaticArgs, ADTensorContraction<A, B, N>>;

    fn contract(&self, _other: &AutoDiff<StaticArgs, B>, axes: (&[usize; N], &[usize; N])) -> Self::Output {
        AutoDiff(ADTensorContraction(self.0.clone(), _other.0.clone(), (*axes.0, *axes.1)), PhantomData)
    }
}

/// Impl of constant TensorContraction for AutoDiff, N is the number of axes to contract
impl<const N: usize, StaticArgs, A, B> func_traits::TensorContraction<N, B> for AutoDiff<StaticArgs, A>
where
    // ensure B is Clone and is a constant, distict from Diffable
    // to avoid conflict with the other impl
    A: Clone,
    B: Clone + InstZero + InstOne,
{
    type Output = AutoDiff<StaticArgs, ADConstantTensorContraction<A, B, N>>;

    fn contract(&self, _other: &B, axes: (&[usize; N], &[usize; N])) -> Self::Output {
        AutoDiff(ADConstantTensorContraction(self.0.clone(), _other.clone(), (*axes.0, *axes.1)), PhantomData)
    }
}

/// Impl of constant TensorContraction from the left, N is the number of axes to contract
impl<const N: usize, StaticArgs, B, D, S, A> TensorContraction<N, AutoDiff<StaticArgs, B>> for ArrayBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: DataOwned<Elem = A> + RawDataClone,
    B: Clone,
{
    type Output = AutoDiff<StaticArgs, ADConstantLeftTensorContraction<ArrayBase<S, D>, B, N>>;

    fn contract(&self, _other: &AutoDiff<StaticArgs, B>, axes: (&[usize; N], &[usize; N])) -> Self::Output {
        AutoDiff(ADConstantLeftTensorContraction((*self).clone(), _other.0.clone(), (*axes.0, *axes.1)), PhantomData)
    }
}
