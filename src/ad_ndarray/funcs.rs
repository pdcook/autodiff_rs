#![allow(dead_code)]

use crate::ad_ndarray::scalar::*;
use crate::diffable::*;
use crate::autodiffable::*;
use crate::autotuple::*;
use crate::gradienttype::GradientType;
use ndarray::prelude::*;
use ndarray::{ArrayBase, OwnedRepr, DataOwned, Dimension, Dim, IxDyn, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, LinalgScalar};
use ndarray_einsum_beta::{ArrayLike, einsum, Contraction};
use num::Float;
use std::marker::PhantomData;

use crate as autodiff;
use autodiff_derive::*;

#[cfg(test)]
use crate::autodiff::*;

#[derive(Debug, Clone, FuncCompose)]
pub struct Einsum<'a, A: LinalgScalar, OutDim: Dimension, const N: usize>(pub PhantomData<(&'a (), A, OutDim)>);
// N is the number of arrays to einsum, OutDim is the output dimension of the einsum
// which can be Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, or IxDyn


impl<'a, StaticArgs, A: LinalgScalar, OutDim: Dimension, const N: usize> Diffable<StaticArgs> for Einsum<'a, A, OutDim, N>
{
    type Input = [&'a dyn ArrayLike<A>; N];
    type Output = ArrayBase<OwnedRepr<A>, OutDim>;
}

/// the derivative of a general einsum is straightforward by example
///
/// R = einsum("[Astr],[Bstr],[Cstr],...->[Rstr]", A, B, C, ...)
/// then
/// dR/dA = einsum("[Rstr],[Bstr],[Cstr],...->[Astr]", ones_like(R), B, C, ...)
/// dR/dB = einsum("[Astr],[Rstr],[Cstr],...->[Bstr]", A, ones_like(R), C, ...)
/// ...
///
/// that is, flip the position of the result and the derivative array in the einsum string
/// and replace the original array with ones_like(R)
///
/// NOTE: this might only be true for propagation (forward) instead of for the full derivative
///
/// for the full derivative, ones_like(R) would be instead the gradient identity of R, i.e. the
/// identity tensor of rank 2*rank(R)
///
///
/// source: https://stackoverflow.com/questions/43686534/how-does-tf-einsum-in-tensorflow-calculates-gradients-for-matrix-multiplicatio
///
///

impl<'a, StaticArgs, A: LinalgScalar, OutDim: Dimension, const N: usize> AutoDiffable<StaticArgs> for Einsum<'a, A, OutDim, N>
{
}
