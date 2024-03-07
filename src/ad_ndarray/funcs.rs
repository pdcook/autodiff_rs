#![allow(dead_code)]

use ndarray::prelude::*;
use ndarray::*;
use crate::autodiffable::*;
use crate::gradienttype::GradientType;
use std::marker::PhantomData;
use crate::autotuple::*;
use num::Float;
use crate::ad_ndarray::scalar::*;
use ndarray_einsum_beta::einsum;

use crate as autodiff;
use autodiff_derive::*;

#[cfg(test)]
use crate::autodiff::*;


#[derive(Debug, Clone, FuncCompose, SimpleForwardDiffable)]
/// The differentiable dot product of two 1D or 2D arrays with dimensions D1
/// and D2 respectively and datatype A.
pub struct Dot<S, D1, D2, A>(pub PhantomData<(S, D1, D2, A)>);

// impl copy
impl<S: Clone, D1: Clone, D2: Clone, A: Clone> Copy for Dot<S, D1, D2, A> {}

impl<S, D1: Dimension, D2: Dimension, A: LinalgScalar> Dot<S, D1, D2, A> {
    pub fn new() -> Self {
        Dot(PhantomData)
    }
}

impl<S, D1: Dimension, D2: Dimension, A: LinalgScalar> Default for Dot<S, D1, D2, A> {
    fn default() -> Self {
        Self::new()
    }
}

// 1D dot 1D
impl<S, A: LinalgScalar + PartialEq> Diffable<S> for Dot<S, Ix1, Ix1, A> {
    type Input = AutoTuple<(Array<A, Ix1>, Array<A, Ix1>)>;
    type Output = AutoTuple<(Array<A, Ix0>,)>;
}

// 1D dot 2D
impl<S, A: LinalgScalar + PartialEq> Diffable<S> for Dot<S, Ix1, Ix2, A> {
    type Input = AutoTuple<(Array<A, Ix1>, Array<A, Ix2>)>;
    type Output = AutoTuple<(Array<A, Ix1>,)>;
}

// 2D dot 1D
impl<S, A: LinalgScalar + PartialEq> Diffable<S> for Dot<S, Ix2, Ix1, A> {
    type Input = AutoTuple<(Array<A, Ix2>, Array<A, Ix1>)>;
    type Output = AutoTuple<(Array<A, Ix1>,)>;
}

// 2D dot 2D
impl<S, A: LinalgScalar + PartialEq> Diffable<S> for Dot<S, Ix2, Ix2, A> {
    type Input = AutoTuple<(Array<A, Ix2>, Array<A, Ix2>)>;
    type Output = AutoTuple<(Array<A, Ix2>,)>;
}

/// 1D dot 1D for real numbers
impl<S, A: LinalgScalar + Float + GradientType<A, GradientType = A>> AutoDiffable<S> for Dot<S, Ix1, Ix1, A>
where
    <A as GradientType<A>>::GradientType: Clone + PartialEq,
{
    fn eval(&self, input: &Self::Input, _: &S) -> Self::Output {
        let (a, b) = (**input).clone();
        AutoTuple::from(Scalar::new(a.t().dot(&b)),)
    }

    fn eval_grad(&self, input: &Self::Input, _: &S) -> (Self::Output, <Self::Input as GradientType<Self::Output>>::GradientType) {
        let (a, b) = (**input).clone();
        let res = AutoTuple::from(Scalar::new(a.t().dot(&b)));
        let grad = AutoTuple::from((b, a));
        (res, grad)
    }
}

/// 1D dot 2D for real numbers
impl<S, A: LinalgScalar + Float + GradientType<A, GradientType = A>> AutoDiffable<S> for Dot<S, Ix1, Ix2, A>
where
    <A as GradientType<A>>::GradientType: Clone + PartialEq,
{
    fn eval(&self, input: &Self::Input, _: &S) -> Self::Output {
        let (a, b) = (**input).clone();
        AutoTuple::from(a.t().dot(&b))
    }

    fn eval_grad(&self, input: &Self::Input, _: &S) -> (Self::Output, <Self::Input as GradientType<Self::Output>>::GradientType) {
        let (a, b) = (**input).clone();
        let res = AutoTuple::from(a.t().dot(&b));
        // d(aT B)_j/dB_ik = delta_jk a_i
        let grad_b = einsum("jk,i->ikj", &[&Array::eye(b.shape()[0]), &a]).unwrap().into_dimensionality::<Ix3>().unwrap();
        let grad = AutoTuple::from((b, grad_b));
        (res, grad)
    }
}

/// 2D dot 1D for real numbers
impl<S, A: LinalgScalar + Float + GradientType<A, GradientType = A>> AutoDiffable<S> for Dot<S, Ix2, Ix1, A>
where
    <A as GradientType<A>>::GradientType: Clone + PartialEq,
{
    fn eval(&self, input: &Self::Input, _: &S) -> Self::Output {
        let (a, b) = (**input).clone();
        AutoTuple::from(a.dot(&b))
    }

    fn eval_grad(&self, input: &Self::Input, _: &S) -> (Self::Output, <Self::Input as GradientType<Self::Output>>::GradientType) {
        let (a, b) = (**input).clone();
        let res = AutoTuple::from(a.dot(&b));
        // Ab_i = A_ij b_j
        // d(Ab)_i/dA_kl = delta_ik delta_jl b_j = delta_ik b_j
        let grad_a = einsum("ik,j->kji", &[&Array::eye(a.shape()[0]), &b]).unwrap().into_dimensionality::<Ix3>().unwrap();
        let grad = AutoTuple::from((grad_a, a));
        (res, grad)
    }
}

/// 2D dot 2D for real numbers
impl<S, A: LinalgScalar + Float + GradientType<A, GradientType = A>> AutoDiffable<S> for Dot<S, Ix2, Ix2, A>
where
    <A as GradientType<A>>::GradientType: Clone + PartialEq,
{
    fn eval(&self, input: &Self::Input, _: &S) -> Self::Output {
        let (a, b) = (**input).clone();
        AutoTuple::from(a.dot(&b))
    }

    fn eval_grad(&self, input: &Self::Input, _: &S) -> (Self::Output, <Self::Input as GradientType<Self::Output>>::GradientType) {
        let (a, b) = (**input).clone();
        let res = AutoTuple::from(a.dot(&b));
        // AB_ij = A_ik B_kj
        // d(AB)_ij/dA_lm = delta_il delta_km B_kj = delta_il B_kj
        // d(AB)_ij/dB_lm = A_ik delta_kl delta_jm = delta_jm A_ik
        let grad_a = einsum("il,kj->likj", &[&Array::eye(a.shape()[0]), &b]).unwrap().into_dimensionality::<Ix4>().unwrap();
        let grad_b = einsum("jm,ik->jmik", &[&Array::eye(b.shape()[0]), &a]).unwrap().into_dimensionality::<Ix4>().unwrap();
        let grad = AutoTuple::from((grad_a, grad_b));
        (res, grad)
    }
}

#[test]
fn test_dot() {
    let a_mat = arr2(&[[1.0_f64, 2.], [3., 4.]]);
    let a_vec = arr1(&[1.0_f64, 2.]);
    let b_mat = arr2(&[[5.0_f64, 6.], [7., 8.]]);
    let b_vec = arr1(&[5.0_f64, 6.]);

    let dot_mat_mat = AutoDiff::new(Dot::<(), Ix2, Ix2, f64>::new());
    let dot_mat_vec = AutoDiff::new(Dot::<(), Ix2, Ix1, f64>::new());
    let dot_vec_mat = AutoDiff::new(Dot::<(), Ix1, Ix2, f64>::new());
    let dot_vec_vec = AutoDiff::new(Dot::<(), Ix1, Ix1, f64>::new());

    let input_mat_mat = AutoTuple::from((a_mat.clone(), b_mat.clone()));
    let input_mat_vec = AutoTuple::from((a_mat.clone(), b_vec.clone()));
    let input_vec_mat = AutoTuple::from((a_vec.clone(), b_mat.clone()));
    let input_vec_vec = AutoTuple::from((a_vec.clone(), b_vec.clone()));

    let (res_mat_mat, grad_mat_mat) = dot_mat_mat.eval_forward_grad(&input_mat_mat, &());
    let (res_mat_vec, grad_mat_vec) = dot_mat_vec.eval_forward_grad(&input_mat_vec, &());
    let (res_vec_mat, grad_vec_mat) = dot_vec_mat.eval_forward_grad(&input_vec_mat, &());
    let (res_vec_vec, grad_vec_vec) = dot_vec_vec.eval_forward_grad(&input_vec_vec, &());


    assert_eq!(res_mat_mat, AutoTuple::from(a_mat.dot(&b_mat)));
    assert_eq!(res_mat_vec, AutoTuple::from(a_mat.dot(&b_vec)));
    assert_eq!(res_vec_mat, AutoTuple::from(a_vec.dot(&b_mat)));
    assert_eq!(res_vec_vec, AutoTuple::from(Scalar::new(a_vec.dot(&b_vec))));

    assert_eq!(grad_mat_mat, AutoTuple::from((b_mat.clone(), a_mat.clone())));



}
