use crate::ad_ndarray::dimabssub::DimAbsSub;
use crate::autotuple::AutoTuple;
use crate::forward::ForwardMul;
use crate::gradienttype::GradientType;
use crate::traits::{
    Abs, AbsSqr, Arg, Conjugate, GradientIdentity, InstOne, InstZero, PossiblyComplex, Signum,
};
use ndarray::{
    ArrayBase, Axis, DataOwned, DimAdd, DimMax, Dimension, IxDyn, LinalgScalar, OwnedRepr,
    RawDataClone,
};
use ndarray_einsum_beta;
use num::traits::{One, Zero};
use std::ops::{Add, Mul};

#[cfg(test)]
use ndarray::{arr1, arr2, Array0, Array1, Array2, Dim};

impl<A, S, D> InstZero for ArrayBase<S, D>
where
    D: Dimension,
    S: DataOwned<Elem = A>,
    A: Clone + InstZero + Zero,
    Self: Sized + Add<Self, Output = Self>,
{
    fn zero(&self) -> Self {
        Self::zeros(self.dim())
    }

    fn is_zero(&self) -> bool {
        self.iter().all(|x| Zero::is_zero(x))
    }
}

impl<A, S, D> InstOne for ArrayBase<S, D>
where
    D: Dimension,
    S: DataOwned<Elem = A>,
    A: Clone + InstOne + One,
    Self: Sized + Mul<Self, Output = Self>,
{
    fn one(&self) -> Self {
        Self::ones(self.dim())
    }
}

impl<AI, DI, AG, DG> GradientIdentity for ArrayBase<OwnedRepr<AI>, DI>
where
    DI: Dimension,
    DG: Dimension,
    AI: Clone + GradientType<AI, GradientType = AG>,
    AG: Clone + InstOne + One + Zero,
    Self: Sized + GradientType<Self, GradientType = ArrayBase<OwnedRepr<AG>, DG>>,
{
    fn grad_identity(&self) -> ArrayBase<OwnedRepr<AG>, DG> {
        // for an input with shape (a, b, ...) ndim
        // the gradient identity is a tensor with shape (a, b, ..., a, b, ...) 2ndim
        // where g[a, b, ..., z, y, ...] = 1 if a == z && b == y && ... else 0

        // first, we need to get the shape of the gradient
        let grad_shape = self
            .shape()
            .iter()
            .chain(self.shape().iter())
            .map(|x| *x)
            .collect::<Vec<_>>();

        // make the gradient
        let mut grad: ArrayBase<OwnedRepr<AG>, IxDyn> =
            ArrayBase::<OwnedRepr<AG>, IxDyn>::zeros(grad_shape);

        // then set the values
        for (i, x) in self.shape().iter().enumerate() {
            for j in 0..*x {
                for (k, _) in self.shape().iter().enumerate() {
                    let mut idx = vec![k; grad.ndim()];
                    idx[i] = j;
                    idx[i + self.ndim()] = j;
                    grad[idx.as_slice()] = <AG as One>::one();
                }
            }
        }

        // convert to static dimension
        grad.into_dimensionality::<DG>().unwrap()
    }
}

// implement PossiblyComplex for ArrayBase<S, D>
impl<A, S, D> PossiblyComplex for ArrayBase<S, D>
where
    D: Dimension,
    A: Clone + PossiblyComplex,
    S: DataOwned<Elem = A>,
{
    fn is_always_real() -> bool {
        A::is_always_real()
    }
}

// implement Conjugate for ArrayBase<OwnedRepr<_>, _>
impl<A, D> Conjugate for ArrayBase<OwnedRepr<A>, D>
where
    D: Dimension,
    A: Clone + Conjugate,
{
    type Output = ArrayBase<OwnedRepr<A::Output>, D>;
    fn conj(&self) -> Self::Output {
        self.mapv(|x| x.conj())
    }
}

// implement Abs for ArrayBase<OwnedRepr<_>, _>
impl<A, D> Abs for ArrayBase<OwnedRepr<A>, D>
where
    D: Dimension,
    A: Clone + Abs,
{
    type Output = ArrayBase<OwnedRepr<A::Output>, D>;
    fn abs(self) -> Self::Output {
        self.mapv(|x| x.abs())
    }
}

// implement AbsSqr for ArrayBase<OwnedRepr<_>, _>
impl<A, D> AbsSqr for ArrayBase<OwnedRepr<A>, D>
where
    D: Dimension,
    A: Clone + AbsSqr,
{
    type Output = ArrayBase<OwnedRepr<A::Output>, D>;
    fn abs_sqr(self) -> Self::Output {
        self.mapv(|x| x.abs_sqr())
    }
}

// implement Arg for ArrayBase<OwnedRepr<_>, _>
impl<A, D> Arg for ArrayBase<OwnedRepr<A>, D>
where
    D: Dimension,
    A: Clone + Arg,
{
    type Output = ArrayBase<OwnedRepr<A::Output>, D>;
    fn arg(self) -> Self::Output {
        self.mapv(|x| x.arg())
    }
}

// implement Signum for ArrayBase<OwnedRepr<_>, _>
impl<A, D> Signum for ArrayBase<OwnedRepr<A>, D>
where
    D: Dimension,
    A: Clone + Signum,
{
    type Output = ArrayBase<OwnedRepr<A::Output>, D>;
    fn signum(self) -> Self::Output {
        self.mapv(|x| x.signum())
    }
}

// implement From<ArrayBase<S, D>> for AutoTuple<(ArrayBase<S, D>,)>
impl<A, S, D> From<ArrayBase<S, D>> for AutoTuple<(ArrayBase<S, D>,)>
where
    D: Dimension,
    S: DataOwned<Elem = A> + RawDataClone,
    A: Clone + PartialEq,
{
    fn from(arr: ArrayBase<S, D>) -> Self {
        AutoTuple::new((arr,))
    }
}

// gradienttype of two arrays is the dimensional sum of the two
impl<AI, DI, AO, DO, AG, DG> GradientType<ArrayBase<OwnedRepr<AO>, DO>>
    for ArrayBase<OwnedRepr<AI>, DI>
where
    DI: Dimension,
    DO: Dimension,
    DG: Dimension,
    DI: DimAdd<DO, Output = DG>,
    AI: GradientType<AO, GradientType = AG>,
{
    type GradientType = ArrayBase<OwnedRepr<AG>, DG>;
}

#[test]
fn test_gradient_type() {
    let a: Array1<f64> = <Array1<f64> as GradientType<Array0<f64>>>::GradientType::zeros(1);
    assert_eq!(a, arr1(&[0.0]));

    let b: AutoTuple<(Array1<f64>, Array2<f64>)> = <<AutoTuple<(Array1<f64>,)> as GradientType<
        AutoTuple<(Array0<f64>, Array1<f64>)>,
    >>::GradientType as Default>::default();

    assert_eq!(
        b,
        AutoTuple::new((
            <Array1<f64> as Default>::default(),
            <Array2<f64> as Default>::default()
        ))
    );
}

// multiplication for df/dx * dx -> df as well as chain rule:
//
// x: ArrayBase<OwnedRepr<AInput>, DInput>
// f(g): ArrayBase<OwnedRepr<AOutput>, DOutput>
// df/dg: ArrayBase<OwnedRepr<ASelf>, DSelf> = Self
// dg/dx: ArrayBase<OwnedRepr<AOtherGrad>, DOtherGrad>
// df/dx: ArrayBase<OwnedRepr<AResult>, DResult>
//
// or
// dx: ArrayBase<OwnedRepr<AOtherGrad>, DOtherGrad>
// df: ArrayBase<OwnedRepr<AResult>, DResult>

impl<
        AI,
        DI, // g input
        AS,
        DS,    // self (grad)
        DG,    // g grad dim
        DR,    // result dim
        MAXGD, // max(DS, DG)
    > ForwardMul<ArrayBase<OwnedRepr<AI>, DI>, ArrayBase<OwnedRepr<AS>, DG>>
    for ArrayBase<OwnedRepr<AS>, DS>
where
    // basic bounds for static operations on dimensions
    DI: Dimension,
    DS: Dimension + DimMax<DG, Output = MAXGD>,
    MAXGD: Dimension + DimAbsSub<DI, Output = DR>,
    DG: Dimension + DimMax<DS, Output = MAXGD>,
    DR: Dimension,

    // constraints on the types of the arrays
    AI: Clone,
    AS: Clone + LinalgScalar,
{
    type ResultGrad = ArrayBase<OwnedRepr<AS>, DR>;
    fn forward_mul(&self, other: &ArrayBase<OwnedRepr<AS>, DG>) -> Self::ResultGrad {
        // better implementation using tensordot
        // df/dx * dg/dx requires the summation over the first DI::NDIM dimensions of df/dx and the last DG::NDIM dimensions of dg/dx
        // this is because the array df/dx[i,j,k,..., a, b, c, ...] is df[a, b, c, ...] / dx[i, j, k, ...]

        let sum_idxs: usize = DI::NDIM.unwrap().try_into().unwrap();

        let lhs = (0usize..sum_idxs).map(|x| Axis(x)).collect::<Vec<_>>();
        let rhs = (other.ndim() - sum_idxs..other.ndim())
            .map(|x| Axis(x))
            .collect::<Vec<_>>();

        let res_dyn: ArrayBase<OwnedRepr<AS>, IxDyn> =
            ndarray_einsum_beta::tensordot(self, other, lhs.as_slice(), rhs.as_slice());

        // convert to static dimension
        let res: ArrayBase<OwnedRepr<AS>, DR> = res_dyn.into_dimensionality::<DR>().unwrap();

        res
    }
}

#[test]
fn test_forward_mul() {
    let a = arr1(&[1.0, 2.0, 3.0]);
    let b = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    // result if the gradient type is 1d,
    // i.e. b represents the gradient of a 1d function of 1d variables and
    // a represents the gradient of a 1d function of 0d variables
    // so db[grad_i, out_i] * da[out_i] = dc[grad_i]
    let c1 = arr1(&[22.0, 28.0]);

    let res: Array1<f64> = <ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> as ForwardMul<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    >>::forward_mul(&b, &a);
    assert_eq!(res, c1);
}
