use crate::autotuple::AutoTuple;
use crate::traits::{InstOne, InstZero};
use ndarray::{ArrayBase, DataOwned, Dimension, RawDataClone, DimAdd, RemoveAxis, DimMax, OwnedRepr, Axis};
use num::traits::{One, Zero};
use std::ops::{Add, Mul};
use crate::forward::ForwardMul;
use crate::gradienttype::GradientType;
use crate::ad_ndarray::dimsub::DimSub;

#[cfg(test)]
use ndarray::{arr1, arr2, Ix1};

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
impl<AI, DI, AO, DO, AG, DG> GradientType<ArrayBase<OwnedRepr<AO>, DO>> for ArrayBase<OwnedRepr<AI>, DI>
where
    DI: Dimension,
    DO: Dimension,
    DG: Dimension,
    DI: DimAdd<DO, Output = DG>,
    AI: GradientType<AO, GradientType = AG>,
{
    type GradientType = ArrayBase<OwnedRepr<AG>, DG>;
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

impl<AI, DI, // input
     AO, DO, // output
     AS, DS, // self (grad)
     AG, DG, // other grad
     AR, DR, // result
     MAXGD, // max grad dimension
     SUMD> // number of dimensions to sum over
     ForwardMul<
        ArrayBase<OwnedRepr<AI>, DI>,
        ArrayBase<OwnedRepr<AO>, DO>,
        ArrayBase<OwnedRepr<AG>, DG>,
        ArrayBase<OwnedRepr<AR>, DR>,
    > for ArrayBase<OwnedRepr<AS>, DS>
where
    // basic bounds for static operations on dimensions
    DI: Dimension,
    DO: Dimension,
    // MAXGD = max(DG, DS)
    DS: Dimension + DimMax<DG, Output = MAXGD>,
    DG: Dimension,
    DR: Dimension,
    SUMD: Dimension,
    MAXGD: Dimension + RemoveAxis,
    // ensure grad type matches, i.e. Gradient of a function with
    // input Array<AI, DI> and output Array<AO, DO> is Array<AS, DS> or Self
    ArrayBase<OwnedRepr<AI>, DI>: GradientType<ArrayBase<OwnedRepr<AO>, DO>, GradientType = ArrayBase<OwnedRepr<AS>, DS>>,
    // the number of axes to sum over is
    // DR - MAXGD = SUMD
    // and using only DimAdd, we have
    // DR = MAXGD + SUMD
    SUMD: DimAdd<MAXGD, Output = DR>,
    MAXGD: DimAdd<SUMD, Output = DR>,
    DR: DimSub<MAXGD, Output = SUMD>,

    // constraints on the types of the arrays
    AI: Clone,
    AO: Clone,
    AS: Clone + Mul<AG, Output = AR>,
    AG: Clone,
    AR: Clone + Zero,

    // finally ensure multiplication is defined
    ArrayBase<OwnedRepr<AS>, DS>: Mul<ArrayBase<OwnedRepr<AG>, DG>, Output = ArrayBase<OwnedRepr<AR>, MAXGD>>,
{
    fn forward_mul(
        self,
        other: &ArrayBase<OwnedRepr<AG>, DG>,
    ) -> ArrayBase<OwnedRepr<AR>, DR> {
        // multiply self * other
        let oversized_res: ArrayBase<OwnedRepr<AR>, MAXGD> = self.clone() * other.clone();

        // sum over the final SUMD axes
        let res: ArrayBase<OwnedRepr<AR>, DR> = sum_over_final_axes::<
            DR,
            AR,
            MAXGD,
        >(oversized_res);

        res
    }
}

fn sum_over_final_axes<OutDim, A, D>(
    arr: ArrayBase<OwnedRepr<A>, D>,
) -> ArrayBase<OwnedRepr<A>, OutDim>
where
    A: Clone + Zero,
    D: Dimension + RemoveAxis,
    OutDim: Dimension,
    ArrayBase<OwnedRepr<A>, D>: Clone,
{
    let mut a = arr.clone();
    let n = OutDim::NDIM.unwrap().min(a.ndim());

    for i in n..(a.ndim() - 1) {
        a.merge_axes(Axis(i), Axis(i + 1));
    }
    let a_dyn = a
        .sum_axis(Axis(a.ndim() - 1))
        .into_shape(&(a.shape()[..n]))
        .unwrap();
    a_dyn.into_dimensionality::<OutDim>().unwrap()
}

#[test]
fn test_sum_over_final_axes() {
    let a = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let b = sum_over_final_axes::<Ix1, _, _>(a);
    assert_eq!(b, arr1(&[6, 15]));
}
