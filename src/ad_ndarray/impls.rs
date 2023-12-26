use crate::traits::{InstOne, InstZero, ComposedGradMul};
use ndarray::{ArrayBase, DimMax, DataOwned, Dimension, DimAdd, Axis, RemoveAxis, OwnedRepr};
use num::traits::{One, Zero};
use std::ops::{Add, Mul};

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

// ComposedGradMul represents multiplication of the output type of a
// composed function with the gradient of the inner function,
// for arrays, we must sum over all dimensions except the first N + M
// where N is the number of dimensions of the input type and M is the
// number of dimensions of the output type of the composed function.
/*
impl<
    AInput, SInput, DInput,
    AOutput, SOutput, DOutput,
    AGrad, SGrad, DGrad,
    >
    ComposedGradMul<ArrayBase<SInput, DInput>, ArrayBase<SOutput, DOutput>, ArrayBase<SGrad, DGrad>>
    for ArrayBase<SOutput, DOutput>
where
    Self: Clone,
    DInput: Dimension + DimAdd<DOutput>,
    DOutput: Dimension,
    DGrad: Dimension + RemoveAxis,
    SInput: DataOwned<Elem = AInput>,
    SOutput: DataOwned<Elem = AOutput>,
    SGrad: DataOwned<Elem = AGrad>,
    AInput: Clone,
    AOutput: Clone,
    AGrad: Clone + Mul<AOutput> + Zero,
    ArrayBase<SGrad, DGrad>: Clone + Mul<ArrayBase<SOutput, DOutput>, Output = ArrayBase<SGrad, DGrad>>,
{
    type Output = ArrayBase<OwnedRepr<AGrad>, <DInput as DimAdd<DOutput>>::Output>;

    fn compose_mul(
        &self,
        _x: &ArrayBase<SInput, DInput>,
        _f_of_g: &ArrayBase<SOutput, DOutput>,
        dg: &ArrayBase<SGrad, DGrad>,
    ) -> Self::Output {

        let oversized_res = dg.clone() * self.clone();

        // sum over all dimensions except the first N + M
        sum_over_final_axes::<<DInput as DimAdd<DOutput>>::Output, _, _, _>(oversized_res)
    }
}*/
impl<
    AInput, DInput,
    AOutput, DOutput,
    AGrad, DGrad,
    ASelf, DSelf,
    >
    ComposedGradMul<ArrayBase<OwnedRepr<AInput>, DInput>, ArrayBase<OwnedRepr<AOutput>, DOutput>, ArrayBase<OwnedRepr<AGrad>, DGrad>>
    for ArrayBase<OwnedRepr<ASelf>, DSelf>
where
    DInput: Dimension + DimAdd<DOutput>,
    DOutput: Dimension,
    DGrad: Dimension,
    DSelf: Dimension + RemoveAxis + DimMax<DGrad>,
    AInput: Clone,
    AOutput: Clone,
    AGrad: Clone,
    ASelf: Clone + Zero + Mul<AGrad, Output = ASelf>,
    <DSelf as DimMax<DGrad>>::Output: Dimension + RemoveAxis,
{
    type Output = ArrayBase<OwnedRepr<ASelf>, <DInput as DimAdd<DOutput>>::Output>;

    fn compose_mul(
        &self,
        _x: &ArrayBase<OwnedRepr<AInput>, DInput>,
        _f_of_g: &ArrayBase<OwnedRepr<AOutput>, DOutput>,
        dg: &ArrayBase<OwnedRepr<AGrad>, DGrad>,
    ) -> Self::Output {

        let oversized_res: ArrayBase<OwnedRepr<ASelf>, <DSelf as DimMax<DGrad>>::Output> = self.clone() * dg.clone();

        // sum over all dimensions except the first N + M
        let res: Self::Output = sum_over_final_axes::<<DInput as DimAdd<DOutput>>::Output, ASelf, <DSelf as DimMax<DGrad>>::Output>(oversized_res);

        res
    }
}

fn sum_over_final_axes<OutDim, A, D>
    (
        arr: ArrayBase<OwnedRepr<A>, D>
    ) -> ArrayBase<OwnedRepr<A>, OutDim>
    where
    A: Clone + Zero,
    //S: DataOwned<Elem = A>,
    D: Dimension + RemoveAxis,
    OutDim: Dimension,
    ArrayBase<OwnedRepr<A>, D>: Clone,
    {
        let mut a = arr.clone();
        let n = OutDim::NDIM.unwrap().min(a.ndim());

        for i in n..(a.ndim()-1) {
            a.merge_axes(Axis(i), Axis(i+1));
        }
        let a_dyn = a.sum_axis(Axis(a.ndim()-1)).into_shape(&(a.shape()[..n])).unwrap();
        a_dyn.into_dimensionality::<OutDim>().unwrap()
    }

#[test]
fn test_sum_over_final_axes() {
    let a = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let b = sum_over_final_axes::<Ix1, _, _>(a);
    assert_eq!(b, arr1(&[6, 15]));
}
