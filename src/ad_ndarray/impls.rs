use crate::autotuple::AutoTuple;
use crate::traits::{InstOne, InstZero};
use ndarray::{ArrayBase, DataOwned, Dimension, RawDataClone, DimAdd, RemoveAxis, DimMax, OwnedRepr};
use num::traits::{One, Zero};
use std::ops::{Add, Mul};
use crate::forward::ForwardMul;
use crate::gradienttype::GradientType;

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

///// ForwardMul is the multiplication used in the chain rule
///// i.e. d/dx f(g(x)) = df/dx(g(x)).forward_mul( dg/dx(x) )
//impl<AInput, DInput, AOutput, DOutput, AGrad, DGrad, ASelf, DSelf>
//    ForwardMul<
//        ArrayBase<OwnedRepr<AOutput>, DOutput>,
//    > for ArrayBase<OwnedRepr<ASelf>, DSelf>
//where
//    // ensure that the DGrad = DInput + DOutput
//    DInput: Dimension + DimAdd<DOutput, Output = DGrad>,
//    DOutput: Dimension,
//    DGrad: Dimension + RemoveAxis + DimMax<DGrad>,
//    AInput: Clone,
//    AOutput: Clone,
//    AGrad: Clone,
//    AGrad: Clone + Zero + Mul<AGrad, Output = AGrad>,
//    <DGrad as DimMax<DGrad>>::Output: Dimension + RemoveAxis,
//{
//    // gradient should be an array of dimension DGrad = DInput + DOutput
//    type GradientType = 
//
//    fn compose_mul(
//        &self,
//        _x: &ArrayBase<OwnedRepr<AInput>, DInput>,
//        _f_of_g: &ArrayBase<OwnedRepr<AOutput>, DOutput>,
//        dg: &ArrayBase<OwnedRepr<AGrad>, DGrad>,
//    ) -> Self::Output {
//        let oversized_res: ArrayBase<OwnedRepr<ASelf>, <DSelf as DimMax<DGrad>>::Output> =
//            self.clone() * dg.clone();
//
//        // sum over all dimensions except the first N + M
//        let res: Self::Output = sum_over_final_axes::<
//            <DInput as DimAdd<DOutput>>::Output,
//            ASelf,
//            <DSelf as DimMax<DGrad>>::Output,
//        >(oversized_res);
//
//        res
//    }
//}

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
