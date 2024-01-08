use crate::autotuple::AutoTuple;
use crate::traits::{InstOne, InstZero, GradientIdentity};
use ndarray::{ArrayBase, DataOwned, Dimension, RawDataClone, DimAdd, RemoveAxis, DimMax, OwnedRepr, Axis, IxDyn, LinalgScalar};
use num::traits::{One, Zero};
use std::ops::{Add, Mul};
use crate::forward::ForwardMul;
use crate::gradienttype::GradientType;
use crate::ad_ndarray::dimabssub::DimAbsSub;
use ndarray_einsum_beta::einsum;

#[cfg(test)]
use ndarray::{Array0, Array1, Array2, arr1, arr2, Ix1};

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
    fn grad_identity(&self) -> ArrayBase<OwnedRepr<AG>, DG>
    {
        // for an input with shape (a, b, ...) ndim
        // the gradient identity is a tensor with shape (a, b, ..., a, b, ...) 2ndim
        // where g[a, b, ..., z, y, ...] = 1 if a == z && b == y && ... else 0

        // first, we need to get the shape of the gradient
        let grad_shape = self.shape().iter().chain(self.shape().iter()).map(|x| *x).collect::<Vec<_>>();

        // make the gradient
        let mut grad: ArrayBase<OwnedRepr<AG>, IxDyn> = ArrayBase::<OwnedRepr<AG>, IxDyn>::zeros(grad_shape);

        // then set the values
        for (i, x) in self.shape().iter().enumerate()
        {
            for j in 0..*x
            {
                for (k, _) in self.shape().iter().enumerate() {
                    let mut idx = vec![k; grad.ndim()];
                    idx[i] = j;
                    idx[i + self.ndim()] = j;
                    println!("{:?}", idx);
                    grad[idx.as_slice()] = <AG as One>::one();
                }
            }
        }

        // convert to static dimension
        grad.into_dimensionality::<DG>().unwrap()
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

#[test]
fn test_gradient_type() {
    let a: Array1<f64> = <Array1<f64> as GradientType<Array0<f64>>>::GradientType::zeros(1);
    assert_eq!(a, arr1(&[0.0]));

    let b: AutoTuple<(Array1<f64>, Array2<f64>)> = <<AutoTuple<(Array1<f64>,)> as GradientType<AutoTuple<(Array0<f64>, Array1<f64>)>>>::GradientType as Default>::default();

    assert_eq!(b, AutoTuple::new((<Array1<f64> as Default>::default(), <Array2<f64> as Default>::default())));
}

// get einsum string for forward mul
fn get_einsum_str(op1_ndim: u8, op2_ndim: u8, sum_idxs: u8) -> String {
    /// Get einsum string from two operands
    /// op1[...op1_ndim...] op2[...op2_ndim...]
    /// where the first sum_idxs of op1 and the last sum_idxs of op2 are summed over
    /// and the result is ordered by the indices of op2 first, then op1
    /// examples:
    /// op1_ndim = 2, op2_ndim = 3, sum_idxs = 2 => "ab,cab->c" | f[a,b] g[c,a,b] -> h[c]
    /// op1_ndim = 3, op2_ndim = 2, sum_idxs = 1 => "abc,da->dbc" | f[a,b,c] g[d,a] -> h[d,b,c]
    /// op1_ndim = 3, op2_ndim = 2, sum_idxs = 2 => "abc,ab->c" | f[a,b,c] g[a,b] -> h[c]
    /// op1_ndim = 6, op2_ndim = 5, sum_idxs = 3 => "abcdef,ghabc->ghdef" | f[a,b,c,d,e,f] g[g,h,a,b,c] -> h[g,h,d,e,f]

    // assertions
    assert!(op1_ndim >= sum_idxs);
    assert!(op2_ndim >= sum_idxs);
    assert!(sum_idxs <= 26u8);
    assert!(op1_ndim <= 26u8);
    assert!(op2_ndim <= 26u8);

    // the sum indices are the first sum_idxs of the alphabet
    let sum_str = (0u8..sum_idxs).map(|i| (i + 97u8) as char).collect::<String>();
    // the unsummed indices of op1 are the next op1_ndim - sum_idxs of the alphabet
    // i.e. from sum_idxs to sum_idxs + op1_ndim - sum_idxs = op1_ndim
    let op1_str = (sum_idxs..op1_ndim).map(|i| (i + 97u8) as char).collect::<String>();
    // the unsummed indices of op2 are the next op2_ndim - sum_idxs of the alphabet
    // i.e. from op1_ndim to op1_ndim + op2_ndim - sum_idxs
    let op2_str = (op1_ndim..op1_ndim + op2_ndim - sum_idxs).map(|i| (i + 97u8) as char).collect::<String>();

    // the result is then {sum_str}{op1_str},{op2_str}{sum_str} -> {op2_str}{op1_str}
    format!("{}{},{}{}->{}{}", sum_str, op1_str, op2_str, sum_str, op2_str, op1_str)
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
     //AG, DG, // other grad
     DG,
     //AR, DR, // result
     DR,
     MAXGD, // max grad dimension
     SUMD> // number of dimensions to sum over
     ForwardMul<
        ArrayBase<OwnedRepr<AI>, DI>,
        ArrayBase<OwnedRepr<AO>, DO>,
        ArrayBase<OwnedRepr<AS>, DG>,
        ArrayBase<OwnedRepr<AS>, DR>,
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
    //SUMD: DimAdd<MAXGD, Output = DR>,
    //MAXGD: DimAdd<SUMD, Output = DR>,
    DR: DimAbsSub<MAXGD, Output = SUMD>,

    // constraints on the types of the arrays
    AI: Clone,
    AO: Clone,
    AS: Clone + Mul<AS, Output = AS> + LinalgScalar,
    AS: Clone,
    //AR: Clone + Zero,

    // finally ensure multiplication is defined
    ArrayBase<OwnedRepr<AS>, DS>: Mul<ArrayBase<OwnedRepr<AS>, DG>, Output = ArrayBase<OwnedRepr<AS>, MAXGD>>,
{
    fn forward_mul(
        self,
        other: &ArrayBase<OwnedRepr<AS>, DG>,
    ) -> ArrayBase<OwnedRepr<AS>, DR> {

        println!("forward_mul: {:?} {:?}", self.shape(), other.shape());

        // contract over the first SUMD indices of self and the last SUMD indices of other
        // return the result indexed first by the remaining other indices and then the remaining self indices
        // i.e. if self is [a,b,c,d,e,f] and other is [g,h,a,b,c], then
        // self * other -> [g,h,d,e,f]
        let res_dyn: ArrayBase<OwnedRepr<AS>, IxDyn> =
            einsum(&get_einsum_str(self.ndim().try_into().unwrap(), other.ndim().try_into().unwrap(), SUMD::NDIM.unwrap().try_into().unwrap()), &[&self, other]).unwrap();

        println!("res pre-conv: {:?}", res_dyn.shape());
        println!("desired dim: {:?}", DR::NDIM.unwrap());

        // convert to static dimension
        let res: ArrayBase<OwnedRepr<AS>, DR> = res_dyn.into_dimensionality::<DR>().unwrap();

        println!("res: {:?}", res.shape());

        res

        /*
        let oversized_res: ArrayBase<OwnedRepr<AR>, MAXGD> = self.clone().reversed_axes() * other.clone().reversed_axes();

        println!("mul: {:?}", oversized_res.shape());

        // sum over the final SUMD axes
        let res: ArrayBase<OwnedRepr<AR>, DR> = sum_over_final_axes::<
            DR,
            AR,
            MAXGD,
        >(oversized_res);

        res.reversed_axes()
        */
    }
}
/*
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
*/
