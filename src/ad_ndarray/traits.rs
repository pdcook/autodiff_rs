use crate::ad_ndarray::dimabssub::DimAbsSub;
use ndarray::{ArrayBase, Axis, Data, Dim, DimAdd, DimMax, Dimension, LinalgScalar, OwnedRepr};
use ndarray_einsum_beta;

pub use ndarray::linalg::Dot; // re-export ndarray::linalg::Dot

/// N-dimensional dot product trait
/// performs contraction of the last axis of the first array and the first axis of the second array
pub trait TensorDot<T> {
    type Output;
    fn tensordot(&self, other: &T) -> Self::Output;
}

impl<
        A,    // element type, must be the same for both arrays and the result
        SS,   // array storage for self
        DS,   // self dimension
        SO,   // other storage
        DO,   // other dimension
        DR,   // result dimension
        DSUM, // sum of self and other dimension
    > TensorDot<ArrayBase<SO, DO>> for ArrayBase<SS, DS>
where
    A: LinalgScalar,
    SS: Data<Elem = A>,
    SO: Data<Elem = A>,
    DS: Dimension + DimAdd<DO, Output = DSUM> + DimMax<Dim<[usize; 1]>, Output = DS>, // DS + DO = DSUM and DS >= 1
    DO: Dimension + DimAdd<DS, Output = DSUM> + DimMax<Dim<[usize; 1]>, Output = DO>, // DO + DS = DSUM and DO >= 1
    DR: Dimension,
    DSUM: Dimension + DimAbsSub<Dim<[usize; 2]>, Output = DR>, // DSUM - 2 = DR
{
    type Output = ArrayBase<OwnedRepr<A>, DR>;

    fn tensordot(&self, other: &ArrayBase<SO, DO>) -> Self::Output {
        ndarray_einsum_beta::tensordot(self, other, &[Axis(self.ndim() - 1)], &[Axis(0)]).into_dimensionality::<DR>().expect(
            "AutoDiff tensordot only supports contraction if the resulting dimension is small enough to be static")
    }
}

/// tensor contraction trait with arbitrary axes
/// using const generics to enforce that the number of axes is known at compile time
pub trait TensorContraction<const N: usize, T> {
    type Output;
    fn contract(&self, other: &T, axes: (&[usize; N], &[usize; N])) -> Self::Output;
}

macro_rules! impl_tensor_contraction {
    ($n:literal) => {
        impl<A, // element type, must be the same for both arrays and the result
             SS, // array storage for self
             DS, // self dimension
             SO, // other storage
             DO, // other dimension
             DR, // result dimension
             DSUBS, // DS minus the contraction axes
             DSUBO, // DO minus the contraction axes
            >
            TensorContraction<$n, ArrayBase<SO, DO>> for ArrayBase<SS, DS>
        where
            A: LinalgScalar,
            SS: Data<Elem = A>,
            SO: Data<Elem = A>,
            DS: Dimension
                + DimMax<Dim<[usize; $n]>, Output = DS> // DS >= N
                + DimAbsSub<Dim<[usize; $n]>, Output = DSUBS>, // DS - N = DSUBS
            DO: Dimension
                + DimMax<Dim<[usize; $n]>, Output = DO> // DO >= N
                + DimAbsSub<Dim<[usize; $n]>, Output = DSUBO>, // DO - N = DSUBO

            DSUBS: Dimension + DimAdd<DSUBO, Output = DR>, // DSUBS + DSUBO = DR
            DSUBO: Dimension + DimAdd<DSUBS, Output = DR>, // DSUBO + DSUBS = DR
            DR: Dimension,
        {
            type Output = ArrayBase<OwnedRepr<A>, DR>;

            fn contract(&self, other: &ArrayBase<SO, DO>, axes: (&[usize; $n], &[usize; $n])) -> Self::Output {
                let lhs_axes = axes.0.map(|i| Axis(i)).into_iter().collect::<Vec<_>>();
                let rhs_axes = axes.1.map(|i| Axis(i)).into_iter().collect::<Vec<_>>();

                ndarray_einsum_beta::tensordot(self, other, lhs_axes.as_slice(), rhs_axes.as_slice()).into_dimensionality::<DR>().expect(
                    "AutoDiff contract only supports contraction if the resulting dimension is small enough to be static")
            }
        }
    };
}

// ndarray only supports Dim<[usize; N]> for N <= 6, everything else is IxDyn
impl_tensor_contraction!(0);
impl_tensor_contraction!(1);
impl_tensor_contraction!(2);
impl_tensor_contraction!(3);
impl_tensor_contraction!(4);
impl_tensor_contraction!(5);
impl_tensor_contraction!(6);

#[test]
fn test_tensor_dot() {
    let a = ndarray::arr2(&[[1, 2], [3, 4]]);
    let b = ndarray::arr2(&[[5, 6], [7, 8]]);
    let c = a.tensordot(&b);
    assert_eq!(c, ndarray::arr2(&[[19, 22], [43, 50]]));
    assert_eq!(c, a.dot(&b));
}

#[test]
fn test_tensor_contraction() {
    let a = ndarray::arr2(&[[1, 2], [3, 4]]);
    let b = ndarray::arr2(&[[5, 6], [7, 8]]);
    let c = a.contract(&b, (&[1], &[0]));
    assert_eq!(c, ndarray::arr2(&[[19, 22], [43, 50]]));
    assert_eq!(c, a.dot(&b));
    let ct = a.contract(&b, (&[0], &[0]));
    assert_eq!(ct, a.t().dot(&b));

    let s = a.contract(&b, (&[0, 1], &[1, 0]));
    assert_eq!(s[()], a.dot(&b).diag().sum());

    let ex = a.contract(&b, (&[], &[]));

    let mut res = ndarray::Array4::zeros((2, 2, 2, 2));
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    res[[i, j, k, l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    assert_eq!(ex, res);
}
